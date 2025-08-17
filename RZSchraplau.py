import streamlit as st
import pandas as pd
import numpy as np
import pulp
from io import BytesIO
import locale
import json

# ── Fortschrittsanzeige ─────────────────────────────
def set_progress(pct: int):
    st.session_state.progress_bar.progress(pct)
    st.session_state.progress_text.markdown(f"**Fortschritt:** {pct}%")

# ── Euro-Format ─────────────────────────────────────
try:
    locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')
    def fmt_euro(x): return locale.currency(x, symbol=True, grouping=True)
except locale.Error:
    def fmt_euro(x): return f"{x:,.2f}".replace(",","X").replace(".",",").replace("X",".") + ' €'

# ── Flexibler Zeitstempel-Parser ────────────────────
def parse_flexible_timestamp(ts_series):
    parsed = []
    for ts in ts_series:
        if pd.isna(ts):
            parsed.append(pd.NaT)
            continue
        s = str(ts).strip()
        try:
            parsed.append(pd.to_datetime(s, dayfirst=True))
        except:
            try:
                if '.' in s and len(s.split('.')[-1].split(' ')[0]) <= 2:
                    parts = s.split('.')
                    y = parts[1].split(' ')
                    if len(y) == 2:
                        yy = int(y)
                        full = 2000 + yy if yy < 50 else 1900 + yy
                        s = s.replace(f".{y} ", f".{full} ")
                parsed.append(pd.to_datetime(s, infer_datetime_format=True))
            except:
                parsed.append(pd.NaT)
    return pd.Series(parsed)

# ── Generischer Daten-Loader ────────────────────────
def load_generic_series(upl, col_name):
    if upl.name.lower().endswith('.csv'):
        df = pd.read_csv(upl, sep=';', decimal=',', usecols=[0,1], header=0)
    else:
        df = pd.read_excel(upl, usecols=[0,1], engine='openpyxl', header=0)
    df.columns = ['Zeitstempel', col_name]
    df['Zeitstempel'] = parse_flexible_timestamp(df['Zeitstempel'])
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
    return df

def load_price_df(upl): return load_generic_series(upl, 'Preis_€/MWh')
def load_pv_df(upl):    return load_generic_series(upl, 'PV_kWh')
def load_ev_df(upl):    return load_generic_series(upl, 'EV_kWh')

# ── Datenvalidierung ────────────────────────────────
def validate_data(p_df, pv_df, ev_df):
    if p_df.empty or pv_df.empty or ev_df.empty:
        return False, 'Leere Datei'
    # Zeitstempel
    for df, name in [(p_df,'Preis'), (pv_df,'PV'), (ev_df,'EV')]:
        if df['Zeitstempel'].isna().any():
            return False, f"{name} Zeitstempel fehlerhaft"
    # Negative Werte nur für PV und EV
    for df, name in [(pv_df,'PV'), (ev_df,'EV')]:
        if (df.iloc[:,1] < 0).any():
            return False, f"Neg. Last in {name} Datei"
    return True, 'OK'

# ── Solver Funktion (Single Batterie) ───────────────
def solve_battery(prices, pv, ev, cfg, grid_kw, interval_h, progress=None):
    T = len(prices)
    batt_max = cfg['bat_kw'] * interval_h
    grid_max = grid_kw * interval_h
    cap, eff, max_cyc = cfg['cap'], cfg['eff_pct']**0.5, cfg['max_cycles']
    m = pulp.LpProblem('BESS', pulp.LpMaximize)
    c = pulp.LpVariable.dicts('c', range(T), cat='Binary')
    d = pulp.LpVariable.dicts('d', range(T), cat='Binary')
    ch = pulp.LpVariable.dicts('ch', range(T), lowBound=0, upBound=batt_max)
    dh = pulp.LpVariable.dicts('dh', range(T), lowBound=0, upBound=batt_max)
    soc = pulp.LpVariable.dicts('soc', range(T), lowBound=0, upBound=cap)
    m += pulp.lpSum(prices[t] * (dh[t] - ch[t]) for t in range(T))
    for t in range(T):
        m += c[t] + d[t] <= 1
        m += ch[t] <= batt_max * c[t]
        m += dh[t] <= batt_max * d[t]
        m += pv[t] + ev[t] + ch[t] - dh[t] <= grid_max
        prev = cfg['start_soc'] if t == 0 else soc[t-1]
        m += soc[t] == prev + eff * ch[t] - dh[t] / eff
        if progress and t % (max(1, T // 50)) == 0:
            progress(5 + int(45 * t / T))
    m += pulp.lpSum((ch[t] + dh[t]) / (2 * cap) for t in range(T)) <= max_cyc
    if progress: progress(50)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=120)
    status = solver.solve(m)
    if progress: progress(90)
    if status != pulp.LpStatusOptimal:
        st.warning(f"Solver Status: {pulp.LpStatus[status]}")
        return 0.0, np.zeros(T), np.zeros(T)
    obj = pulp.value(m.objective) or 0.0
    ch_v = np.array([ch[t].value() or 0 for t in range(T)])
    dh_v = np.array([dh[t].value() or 0 for t in range(T)])
    if progress: progress(100)
    return obj, ch_v, dh_v

# ── Solver Funktion (Multi-Batterie/Szenario) ───────
def solve_joint(prices, pv, ev, cfgs, grid_kw, interval_h, progress=None):
    n, T = len(cfgs), len(prices)
    grid_max = grid_kw * interval_h
    effs = [c['eff_pct']**0.5 for c in cfgs]
    batt_maxs = [c['bat_kw'] * interval_h for c in cfgs]
    caps = [c['cap'] for c in cfgs]
    starts = [c['start_soc'] for c in cfgs]
    max_cyc = [c['max_cycles'] for c in cfgs]
    m = pulp.LpProblem('BESS_Joint', pulp.LpMaximize)
    c_vars, d_vars, ch_vars, dh_vars, soc_vars = {}, {}, {}, {}, {}
    for i in range(n):
        c_vars[i] = pulp.LpVariable.dicts(f'c{i}', range(T), cat='Binary')
        d_vars[i] = pulp.LpVariable.dicts(f'd{i}', range(T), cat='Binary')
        ch_vars[i] = pulp.LpVariable.dicts(f'ch{i}', range(T), lowBound=0, upBound=batt_maxs[i])
        dh_vars[i] = pulp.LpVariable.dicts(f'dh{i}', range(T), lowBound=0, upBound=batt_maxs[i])
        soc_vars[i] = pulp.LpVariable.dicts(f'so{i}', range(T), lowBound=0, upBound=caps[i])
    m += pulp.lpSum(prices[t] * pulp.lpSum(dh_vars[i][t] - ch_vars[i][t] for i in range(n)) for t in range(T))
    for t in range(T):
        for i in range(n):
            m += c_vars[i][t] + d_vars[i][t] <= 1
            m += ch_vars[i][t] <= batt_maxs[i] * c_vars[i][t]
            m += dh_vars[i][t] <= batt_maxs[i] * d_vars[i][t]
            prev = starts[i] if t == 0 else soc_vars[i][t-1]
            m += soc_vars[i][t] == prev + effs[i] * ch_vars[i][t] - dh_vars[i][t] / effs[i]
        total = pulp.lpSum(ch_vars[i][t] - dh_vars[i][t] for i in range(n))
        m += pv[t] + ev[t] + total <= grid_max
        if progress and t % (max(1, T // 50)) == 0:
            progress(5 + int(45 * t / T))
    for i in range(n):
        m += pulp.lpSum((ch_vars[i][t] + dh_vars[i][t]) / (2 * caps[i]) for t in range(T)) <= max_cyc[i]
    if progress: progress(50)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=120)
    status = solver.solve(m)
    if progress: progress(90)
    if status != pulp.LpStatusOptimal:
        st.warning(f"Joint Solver Status: {pulp.LpStatus[status]}")
        return 0.0, [np.zeros(T) for _ in range(n)], [np.zeros(T) for _ in range(n)]
    obj = pulp.value(m.objective) or 0.0
    chs = [np.array([ch_vars[i][t].value() or 0 for t in range(T)]) for i in range(n)]
    dhs = [np.array([dh_vars[i][t].value() or 0 for t in range(T)]) for i in range(n)]
    if progress: progress(100)
    return obj, chs, dhs

# ── Streamlit App (UI & Logik modularisiert) ────────
def main():
    st.set_page_config(layout='wide')
    st.title('BESS: Skalierbar & Szenario-Vergleich')

    # Fortschrittsanzeige initialisieren
    if 'progress_bar' not in st.session_state:
        st.session_state.progress_bar = st.sidebar.progress(0)
        st.session_state.progress_text = st.sidebar.empty()

    tab1, tab2, tab3 = st.tabs(['Konfiguration', 'Upload', 'Simulation'])

    # ── Tab 1: Skalierungsfaktoren Konfiguration ──
    with tab1:
        st.markdown('### Skalierungsfaktoren')
        pv_scale = st.number_input('PV-Skalierung (Anzahl)', 1, 20, 1, step=1)
        ev_scale = st.number_input('EV-Skalierung (Anzahl)', 1, 20, 1, step=1)
        grid_scale = st.number_input('Netzanschluss-Skalierung', 1.0, 5.0, 1.0, step=0.1)
        enable2 = st.checkbox('Zweite Batterie aktivieren', True)
        st.markdown('##### Batterie-Konfigurationen')
        configs = []
        bat_count = 2 if enable2 else 1
        for i in range(bat_count):
            st.markdown(f'**Batterie {i+1}**')
            mode = st.selectbox(f'Modus{i+1}', ['Single Use', 'Multi Use'], key=f'm{i+1}')
            start = st.number_input(f'Start-SoC{i+1} (kWh)', 0.0, 1e6, 0.0, key=f's{i+1}')
            cap = st.number_input(f'Kapazität{i+1} (kWh)', 0.1, 1e6, 4472.0, key=f'c{i+1}')
            cap_scale = st.number_input(f'Kapazität-Skalierung{i+1}', 1, 10, 1, key=f'cscale{i+1}')
            bkw = st.number_input(f'Leistung{i+1} (kW)', 0.1, 1e6, 559.0, key=f'p{i+1}')
            bkw_scale = st.number_input(f'Leistungs-Skalierung{i+1}', 1, 10, 1, key=f'pscale{i+1}')
            eff = st.number_input(f'Wirkungsgrad{i+1} (%)', 1.0, 100.0, 91.0, key=f'e{i+1}') / 100.0
            cyc = st.number_input(f'Max. Zyklen{i+1}', 0.0, 1e4, 548.0, key=f'y{i+1}')
            configs.append({
                'mode': mode,
                'start_soc': start,
                'cap': cap * cap_scale,
                'bat_kw': bkw * bkw_scale,
                'eff_pct': eff,
                'max_cycles': cyc
            })
        grid_kw_input = st.number_input('Netzanschluss (kW)', 0.1, 1e6, 37000.0)
        grid_kw = grid_kw_input * grid_scale
        # Szenarien-Speicherung
        st.button('Konfiguration speichern', on_click=lambda: save_configuration(pv_scale, ev_scale, configs, grid_kw))

    # ── Tab 2: Upload ─────────────────────
    with tab2:
        st.markdown('### Datei-Upload')
        price_file = st.file_uploader('Preisgang [csv/xls/xlsx]', type=['csv','xls','xlsx'])
        pv_file    = st.file_uploader('PV-Lastgang [csv/xls/xlsx]', type=['csv','xls','xlsx'])
        ev_file    = st.file_uploader('EV-Lastgang [csv/xls/xlsx]', type=['csv','xls','xlsx'])

        if price_file and pv_file and ev_file:
            p_df = load_price_df(price_file)
            pv_df = load_pv_df(pv_file)
            ev_df = load_ev_df(ev_file)
            valid, msg = validate_data(p_df, pv_df, ev_df)
            if valid:
                st.success("Dateien erfolgreich geladen und validiert.")
                ts_all = p_df['Zeitstempel']
                st.metric('Zeitpunkte gesamt', f"{len(ts_all)}")
                interval = ts_all.diff().dropna().mode()[0].total_seconds()/3600.0
                st.metric('Intervall (h)', f"{interval:.2f}")
            else:
                st.error(msg)

    # ── Tab 3: Simulation & Ergebnisse ─────
    with tab3:
        st.markdown('### Simulation und Ergebnis-Vergleich')
        price_file = st.sidebar.file_uploader('Preise', type=['csv','xls','xlsx'], key='sidebarpf')
        pv_file    = st.sidebar.file_uploader('PV-Last', type=['csv','xls','xlsx'], key='sidebarpv')
        ev_file    = st.sidebar.file_uploader('EV-Last', type=['csv','xls','xlsx'], key='sidebarev')
        if st.button('▶️ Simulation starten'):
            if not(price_file and pv_file and ev_file):
                st.error('Bitte alle Dateien hochladen.')
            else:
                p_df = load_price_df(price_file)
                pv_df = load_pv_df(pv_file)
                ev_df = load_ev_df(ev_file)
                valid, msg = validate_data(p_df, pv_df, ev_df)
                if not valid:
                    st.error(msg)
                    st.stop()
                ts = p_df['Zeitstempel']
                prices = p_df['Preis_€/MWh'].to_numpy() / 1000.0
                pv = pv_df['PV_kWh'].to_numpy() * pv_scale
                ev = ev_df['EV_kWh'].to_numpy() * ev_scale
                interval_h = ts.diff().dropna().mode()[0].total_seconds() / 3600.0
                st.info(f'Intervall: {interval_h:.2f} h')
                # Optimierung
                free_results = [solve_battery(prices, pv, ev, cfg, grid_kw, interval_h, set_progress) for cfg in configs]
                obj_joint, chs_joint, dhs_joint = solve_joint(prices, pv, ev, configs, grid_kw, interval_h, set_progress)
                # Einzeloptimierung anzeigen
                st.subheader('Einzeloptimierung')
                tot_free = sum(obj for obj, *_ in free_results)
                for idx,(cfg,(obj,*,*)) in enumerate(zip(configs, free_results), start=1):
                    st.metric(f"B{idx} ({cfg['mode']})", fmt_euro(obj))
                st.metric('Gesamt Free', fmt_euro(tot_free))
                # Joint
                st.subheader('Gemeinsam')
                st.metric('Gesamterlös', fmt_euro(obj_joint))
                for idx in range(len(configs)):
                    share = float(np.dot(prices, dhs_joint[idx] - chs_joint[idx]))
                    st.metric(f"B{idx+1} Anteil", fmt_euro(share))
                # Vergleich
                delta = obj_joint - tot_free
                pct = (delta/abs(tot_free)*100) if tot_free!=0 else 0
                st.subheader('Vergleich')
                d1,d2,d3 = st.columns(3)
                d1.metric('Δ absolut', fmt_euro(delta))
                d2.metric('Δ %', f"{pct:.2f}%")
                net_load = pv + ev + sum(chs_joint) - sum(dhs_joint)
                util = np.mean(net_load/(grid_kw*interval_h))*100
                d3.metric('Netzauslastung', f"{util:.1f}%")
                viol = (net_load > grid_kw*interval_h).sum()
                if viol > 0:
                    st.warning(f"⚠️ {viol} Überlastungen im Netz")
                # Ergebnisse-Tabelle & Download
                st.subheader('Ergebnisse / Export')
                out = pd.DataFrame({
                    'Zeit': ts,
                    'Preis_€/kWh': prices,
                    'PV_kWh': pv,
                    'EV_kWh': ev
                })
                for idx in range(len(configs)):
                    out[f'B{idx+1}_Laden_kWh'] = chs_joint[idx]
                    out[f'B{idx+1}_Entladen_kWh'] = dhs_joint[idx]
                out['Netzlast_kWh'] = net_load
                out['Netzlast_%'] = net_load/(grid_kw*interval_h)*100
                st.dataframe(out)
                buf = BytesIO()
                out.to_excel(buf, index=False, engine='openpyxl')
                buf.seek(0)
                st.download_button(
                    '📥 Excel-Export', buf,
                    file_name=f"res_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

def save_configuration(pv_scale, ev_scale, configs, grid_kw):
    config = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'pv_scale': pv_scale,
        'ev_scale': ev_scale,
        'batteries': configs,
        'grid_kw': grid_kw
    }
    json_str = json.dumps(config, indent=2)
    st.download_button(
        '💾 Konfiguration als JSON speichern',
        json_str,
        file_name=f"BESS_Config_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
        mime='application/json'
    )

if __name__ == '__main__':
    main()
