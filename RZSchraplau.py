import streamlit as st
import pandas as pd
import numpy as np
import pulp
from io import BytesIO
import locale
import json

# ─────────────────────────────────────────────────────────────
# Single-Battery EV-First + After-Trade (2-Pass) – Streamlit App
# Stand: 18.08.2025
# Logik:
#   Pass 1 (EV-Priorität hinter dem Zähler):
#     PV → EV zuerst (deterministisch), dann
#     PV-Überschuss → Batterie laden (ch_pv) bis Power/SoC, dann
#     Batterie → EV (dh_ev) bis Power/SoC, Rest EV → Netz (grid_ev), Rest PV → Export (pv_grid), optional Curtailment.
#     Ziel: Minimierung des Netzbezugs für EV (min Σ grid_ev).
#   Pass 2 (Handel am Netzanschluss mit Rest-Kapazität):
#     Fixiere Pass-1-Flüsse (ch_pv, dh_ev, pv_grid, grid_ev). Erlaube zusätzlich:
#     ch_grid (Netz→Batt), dh_grid (Batt→Netz), mit Power/SoC/Cycle/Netzlimits.
#     Ziel: max Σ price·(dh_grid − ch_grid). Terminal-SoC optional = Start-SoC.
# ─────────────────────────────────────────────────────────────

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
        except Exception:
            try:
                if '.' in s and len(s.split('.')[-1].split(' ')[0]) <= 2:
                    parts = s.split('.')
                    if len(parts) >= 2:
                        y = parts[1].split(' ')
                        if len(y) >= 1:
                            yy = int(y[0])
                            full = 2000 + yy if yy < 50 else 1900 + yy
                            s = s.replace(f".{y[0]} ", f".{full} ")
                parsed.append(pd.to_datetime(s, infer_datetime_format=True))
            except Exception:
                parsed.append(pd.NaT)
    return pd.Series(parsed)

# ── Generischer Daten-Loader ────────────────────────
def load_generic_series(upl, col_name):
    try:
        upl.seek(0)  # Filepointer immer zurücksetzen
        if upl.name.lower().endswith('.csv'):
            try:
                df = pd.read_csv(upl, sep=';', decimal=',', usecols=[0,1], header=0)
            except Exception:
                upl.seek(0)
                df = pd.read_csv(upl, sep=',', decimal='.', usecols=[0,1], header=0)
        else:
            upl.seek(0)
            df = pd.read_excel(upl, usecols=[0,1], engine='openpyxl', header=0)
        if len(df.columns) < 2:
            raise ValueError(f"Datei muss mindestens 2 Spalten haben, gefunden: {len(df.columns)}")
        df.columns = ['Zeitstempel', col_name]
        df['Zeitstempel'] = parse_flexible_timestamp(df['Zeitstempel'])
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
        if df.empty or df[col_name].isna().all():
            raise ValueError(f"Keine gültigen Daten in {col_name} gefunden")
        return df
    except Exception as e:
        st.error(f"Fehler beim Laden der Datei {upl.name}: {str(e)}")
        return pd.DataFrame()

def load_price_df(upl): return load_generic_series(upl, 'Preis_€/MWh')
def load_pv_df(upl):    return load_generic_series(upl, 'PV_kWh')
def load_ev_df(upl):    return load_generic_series(upl, 'EV_kWh')

# ── Datenvalidierung ────────────────────────────────
def validate_data(p_df, pv_df, ev_df):
    for df, name in [(p_df,'Preis'), (pv_df,'PV'), (ev_df,'EV')]:
        if df.empty:
            return False, f'{name}-Datei ist leer oder konnte nicht geladen werden'
    lengths = [len(p_df), len(pv_df), len(ev_df)]
    if not all(l == lengths[0] for l in lengths):
        return False, f'Unterschiedliche Datenlängen: Preis={lengths[0]}, PV={lengths[1]}, EV={lengths[2]}'
    for df, name in [(p_df,'Preis'), (pv_df,'PV'), (ev_df,'EV')]:
        if df['Zeitstempel'].isna().any():
            return False, f"{name}: Zeitstempel fehlerhaft oder unvollständig"
        if len(df['Zeitstempel'].unique()) != len(df):
            return False, f"{name}: Doppelte Zeitstempel gefunden"
    if (p_df['Preis_€/MWh'] < 0).any():
        st.warning("⚠️ Negative Preise in den Daten gefunden (bei viel EE normal)")
    if (p_df['Preis_€/MWh'] > 1000).any():
        st.warning("⚠️ Sehr hohe Preise (>1000 €/MWh) gefunden")
    return True, 'Datenvalidierung erfolgreich'

# ─────────────────────────────────────────────────────────────
# Solver – Pass 1: EV-Priorität hinter dem Zähler
# Minimiert Netzbezug (grid_ev), respektiert Exportlimit, erlaubt PV-Curtailment
# Variablen: ch_pv, dh_ev, pv_grid, grid_ev, pv_curt, c, d, soc
# ─────────────────────────────────────────────────────────────

def solve_pass1_ev_priority(pv, ev, cfg, grid_kw, interval_h, progress=None):
    T = len(pv)
    cap = cfg['cap']
    eff = cfg['eff_pct'] ** 0.5
    pmax = cfg['bat_kw'] * interval_h  # kWh pro Schritt
    gmax = grid_kw * interval_h        # kWh pro Schritt

    # Vorrangaufteilung fix (Parameter): PV→EV zuerst
    pv_ev = np.minimum(pv, ev)
    pv_sur = pv - pv_ev
    ev_res = ev - pv_ev

    m = pulp.LpProblem('EV_FIRST', pulp.LpMinimize)

    # Variablen
    c  = pulp.LpVariable.dicts('c',  range(T), cat='Binary')
    d  = pulp.LpVariable.dicts('d',  range(T), cat='Binary')
    ch_pv  = pulp.LpVariable.dicts('ch_pv',  range(T), lowBound=0, upBound=pmax)
    dh_ev  = pulp.LpVariable.dicts('dh_ev',  range(T), lowBound=0, upBound=pmax)
    pv_grid= pulp.LpVariable.dicts('pv_grid',range(T), lowBound=0, upBound=gmax)
    grid_ev= pulp.LpVariable.dicts('grid_ev',range(T), lowBound=0, upBound=gmax)
    pv_curt= pulp.LpVariable.dicts('pv_curt',range(T), lowBound=0)  # optional, verhindert Infeasible
    soc    = pulp.LpVariable.dicts('soc',    range(T), lowBound=0, upBound=cap)

    # Ziel 1: Minimiere Netzbezug für EV
    m += pulp.lpSum(grid_ev[t] for t in range(T))

    for t in range(T):
        # Power + keine Gleichzeitigkeit
        m += c[t] + d[t] <= 1
        m += ch_pv[t] <= pmax * c[t]
        m += dh_ev[t] <= pmax * d[t]

        # Splits: PV-Überschuss & EV-Rest
        m += ch_pv[t] + pv_grid[t] + pv_curt[t] == float(pv_sur[t])
        m += dh_ev[t] + grid_ev[t] == float(ev_res[t])

        # SoC-Dynamik (nur PV-Ladung & EV-Entladung)
        prev = cfg['start_soc'] if t == 0 else soc[t-1]
        m += soc[t] == prev + eff * ch_pv[t] - dh_ev[t] / eff

        if progress and t % max(1, T//50) == 0:
            progress(5 + int(35*t/T))

    # Zyklenbudget
    m += pulp.lpSum((ch_pv[t] + dh_ev[t]) / (2 * cap) for t in range(T)) <= cfg['max_cycles']

    if progress: progress(40)
    status = pulp.PULP_CBC_CMD(msg=False, timeLimit=300).solve(m)
    if progress: progress(60)

    if status != pulp.LpStatusOptimal:
        return {'status': pulp.LpStatus[status]}

    to_arr = lambda X: np.array([X[t].value() or 0.0 for t in range(T)])
    ch_pv_v  = to_arr(ch_pv)
    dh_ev_v  = to_arr(dh_ev)
    pv_grid_v= to_arr(pv_grid)
    grid_ev_v= to_arr(grid_ev)
    pv_curt_v= to_arr(pv_curt)
    soc_v    = to_arr(soc)

    cycles1 = ((ch_pv_v + dh_ev_v) / (2*cap)).sum() if cap > 0 else 0.0

    return {
        'status': 'Optimal',
        'pv_ev': pv_ev,
        'pv_sur': pv_sur,
        'ev_res': ev_res,
        'ch_pv': ch_pv_v,
        'dh_ev': dh_ev_v,
        'pv_grid': pv_grid_v,
        'grid_ev': grid_ev_v,
        'pv_curt': pv_curt_v,
        'soc': soc_v,
        'cycles1': cycles1,
        'obj_ev_min_grid': grid_ev_v.sum()
    }

# ─────────────────────────────────────────────────────────────
# Solver – Pass 2: Handel (zusätzliche Grid↔Batt Flüsse, EV-Priorität bleibt fix)
# Variablen: ch_grid, dh_grid, c2, d2, soc2
# Limits: (ch_pv+ch_grid) ≤ pmax*c2, (dh_ev+dh_grid) ≤ pmax*d2, c2+d2≤1
#         grid_ev + ch_grid ≤ gmax, pv_grid + dh_grid ≤ gmax
# Terminal-SoC optional = Start-SoC
# ─────────────────────────────────────────────────────────────

def solve_pass2_trade(prices, pass1, cfg, grid_kw, interval_h, terminal_soc_equal=True, progress=None):
    T = len(prices)
    cap = cfg['cap']
    eff = cfg['eff_pct'] ** 0.5
    pmax = cfg['bat_kw'] * interval_h
    gmax = grid_kw * interval_h

    ch_pv = pass1['ch_pv']
    dh_ev = pass1['dh_ev']
    pv_grid = pass1['pv_grid']
    grid_ev = pass1['grid_ev']

    # Verbleibendes Zyklenbudget
    remaining_cycles = max(cfg['max_cycles'] - pass1['cycles1'], 0.0)

    m = pulp.LpProblem('TRADE_AFTER_EV', pulp.LpMaximize)

    # Variablen
    c2  = pulp.LpVariable.dicts('c2',  range(T), cat='Binary')
    d2  = pulp.LpVariable.dicts('d2',  range(T), cat='Binary')
    chg = pulp.LpVariable.dicts('ch_grid', range(T), lowBound=0, upBound=pmax)  # Netz→Batt
    dhg = pulp.LpVariable.dicts('dh_grid', range(T), lowBound=0, upBound=pmax)  # Batt→Netz
    soc2= pulp.LpVariable.dicts('soc2',    range(T), lowBound=0, upBound=cap)

    # Ziel 2: Handelsgewinn
    m += pulp.lpSum(prices[t] * (dhg[t] - chg[t]) for t in range(T))

    for t in range(T):
        # Keine Gleichzeitigkeit (bezogen auf Gesamtleistung)
        m += c2[t] + d2[t] <= 1
        m += ch_pv[t] + chg[t] <= pmax * c2[t]
        m += dh_ev[t] + dhg[t] <= pmax * d2[t]

        # Grid-Limits je Richtung (Totals)
        m += grid_ev[t] + chg[t] <= gmax     # Import gesamt
        m += pv_grid[t] + dhg[t] <= gmax     # Export gesamt

        # SoC-Dynamik (alle Flüsse)
        prev = cfg['start_soc'] if t == 0 else soc2[t-1]
        m += soc2[t] == prev + eff * (ch_pv[t] + chg[t]) - (dh_ev[t] + dhg[t]) / eff

        if progress and t % max(1, T//50) == 0:
            progress(60 + int(35*t/T))

    # Terminal-SoC optional fest auf Start
    if terminal_soc_equal:
        m += soc2[T-1] == cfg['start_soc']

    # Rest-Zyklenbudget nur für Handelsflüsse
    if cap > 0:
        m += pulp.lpSum((chg[t] + dhg[t]) / (2 * cap) for t in range(T)) <= remaining_cycles + 1e-9

    if progress: progress(95)
    status = pulp.PULP_CBC_CMD(msg=False, timeLimit=300).solve(m)

    if status != pulp.LpStatusOptimal:
        return {'status': pulp.LpStatus[status]}

    to_arr = lambda X: np.array([X[t].value() or 0.0 for t in range(T)])
    chg_v = to_arr(chg)
    dhg_v = to_arr(dhg)
    soc2_v= to_arr(soc2)

    return {
        'status': 'Optimal',
        'obj_trade': float(pulp.value(m.objective) or 0.0),
        'ch_grid': chg_v,
        'dh_grid': dhg_v,
        'soc': soc2_v,
        'cycles2': ((chg_v + dhg_v)/(2*cap)).sum() if cap>0 else 0.0
    }

# ── Session State Initialisierung ───────────────────
def init_session_state():
    if 'progress_bar' not in st.session_state:
        st.session_state.progress_bar = st.sidebar.progress(0)
        st.session_state.progress_text = st.sidebar.empty()
    if 'price_file' not in st.session_state:
        st.session_state.price_file = None
    if 'pv_file' not in st.session_state:
        st.session_state.pv_file = None
    if 'ev_file' not in st.session_state:
        st.session_state.ev_file = None

# ── Streamlit App ───────────────────────────────────
def main():
    st.set_page_config(layout='wide')
    st.title('BESS – Single Battery · EV-First + After-Trade')

    init_session_state()

    tab1, tab2, tab3 = st.tabs(['Konfiguration', 'Upload', 'Simulation'])

    # ── Tab 1: Konfiguration ────────────────────────
    with tab1:
        st.markdown('### Batterie & Netz')
        start_soc = st.number_input('Start-SoC (kWh)', 0.0, 1e9, 0.0)
        cap       = st.number_input('Kapazität (kWh)', 0.1, 1e9, 4472.0)
        bat_kw    = st.number_input('Leistung (kW)', 0.1, 1e9, 559.0)
        eff_pct   = st.number_input('Wirkungsgrad (Roundtrip) %', 1.0, 100.0, 91.0) / 100.0
        max_cycles= st.number_input('Max. Zyklen (im Zeitraum)', 0.0, 1e6, 548.0)
        grid_kw   = st.number_input('Netzanschluss Limit (kW, Import & Export je Richtung)', 0.1, 1e9, 37000.0)
        terminal_soc_equal = st.checkbox('Terminal-SoC = Start-SoC (empfohlen für Handel)', True)

        st.markdown('### Skalierung')
        pv_scale = st.number_input('PV-Skalierung (Anzahl)', 1, 50, 1, step=1)
        ev_scale = st.number_input('EV-Skalierung (Anzahl)', 1, 50, 1, step=1)

        cfg = {
            'start_soc': start_soc,
            'cap': cap,
            'bat_kw': bat_kw,
            'eff_pct': eff_pct,
            'max_cycles': max_cycles
        }

        st.button('Konfiguration speichern', on_click=lambda: save_configuration(pv_scale, ev_scale, cfg, grid_kw, terminal_soc_equal))

    # ── Tab 2: Upload ───────────────────────────────
    with tab2:
        st.markdown('### Datei-Upload')
        st.info('📁 Preis, PV und EV laden. Diese stehen in Tab 3 zur Verfügung.')
        price_file = st.file_uploader('Preisgang [csv/xls/xlsx] (Zeit, €/MWh)', type=['csv','xls','xlsx'])
        if price_file:
            st.session_state.price_file = price_file
        pv_file = st.file_uploader('PV-Lastgang [csv/xls/xlsx] (Zeit, kWh)', type=['csv','xls','xlsx'])
        if pv_file:
            st.session_state.pv_file = pv_file
        ev_file = st.file_uploader('EV-Lastgang [csv/xls/xlsx] (Zeit, kWh)', type=['csv','xls','xlsx'])
        if ev_file:
            st.session_state.ev_file = ev_file

        if st.session_state.price_file and st.session_state.pv_file and st.session_state.ev_file:
            try:
                p_df = load_price_df(st.session_state.price_file)
                pv_df = load_pv_df(st.session_state.pv_file)
                ev_df = load_ev_df(st.session_state.ev_file)
                valid, msg = validate_data(p_df, pv_df, ev_df)
                if valid:
                    st.success('✅ Dateien erfolgreich geladen und validiert.')
                    ts_all = p_df['Zeitstempel']
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric('Zeitpunkte', f"{len(ts_all)}")
                    with c2:
                        try:
                            interval = ts_all.diff().dropna().mode()[0].total_seconds()/3600.0
                            st.metric('Intervall (h)', f"{interval:.2f}")
                        except Exception:
                            st.metric('Intervall (h)', 'Unbekannt')
                    with c3:
                        st.metric('Zeitraum', f"{ts_all.min():%d.%m.%Y} – {ts_all.max():%d.%m.%Y}")
                    st.markdown('##### Vorschau (5 Zeilen)')
                    st.dataframe(pd.DataFrame({
                        'Zeit': ts_all.head(),
                        'Preis (€/MWh)': p_df['Preis_€/MWh'].head(),
                        'PV (kWh)': pv_df['PV_kWh'].head(),
                        'EV (kWh)': ev_df['EV_kWh'].head()
                    }))
                else:
                    st.error(f"❌ Validierungsfehler: {msg}")
            except Exception as e:
                st.error(f"❌ Fehler beim Laden der Dateien: {e}")

    # ── Tab 3: Simulation ───────────────────────────
    with tab3:
        st.markdown('### Simulation')
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f"**Preisdaten:** {'✅' if st.session_state.price_file else '❌'}")
        with c2: st.markdown(f"**PV-Daten:** {'✅' if st.session_state.pv_file else '❌'}")
        with c3: st.markdown(f"**EV-Daten:** {'✅' if st.session_state.ev_file else '❌'}")

        ready = all([st.session_state.price_file, st.session_state.pv_file, st.session_state.ev_file])
        if st.button('▶️ Simulation starten', disabled=not ready):
            try:
                # Daten laden
                p_df = load_price_df(st.session_state.price_file)
                pv_df = load_pv_df(st.session_state.pv_file)
                ev_df = load_ev_df(st.session_state.ev_file)

                valid, msg = validate_data(p_df, pv_df, ev_df)
                if not valid:
                    st.error(f"❌ Validierungsfehler: {msg}")
                    st.stop()

                ts = p_df['Zeitstempel']
                prices = p_df['Preis_€/MWh'].to_numpy() / 1000.0  # €/kWh
                pv = pv_df['PV_kWh'].to_numpy() * pv_scale
                ev = ev_df['EV_kWh'].to_numpy() * ev_scale

                if len(prices) != len(pv) or len(pv) != len(ev):
                    st.error('❌ Datenreihen haben unterschiedliche Längen')
                    st.stop()

                try:
                    interval_h = ts.diff().dropna().mode()[0].total_seconds()/3600.0
                    if not (0 < interval_h <= 24):
                        raise ValueError
                except Exception:
                    interval_h = 1.0
                    st.warning('⚠️ Intervall unbekannt – setze 1h')

                st.info(f'📊 Optimierung für {len(prices)} Schritte @ {interval_h:.2f}h')

                # Pass 1: EV-Priorität
                res1 = solve_pass1_ev_priority(pv, ev, cfg, grid_kw, interval_h, set_progress)
                if res1.get('status') != 'Optimal':
                    st.error(f"Pass 1 nicht optimal: {res1.get('status')}")
                    st.stop()

                # Pass 2: Handel
                res2 = solve_pass2_trade(prices, res1, cfg, grid_kw, interval_h, terminal_soc_equal, set_progress)
                if res2.get('status') != 'Optimal':
                    st.error(f"Pass 2 nicht optimal: {res2.get('status')}")
                    st.stop()

                # Aggregation Flüsse
                pv_ev      = res1['pv_ev']
                pv_sur     = res1['pv_sur']
                ev_res     = res1['ev_res']
                ch_pv      = res1['ch_pv']
                dh_ev      = res1['dh_ev']
                pv_grid    = res1['pv_grid']
                grid_ev    = res1['grid_ev']
                pv_curt    = res1['pv_curt']
                soc_pass2  = res2['soc']
                ch_grid    = res2['ch_grid']
                dh_grid    = res2['dh_grid']

                # Totalströme
                ch_total = ch_pv + ch_grid
                dh_total = dh_ev + dh_grid
                gimp_total = grid_ev + ch_grid
                gexp_total = pv_grid + dh_grid

                # Kennzahlen
                ev_total = ev.sum()
                pv_total = pv.sum()
                ev_from_pv = pv_ev.sum()
                ev_from_batt = dh_ev.sum()
                ev_from_grid = grid_ev.sum()
                autarkie_ev = ((ev_from_pv + ev_from_batt) / ev_total * 100) if ev_total>0 else 0.0

                pv_to_batt = ch_pv.sum()
                pv_export  = pv_grid.sum()
                pv_curt_tot= pv_curt.sum()
                pv_ev_share= (ev_from_pv / pv_total * 100) if pv_total>0 else 0.0
                pv_selfuse = ev_from_pv + ch_pv.sum()  # Näherung (ohne exakte Quelle ch_grid vs ch_pv im SoC)
                pv_selfuse_q = (pv_selfuse / pv_total * 100) if pv_total>0 else 0.0

                revenue_trade = res2['obj_trade']
                revenue_total = float(np.dot(prices, (gexp_total - gimp_total)))

                # Metriken
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1: st.metric('EV-Autarkie', f"{autarkie_ev:.1f}%")
                with m2: st.metric('PV→EV (MWh)', f"{ev_from_pv/1000:.1f}")
                with m3: st.metric('Batt→EV (MWh)', f"{ev_from_batt/1000:.1f}")
                with m4: st.metric('EV aus Netz (MWh)', f"{ev_from_grid/1000:.1f}")
                with m5: st.metric('PV-Export (MWh)', f"{pv_export/1000:.1f}")

                m6, m7, m8, m9, m10 = st.columns(5)
                with m6: st.metric('PV→Batt (MWh)', f"{pv_to_batt/1000:.1f}")
                with m7: st.metric('PV-Curtail (MWh)', f"{pv_curt_tot/1000:.1f}")
                with m8: st.metric('PV-EV-Anteil', f"{pv_ev_share:.1f}%")
                with m9: st.metric('PV-Eigenverbrauchsquote', f"{pv_selfuse_q:.1f}%")
                with m10: st.metric('Handelsgewinn (nur Batt)', fmt_euro(revenue_trade))

                m11, m12 = st.columns(2)
                with m11: st.metric('Gesamtsaldo Netz (inkl. PV)', fmt_euro(revenue_total))
                imp_util = (gimp_total/(grid_kw*interval_h)).mean()*100 if grid_kw>0 else 0
                exp_util = (gexp_total/(grid_kw*interval_h)).mean()*100 if grid_kw>0 else 0
                with m12: st.metric('Import/Export-Ø-Auslastung', f"{imp_util:.1f}% / {exp_util:.1f}%")

                # Zeitreihen-DF
                results_df = pd.DataFrame({
                    'Zeit': ts,
                    'Preis_€/kWh': prices,
                    'PV_kWh': pv,
                    'EV_kWh': ev,
                    'PV→EV_kWh': pv_ev,
                    'EV_Rest_kWh': ev_res,
                    'PV_Überschuss_kWh': pv_sur,
                    'ch_pv_kWh': ch_pv,
                    'dh_ev_kWh': dh_ev,
                    'PV_Export_kWh': pv_grid,
                    'EV_aus_Netz_kWh': grid_ev,
                    'ch_grid_kWh': ch_grid,
                    'dh_grid_kWh': dh_grid,
                    'Import_total_kWh': gimp_total,
                    'Export_total_kWh': gexp_total,
                    'SoC_kWh': soc_pass2
                })

                st.markdown('#### Zeitreihen (Ausschnitt)')
                st.dataframe(results_df.head(200))

                # Quartalsaggregation (Import/Export)
                results_df['Quartal'] = results_df['Zeit'].dt.to_period('Q')
                q = results_df.groupby('Quartal').agg({
                    'Import_total_kWh': 'sum',
                    'Export_total_kWh': 'sum'
                }).reset_index()
                if not q.empty:
                    q['Quartal_str'] = q['Quartal'].astype(str)
                    st.markdown('#### Quartalsbilanz Import/Export (MWh)')
                    chart_df = q.set_index('Quartal_str')/1000.0
                    st.bar_chart(chart_df[['Import_total_kWh','Export_total_kWh']], height=300)

                # Export
                st.markdown('#### Export')
                buf = BytesIO()
                results_df.to_excel(buf, index=False, engine='openpyxl')
                buf.seek(0)
                st.download_button('📥 Excel-Export', buf,
                    file_name=f"BESS_Single_EVFirst_{pd.Timestamp.now():%Y%m%d_%H%M%S}.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

            except Exception as e:
                st.error(f"❌ Fehler bei der Simulation: {e}")
                st.exception(e)

# ── Konfiguration speichern ─────────────────────────
def save_configuration(pv_scale, ev_scale, cfg, grid_kw, terminal_soc_equal):
    config = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'pv_scale': pv_scale,
        'ev_scale': ev_scale,
        'battery': cfg,
        'grid_kw': grid_kw,
        'terminal_soc_equal': terminal_soc_equal
    }
    json_str = json.dumps(config, indent=2)
    st.download_button('💾 Konfiguration als JSON speichern', json_str,
        file_name=f"BESS_Single_Config_{pd.Timestamp.now():%Y%m%d_%H%M}.json", mime='application/json')

if __name__ == '__main__':
    main()
