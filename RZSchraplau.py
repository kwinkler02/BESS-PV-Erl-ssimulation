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
        upl.seek(0)  # Wichtig: Filepointer immer zurücksetzen
        if upl.name.lower().endswith('.csv'):
            # Versuche verschiedene CSV-Formate
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
    # Prüfe auf leere DataFrames
    for df, name in [(p_df,'Preis'), (pv_df,'PV'), (ev_df,'EV')]:
        if df.empty:
            return False, f'{name}-Datei ist leer oder konnte nicht geladen werden'
    
    # Prüfe gleiche Anzahl von Zeitpunkten
    lengths = [len(p_df), len(pv_df), len(ev_df)]
    if not all(l == lengths[0] for l in lengths):
        return False, f'Unterschiedliche Datenlängen: Preis={lengths[0]}, PV={lengths[1]}, EV={lengths[2]}'
    
    # Zeitstempel-Validierung
    for df, name in [(p_df,'Preis'), (pv_df,'PV'), (ev_df,'EV')]:
        if df['Zeitstempel'].isna().any():
            return False, f"{name}: Zeitstempel fehlerhaft oder unvollständig"
        if len(df['Zeitstempel'].unique()) != len(df):
            return False, f"{name}: Doppelte Zeitstempel gefunden"
    
    # Negative Preise (Warnung, aber nicht blockierend)
    if (p_df['Preis_€/MWh'] < 0).any():
        st.warning("⚠️ Negative Preise in den Daten gefunden (bei viel EE normal)")
    
    # Auffällig hohe Preise
    if (p_df['Preis_€/MWh'] > 1000).any():
        st.warning("⚠️ Sehr hohe Preise (>1000 €/MWh) gefunden")
    
    return True, 'Datenvalidierung erfolgreich'

# ── Solver Funktion (Single Batterie) ───────────────
def solve_battery(prices, pv, ev, cfg, grid_kw, interval_h, progress=None):
    T = len(prices)
    batt_max = cfg['bat_kw'] * interval_h
    grid_max = grid_kw * interval_h
    cap, eff, max_cyc = cfg['cap'], cfg['eff_pct']**0.5, cfg['max_cycles']

    m = pulp.LpProblem('BESS', pulp.LpMaximize)

    # Variablen
    c = pulp.LpVariable.dicts('c', range(T), cat='Binary')  # charge aktiv?
    d = pulp.LpVariable.dicts('d', range(T), cat='Binary')  # discharge aktiv?
    ch = pulp.LpVariable.dicts('ch', range(T), lowBound=0, upBound=batt_max)  # kWh
    dh = pulp.LpVariable.dicts('dh', range(T), lowBound=0, upBound=batt_max)  # kWh
    soc = pulp.LpVariable.dicts('soc', range(T), lowBound=0, upBound=cap)

    # Ziel: Preis * (Entladung - Ladung)
    m += pulp.lpSum(prices[t] * (dh[t] - ch[t]) for t in range(T))

    for t in range(T):
        # Nicht gleichzeitig laden & entladen
        m += c[t] + d[t] <= 1
        m += ch[t] <= batt_max * c[t]
        m += dh[t] <= batt_max * d[t]

        # Netzlimit (symmetrisch), Import positiv
        net = ev[t] + ch[t] - pv[t] - dh[t]
        m +=  net <= grid_max
        m += -net <= grid_max

        # SoC-Dynamik
        prev = cfg['start_soc'] if t == 0 else soc[t-1]
        m += soc[t] == prev + eff * ch[t] - dh[t] / eff

        if progress and t % (max(1, T // 50)) == 0:
            progress(5 + int(45 * t / T))

    # Zyklenbudget (Durchschn. von Ladung & Entladung)
    m += pulp.lpSum((ch[t] + dh[t]) / (2 * cap) for t in range(T)) <= max_cyc

    if progress: progress(50)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=300)
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

    # Ziel
    m += pulp.lpSum(prices[t] * pulp.lpSum(dh_vars[i][t] - ch_vars[i][t] for i in range(n)) for t in range(T))

    for t in range(T):
        for i in range(n):
            m += c_vars[i][t] + d_vars[i][t] <= 1
            m += ch_vars[i][t] <= batt_maxs[i] * c_vars[i][t]
            m += dh_vars[i][t] <= batt_maxs[i] * d_vars[i][t]

            prev = starts[i] if t == 0 else soc_vars[i][t-1]
            m += soc_vars[i][t] == prev + effs[i] * ch_vars[i][t] - dh_vars[i][t] / effs[i]

        # Netzlimit (symmetrisch) je Zeitschritt: Import positiv
        total_ch = pulp.lpSum(ch_vars[i][t] for i in range(n))
        total_dh = pulp.lpSum(dh_vars[i][t] for i in range(n))
        net = ev[t] + total_ch - pv[t] - total_dh
        m +=  net <= grid_max
        m += -net <= grid_max

        if progress and t % (max(1, T // 50)) == 0:
            progress(5 + int(45 * t / T))

    # Zyklen je Batterie
    for i in range(n):
        m += pulp.lpSum((ch_vars[i][t] + dh_vars[i][t]) / (2 * caps[i]) for t in range(T)) <= max_cyc[i]

    if progress: progress(50)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=300)
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

# ── Streamlit App (UI & Logik modularisiert) ────────
def main():
    st.set_page_config(layout='wide')
    st.title('BESS: Skalierbar & Szenario-Vergleich')

    # Session State initialisieren
    init_session_state()

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

    # ── Tab 2: Upload (EINZIGER UPLOAD-BEREICH) ─────────────────
    with tab2:
        st.markdown('### Datei-Upload')
        st.info('📁 Laden Sie hier Ihre Dateien hoch. Diese stehen dann auch in Tab 3 zur Verfügung.')
        
        price_file = st.file_uploader('Preisgang [csv/xls/xlsx]', type=['csv','xls','xlsx'])
        if price_file:
            st.session_state.price_file = price_file
            
        pv_file = st.file_uploader('PV-Lastgang [csv/xls/xlsx]', type=['csv','xls','xlsx'])
        if pv_file:
            st.session_state.pv_file = pv_file
            
        ev_file = st.file_uploader('EV-Lastgang [csv/xls/xlsx]', type=['csv','xls','xlsx'])
        if ev_file:
            st.session_state.ev_file = ev_file

        # Validierung und Vorschau
        if st.session_state.price_file and st.session_state.pv_file and st.session_state.ev_file:
            try:
                p_df = load_price_df(st.session_state.price_file)
                pv_df = load_pv_df(st.session_state.pv_file)
                ev_df = load_ev_df(st.session_state.ev_file)
                valid, msg = validate_data(p_df, pv_df, ev_df)
                
                if valid:
                    st.success("✅ Dateien erfolgreich geladen und validiert.")
                    
                    col1, col2, col3 = st.columns(3)
                    ts_all = p_df['Zeitstempel']
                    
                    with col1:
                        st.metric('Zeitpunkte gesamt', f"{len(ts_all)}")
                    
                    with col2:
                        try:
                            interval = ts_all.diff().dropna().mode()[0].total_seconds()/3600.0
                            st.metric('Intervall (h)', f"{interval:.2f}")
                        except (IndexError, AttributeError):
                            st.metric('Intervall (h)', "Unbekannt")
                            
                    with col3:
                        start_date = ts_all.min().strftime('%d.%m.%Y')
                        end_date = ts_all.max().strftime('%d.%m.%Y')
                        st.metric('Zeitraum', f"{start_date} - {end_date}")
                        
                    st.markdown('##### Datenvorschau (erste 5 Zeilen)')
                    preview_df = pd.DataFrame({
                        'Zeit': ts_all.head(),
                        'Preis (€/MWh)': p_df['Preis_€/MWh'].head(),
                        'PV (kWh)': pv_df['PV_kWh'].head(),
                        'EV (kWh)': ev_df['EV_kWh'].head()
                    })
                    st.dataframe(preview_df)
                    
                else:
                    st.error(f"❌ Validierungsfehler: {msg}")
                    
            except Exception as e:
                st.error(f"❌ Fehler beim Laden der Dateien: {str(e)}")

    # ── Tab 3: Simulation & Ergebnisse (OHNE UPLOAD) ─────
    with tab3:
        st.markdown('### Simulation und Ergebnis-Vergleich')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            status = "✅" if st.session_state.price_file else "❌"
            st.markdown(f"**Preisdaten:** {status}")
        with col2:
            status = "✅" if st.session_state.pv_file else "❌"
            st.markdown(f"**PV-Daten:** {status}")
        with col3:
            status = "✅" if st.session_state.ev_file else "❌"
            st.markdown(f"**EV-Daten:** {status}")
        
        if not all([st.session_state.price_file, st.session_state.pv_file, st.session_state.ev_file]):
            st.warning('⚠️ Bitte laden Sie zuerst alle Dateien in Tab 2 hoch.')
            
        if st.button('▶️ Simulation starten', disabled=not all([st.session_state.price_file, st.session_state.pv_file, st.session_state.ev_file])):
            sim_ok = False
            try:
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
                
                if len(prices) != len(pv) or len(prices) != len(ev):
                    st.error("❌ Datenreihen haben unterschiedliche Längen")
                    st.stop()
                
                try:
                    interval_h = ts.diff().dropna().mode()[0].total_seconds() / 3600.0
                    if interval_h <= 0 or interval_h > 24:
                        raise ValueError("Unrealistisches Intervall")
                except (IndexError, AttributeError, ValueError):
                    interval_h = 1.0
                    st.warning("⚠️ Intervall konnte nicht bestimmt werden, verwende 1h als Standard")
                    
                st.info(f'📊 Optimierung für {len(prices)} Zeitpunkte mit {interval_h:.2f}h Intervall')
                
                # Validierung der Konfigurationen
                for idx, cfg in enumerate(configs, 1):
                    if cfg['cap'] <= 0 or cfg['bat_kw'] <= 0:
                        st.error(f"❌ Batterie {idx}: Kapazität und Leistung müssen > 0 sein")
                        st.stop()
                    if cfg['eff_pct'] <= 0 or cfg['eff_pct'] > 1:
                        st.error(f"❌ Batterie {idx}: Wirkungsgrad muss zwischen 0 und 100% liegen")
                        st.stop()
                
                # Optimierung mit Progress
                with st.spinner('🔄 Führe Optimierungsberechnungen durch...'):
                    free_results = [solve_battery(prices, pv, ev, cfg, grid_kw, interval_h, set_progress) for cfg in configs]
                    obj_joint, chs_joint, dhs_joint = solve_joint(prices, pv, ev, configs, grid_kw, interval_h, set_progress)
            
                # Einzeloptimierung anzeigen
                st.subheader('🔋 Einzeloptimierung (unabhängige Batterien)')
                tot_free = sum(obj for obj, *_ in free_results)
                
                col_metrics = st.columns(len(configs) + 1)
                for idx, (cfg, (obj, ch_v, dh_v)) in enumerate(zip(configs, free_results)):
                    with col_metrics[idx]:
                        st.metric(f"B{idx+1} ({cfg['mode']})", fmt_euro(obj))
                        
                with col_metrics[-1]:
                    st.metric('🔄 Gesamt Einzeln', fmt_euro(tot_free))
                
                # Joint Optimierung
                st.subheader('🔗 Gemeinsame Optimierung')
                joint_cols = st.columns(len(configs) + 1)
                
                with joint_cols[-1]:
                    st.metric('💰 Gesamterlös', fmt_euro(obj_joint))
                    
                for idx in range(len(configs)):
                    share = float(np.dot(prices, dhs_joint[idx] - chs_joint[idx]))
                    with joint_cols[idx]:
                        st.metric(f"B{idx+1} Anteil", fmt_euro(share))
                        
                # Vergleich und Analyse
                delta = obj_joint - tot_free
                pct = (delta/abs(tot_free)*100) if tot_free != 0 else 0
                st.subheader('📊 Analyse & Vergleich')
                
                analysis_cols = st.columns(4)
                with analysis_cols[0]:
                    st.metric('💹 Δ absolut', fmt_euro(delta), delta=f"{delta:+.2f}")
                
                with analysis_cols[1]:
                    st.metric('📈 Δ prozentual', f"{pct:+.2f}%", delta=f"{pct:+.2f}")
                
                sum_ch = np.sum(chs_joint, axis=0) if len(chs_joint) else np.zeros_like(ev)
                sum_dh = np.sum(dhs_joint, axis=0) if len(dhs_joint) else np.zeros_like(ev)
                net_load = ev - pv + sum_ch - sum_dh  # kWh pro Schritt, Import positiv
                
                util = np.mean(np.abs(net_load)/(grid_kw*interval_h))*100 if grid_kw > 0 else 0
                viol = (np.abs(net_load) > grid_kw*interval_h).sum()
                
                with analysis_cols[2]:
                    st.metric('⚡ Netzauslastung', f"{util:.1f}%")
                with analysis_cols[3]:
                    st.metric('⚠️ Überlastungen', f"{viol}")
                    
                if viol > 0:
                    st.error(f"🚨 {viol} Zeitpunkte mit Netzüberlastung erkannt!")
                    
                if delta > 0:
                    st.success(f"✅ Gemeinsame Optimierung bringt {fmt_euro(delta)} Mehrerlös ({pct:.1f}%)")
                elif delta < 0:
                    st.warning(f"⚠️ Einzeloptimierung wäre {fmt_euro(-delta)} besser ({-pct:.1f}%)")
                else:
                    st.info("ℹ️ Kein Unterschied zwischen Einzel- und Gemeinschaftsoptimierung")
                
                # ── Netzauslastungs-Analyse ─────────────────────────
                st.subheader('📈 Netzauslastungs-Analyse')
                analysis_df = pd.DataFrame({
                    'Zeit': ts,
                    'Netzlast_kW': net_load / interval_h if interval_h > 0 else net_load * 4,
                    'Netzlast_kWh': net_load,
                    'Auslastung_%': (np.abs(net_load) / (grid_kw * interval_h)) * 100 if grid_kw > 0 else 0,
                    'PV_Erzeugung': pv,
                    'EV_Verbrauch': ev,
                    'Batt_Laden': sum_ch,
                    'Batt_Entladen': sum_dh
                })
                
                # Quartalsweise Aggregation für Säulendiagramm
                analysis_df['Quartal'] = analysis_df['Zeit'].dt.to_period('Q')
                quarterly_stats = analysis_df.groupby('Quartal').agg({
                    'Netzlast_kW': 'mean',
                    'Auslastung_%': 'mean',
                    'Netzlast_kWh': 'sum'
                }).reset_index()
                
                if not quarterly_stats.empty:
                    quarterly_stats['Quartal_str'] = quarterly_stats['Quartal'].astype(str)
                    st.markdown("#### 📊 Netzauslastung - Quartalsübersicht")
                    st.markdown("**Durchschnittliche Netzauslastung pro Quartal (%)**")
                    st.bar_chart(data=quarterly_stats.set_index('Quartal_str')['Auslastung_%'], height=300)
                    st.markdown("**Gesamte Netzlast pro Quartal (MWh)**")
                    quarterly_chart_data = quarterly_stats.set_index('Quartal_str')
                    quarterly_chart_data['Netzlast_MWh'] = quarterly_chart_data['Netzlast_kWh'] / 1000
                    st.bar_chart(data=quarterly_chart_data['Netzlast_MWh'], height=300)
                
                # ── Zahlenmäßige Jahresauswertung ───────────────────
                st.markdown("#### 📋 Zahlenmäßige Jahresauswertung")
                total_net_kwh = analysis_df['Netzlast_kWh'].sum()
                avg_net_kw = analysis_df['Netzlast_kW'].mean() if not analysis_df.empty else 0
                max_net_kw = analysis_df['Netzlast_kW'].max() if not analysis_df.empty else 0
                min_net_kw = analysis_df['Netzlast_kW'].min() if not analysis_df.empty else 0
                
                net_cols = st.columns(4)
                with net_cols[0]:
                    st.metric("🔌 Gesamt-Netzlast", f"{total_net_kwh/1000:.1f} MWh")
                with net_cols[1]:
                    st.metric("⚡ Ø Netzlast", f"{avg_net_kw:.1f} kW")
                with net_cols[2]:
                    st.metric("📈 Max Netzlast", f"{max_net_kw:.1f} kW")
                with net_cols[3]:
                    st.metric("📉 Min Netzlast", f"{min_net_kw:.1f} kW")
                
                # ── Eigenverbrauchsdeckungs-Analyse ────────────────
                st.subheader('🏠 Eigenverbrauchsdeckungs-Analyse')
                if len(pv) > 0 and len(ev) > 0:
                    pv_to_ev_direct = np.zeros(len(pv))
                    batt_to_ev = np.zeros(len(pv))
                    grid_to_ev = np.zeros(len(pv))
                    
                    total_batt_output = sum_dh  # Batterie-Output je t
                    
                    for t in range(len(pv)):
                        ev_demand_t = ev[t]
                        pv_available_t = pv[t]
                        batt_available_t = total_batt_output[t]
                        
                        # 1) PV → EV
                        pv_direct = min(pv_available_t, ev_demand_t)
                        pv_to_ev_direct[t] = pv_direct
                        remaining_ev = ev_demand_t - pv_direct
                        
                        # 2) Batterie → EV
                        if remaining_ev > 0:
                            batt_contribution = min(batt_available_t, remaining_ev)
                            batt_to_ev[t] = batt_contribution
                            remaining_ev -= batt_contribution
                        
                        # 3) Netz → EV
                        if remaining_ev > 0:
                            grid_to_ev[t] = remaining_ev
                    
                    total_pv_to_ev = pv_to_ev_direct.sum()
                    total_batt_to_ev = batt_to_ev.sum()
                    total_grid_to_ev = grid_to_ev.sum()
                    total_ev_demand = ev.sum()
                    
                    if total_ev_demand > 0:
                        pie_data = pd.DataFrame({
                            'Energiequelle': ['🌞 PV-Direktversorgung', '🔋 Batterie-Versorgung', '🔌 Netzbezug'],
                            'MWh': [total_pv_to_ev/1000, total_batt_to_ev/1000, total_grid_to_ev/1000],
                            'Anteil_%': [
                                (total_pv_to_ev/total_ev_demand)*100,
                                (total_batt_to_ev/total_ev_demand)*100, 
                                (total_grid_to_ev/total_ev_demand)*100
                            ]
                        })
                        
                        st.markdown("#### 🥧 EV-Eigenverbrauchsdeckung")
                        st.bar_chart(data=pie_data.set_index('Energiequelle')['MWh'], height=400)
                        st.dataframe(
                            pie_data.style.format({'MWh': '{:.1f}', 'Anteil_%': '{:.1f}%'}),
                            hide_index=True
                        )
                        
                        # Kennzahlen
                        ev_autarkie = ((total_pv_to_ev + total_batt_to_ev) / total_ev_demand * 100)
                        pv_autarkie_ev = (total_pv_to_ev / total_ev_demand * 100)
                        batt_contribution = (total_batt_to_ev / total_ev_demand * 100)
                        grid_dependency = (total_grid_to_ev / total_ev_demand * 100)
                        
                        total_pv = pv.sum()
                        if total_pv > 0:
                            # Näherung: PV für Direktverbrauch + (alle) Batterieladungen
                            pv_eigenverbrauch = total_pv_to_ev + np.sum(chs_joint)
                            pv_eigenverbrauchsquote = (pv_eigenverbrauch / total_pv * 100)
                        else:
                            pv_eigenverbrauchsquote = 0
                        
                        autarkie_cols = st.columns(2)
                        with autarkie_cols[0]:
                            st.markdown("##### 🎯 Autarkie-Grade")
                            st.metric("🔋 Gesamt-Autarkie EV", f"{ev_autarkie:.1f}%")
                            st.metric("🌞 PV-Direktversorgung", f"{pv_autarkie_ev:.1f}%") 
                            st.metric("⚡ Batterie-Beitrag", f"{batt_contribution:.1f}%")
                            st.metric("🔌 Netzbezugs-Anteil", f"{grid_dependency:.1f}%")
                        with autarkie_cols[1]:
                            st.markdown("##### 💰 Wirtschaftlichkeits-Kennzahlen")
                            st.metric("🏠 PV-Eigenverbrauchsquote", f"{pv_eigenverbrauchsquote:.1f}%")
                            st.metric("📈 Eingesparte Netzbezugs-MWh", f"{(total_pv_to_ev + total_batt_to_ev)/1000:.1f}")
                            total_charge = np.sum(chs_joint)
                            total_discharge = np.sum(dhs_joint)
                            if total_charge > 0:
                                batt_efficiency = (total_discharge / total_charge * 100)
                                st.metric("🔄 Batterie-Gesamteffizienz", f"{batt_efficiency:.1f}%")
                            else:
                                st.metric("🔄 Batterie-Gesamteffizienz", "0.0%")
                            cycles_used = 0.0
                            for i, cfg in enumerate(configs):
                                if cfg['cap'] > 0:
                                    cycles_used += (np.sum(chs_joint[i]) + np.sum(dhs_joint[i])) / (2 * cfg['cap'])
                            st.metric("🔄 Batterie-Zyklen genutzt", f"{cycles_used:.1f}")
                        
                        # Monatliche Aufteilung
                        st.markdown("#### 📅 Monatliche Eigenverbrauchsdeckung")
                        try:
                            analysis_df['Monat'] = analysis_df['Zeit'].dt.to_period('M')
                            monthly_stats = analysis_df.groupby('Monat').agg({
                                'PV_Erzeugung': 'sum',
                                'EV_Verbrauch': 'sum', 
                                'Batt_Laden': 'sum',
                                'Batt_Entladen': 'sum'
                            }).reset_index()
                            
                            if not monthly_stats.empty:
                                monthly_stats['Monat_str'] = monthly_stats['Monat'].astype(str)
                                # PV-Direktdeckung pro Monat (viertelstündlich korrekt)
                                monthly_direct_pv = []
                                for month_period in monthly_stats['Monat']:
                                    mask = analysis_df['Monat'] == month_period
                                    month_pv = analysis_df.loc[mask, 'PV_Erzeugung'].values
                                    month_ev = analysis_df.loc[mask, 'EV_Verbrauch'].values
                                    direct_pv = np.minimum(month_pv, month_ev).sum()
                                    monthly_direct_pv.append(direct_pv)
                                monthly_stats['PV_Direktdeckung'] = monthly_direct_pv
                                monthly_stats['Autarkie_%'] = 0.0
                                mask_nonzero = monthly_stats['EV_Verbrauch'] > 0
                                monthly_stats.loc[mask_nonzero, 'Autarkie_%'] = (
                                    (monthly_stats.loc[mask_nonzero, 'PV_Direktdeckung'] + 
                                     monthly_stats.loc[mask_nonzero, 'Batt_Entladen']) / 
                                    monthly_stats.loc[mask_nonzero, 'EV_Verbrauch'] * 100
                                )
                                st.markdown("**Monatliche Energie-Bilanz (MWh)**")
                                monthly_chart_data = monthly_stats.set_index('Monat_str').rename(columns={
                                    'PV_Erzeugung': '🌞 PV-Erzeugung (MWh)',
                                    'EV_Verbrauch': '🚗 EV-Verbrauch (MWh)', 
                                    'Batt_Entladen': '🔋 Batterie-Output (MWh)'
                                })
                                for col in ['🌞 PV-Erzeugung (MWh)', '🚗 EV-Verbrauch (MWh)', '🔋 Batterie-Output (MWh)']:
                                    monthly_chart_data[col] = monthly_chart_data[col] / 1000
                                st.bar_chart(data=monthly_chart_data[['🌞 PV-Erzeugung (MWh)', '🚗 EV-Verbrauch (MWh)', '🔋 Batterie-Output (MWh)']], height=400)
                                st.markdown("**Monatliche Autarkie-Grade**")
                                autarkie_display = monthly_stats[['Monat_str', 'Autarkie_%']].copy()
                                autarkie_display.columns = ['Monat', 'Autarkie (%)']
                                st.dataframe(autarkie_display.style.format({'Autarkie (%)': '{:.1f}%'}), hide_index=True)
                        except Exception as e:
                            st.warning(f"⚠️ Monatliche Auswertung konnte nicht erstellt werden: {str(e)}")
                    else:
                        st.warning("⚠️ Kein EV-Verbrauch gefunden - Eigenverbrauchsanalyse nicht möglich.")
                else:
                    st.warning("⚠️ Keine gültigen Daten für Eigenverbrauchsanalyse verfügbar.")

                # Ergebnisse/Export nur, wenn alles oben durchlief
                sim_ok = True
                
            except Exception as e:
                st.error(f"❌ Fehler bei der Simulation: {str(e)}")
                st.exception(e)
            
            # Ergebnisse-Tabelle & Download (nur bei Erfolg)
            if sim_ok:
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
                out['Netzlast_%'] = np.where(grid_kw > 0, np.abs(net_load)/(grid_kw*interval_h)*100, 0)
                st.dataframe(out)
                buf = BytesIO()
                out.to_excel(buf, index=False, engine='openpyxl')
                buf.seek(0)
                st.download_button(
                    '📥 Excel-Export', buf,
                    file_name=f"res_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

# ── Konfig speichern ────────────────────────────────
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
