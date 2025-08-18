import streamlit as st
import pandas as pd
import numpy as np
import pulp
from io import BytesIO
import locale
import json

# ─────────────────────────────────────────────────────────────
# Single-Battery · Eigenverbrauchsmaximierung (ohne Day-Ahead)
# Stand: 18.08.2025
# Logik (hinter dem Zähler, EV hat Vorrang):
#   1) PV → EV (deterministisch)
#   2) PV-Überschuss → Batterie laden (ch_pv) bis Power/SoC
#   3) Batterie → EV (dh_ev) bis Power/SoC
#   4) Rest-EV → Netzimport (grid_ev), Rest-PV → Netzausgang (pv_grid)
#   5) PV-Curtailment (pv_curt) nur falls nötig (zur Vermeidung von Infeasible)
# Ziel: Minimiere Σ grid_ev (Netzbezug für EV). Kleiner Zusatz: minimiere auch Curtailment (sehr kleine Gewichtung).
# ─────────────────────────────────────────────────────────────

# ── Fortschrittsanzeige ─────────────────────────────
def set_progress(pct: int):
    st.session_state.progress_bar.progress(pct)
    st.session_state.progress_text.markdown(f"**Fortschritt:** {pct}%")

# ── Euro-Format (für evtl. spätere Erweiterung) ─────
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
        upl.seek(0)
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

def load_pv_df(upl):    return load_generic_series(upl, 'PV_kWh')
def load_ev_df(upl):    return load_generic_series(upl, 'EV_kWh')

# ── Datenvalidierung (nur PV & EV) ──────────────────
def validate_data_ev(pv_df, ev_df):
    for df, name in [(pv_df,'PV'), (ev_df,'EV')]:
        if df.empty:
            return False, f'{name}-Datei ist leer oder konnte nicht geladen werden'
    lengths = [len(pv_df), len(ev_df)]
    if lengths[0] != lengths[1]:
        return False, f'Unterschiedliche Datenlängen: PV={lengths[0]}, EV={lengths[1]}'
    for df, name in [(pv_df,'PV'), (ev_df,'EV')]:
        if df['Zeitstempel'].isna().any():
            return False, f"{name}: Zeitstempel fehlerhaft oder unvollständig"
        if len(df['Zeitstempel'].unique()) != len(df):
            return False, f"{name}: Doppelte Zeitstempel gefunden"
    return True, 'Datenvalidierung erfolgreich'

# ─────────────────────────────────────────────────────────────
# Solver – Eigenverbrauchsmaximierung (Single Battery)
# Variablen: ch_pv, dh_ev, pv_grid, grid_ev, pv_curt, c, d, soc
# Ziel: min Σ grid_ev + ε·Σ pv_curt
# ─────────────────────────────────────────────────────────────

def solve_ev_only(pv, ev, cfg, grid_kw, interval_h, progress=None, eps_curt=1e-6, bigM_shed=1e6):
    T = len(pv)
    cap = cfg['cap']
    eff = cfg['eff_pct'] ** 0.5
    pmax = cfg['bat_kw'] * interval_h  # kWh pro Schritt
    gmax = grid_kw * interval_h        # kWh pro Richtung & Schritt

    # Vorrangaufteilung fix: PV→EV
    pv_ev = np.minimum(pv, ev)
    pv_sur = pv - pv_ev
    ev_res = ev - pv_ev

    m = pulp.LpProblem('EV_ONLY', pulp.LpMinimize)

    # Variablen (ohne Binärvariablen → LP, stabiler)
    ch_pv  = pulp.LpVariable.dicts('ch_pv',  range(T), lowBound=0, upBound=pmax)
    dh_ev  = pulp.LpVariable.dicts('dh_ev',  range(T), lowBound=0, upBound=pmax)
    pv_grid= pulp.LpVariable.dicts('pv_grid',range(T), lowBound=0, upBound=gmax)
    grid_ev= pulp.LpVariable.dicts('grid_ev',range(T), lowBound=0, upBound=gmax)
    pv_curt= pulp.LpVariable.dicts('pv_curt',range(T), lowBound=0)
    ev_short= pulp.LpVariable.dicts('ev_short',range(T), lowBound=0)  # weiche EV-Deckung, nur wenn zwingend
    soc    = pulp.LpVariable.dicts('soc',    range(T), lowBound=0, upBound=cap)

    # Ziel: Netzbezug minimieren + starke Strafe für EV-Undeckung + kleine Strafe für Curtailment
    m += pulp.lpSum(grid_ev[t] + bigM_shed*ev_short[t] + eps_curt*pv_curt[t] for t in range(T))

    for t in range(T):
        # Aufteilung Überschüsse/Bedarfe
        m += ch_pv[t] + pv_grid[t] + pv_curt[t] == float(pv_sur[t])
        m += dh_ev[t] + grid_ev[t] + ev_short[t] == float(ev_res[t])

        # SoC-Dynamik
        prev = cfg['start_soc'] if t == 0 else soc[t-1]
        m += soc[t] == prev + eff * ch_pv[t] - dh_ev[t] / eff

        # Leistungslimits
        m += ch_pv[t] <= pmax
        m += dh_ev[t] <= pmax

        # Netzanschluss je Richtung
        m += pv_grid[t] <= gmax
        m += grid_ev[t] <= gmax

        if progress and t % max(1, T//50) == 0:
            progress(5 + int(90*t/T))

    # Zyklenbudget
    if cap > 0:
        m += pulp.lpSum((ch_pv[t] + dh_ev[t]) / (2 * cap) for t in range(T)) <= cfg['max_cycles']

    status = pulp.PULP_CBC_CMD(msg=False, timeLimit=300).solve(m)

    if status != pulp.LpStatusOptimal:
        return {'status': pulp.LpStatus[status]}
# ── Session State Initialisierung ───────────────────
def init_session_state():
    if 'progress_bar' not in st.session_state:
        st.session_state.progress_bar = st.sidebar.progress(0)
        st.session_state.progress_text = st.sidebar.empty()
    if 'pv_file' not in st.session_state:
        st.session_state.pv_file = None
    if 'ev_file' not in st.session_state:
        st.session_state.ev_file = None

# ── Streamlit App ───────────────────────────────────
def main():
    st.set_page_config(layout='wide')
    st.title('BESS – Single Battery · Eigenverbrauchsmaximierung')

    init_session_state()

    tab1, tab2, tab3 = st.tabs(['Konfiguration', 'Upload', 'Simulation'])

    # ── Tab 1: Konfiguration ────────────────────────
    with tab1:
        st.markdown('### Batterie & Netz')
        start_soc = st.number_input('Start-SoC (kWh)', 0.0, 1e9, 0.0)
        cap       = st.number_input('Kapazität (kWh)', 0.1, 1e9, 4472.0)
        bat_kw    = st.number_input('Leistung (kW)', 0.1, 1e9, 559.0)
        eff_pct   = st.slider('Round-Trip-Effizienz RTE (%)', min_value=1.0, max_value=100.0, value=91.0, step=0.1, format='%.1f') / 100.0
        max_cycles= st.number_input('Max. Zyklen (im Zeitraum)', 0.0, 1e6, 548.0)
        grid_kw   = st.number_input('Netzanschluss Limit (kW je Richtung)', 0.1, 1e9, 37000.0)

        st.markdown('### Skalierung')
        pv_scale = st.slider('PV-Skalierung (Anzahl)', min_value=1, max_value=50, value=1, step=1)
        ev_scale = st.slider('EV-Skalierung (Anzahl)', min_value=1, max_value=50, value=1, step=1)

        cfg = {
            'start_soc': start_soc,
            'cap': cap,
            'bat_kw': bat_kw,
            'eff_pct': eff_pct,
            'max_cycles': max_cycles
        }

        st.button('Konfiguration speichern', on_click=lambda: save_configuration(pv_scale, ev_scale, cfg, grid_kw))

    # ── Tab 2: Upload ───────────────────────────────
    with tab2:
        st.markdown('### Datei-Upload')
        st.info('📁 PV- und EV-Zeitreihen laden. Diese stehen in Tab 3 zur Verfügung.')
        pv_file = st.file_uploader('PV-Lastgang [csv/xls/xlsx] (Zeit, kWh)', type=['csv','xls','xlsx'])
        if pv_file:
            st.session_state.pv_file = pv_file
        ev_file = st.file_uploader('EV-Lastgang [csv/xls/xlsx] (Zeit, kWh)', type=['csv','xls','xlsx'])
        if ev_file:
            st.session_state.ev_file = ev_file

        if st.session_state.pv_file and st.session_state.ev_file:
            try:
                pv_df = load_pv_df(st.session_state.pv_file)
                ev_df = load_ev_df(st.session_state.ev_file)
                valid, msg = validate_data_ev(pv_df, ev_df)
                if valid:
                    st.success('✅ Dateien erfolgreich geladen und validiert.')
                    ts_all = pv_df['Zeitstempel']
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
        c1, c2 = st.columns(2)
        with c1: st.markdown(f"**PV-Daten:** {'✅' if st.session_state.pv_file else '❌'}")
        with c2: st.markdown(f"**EV-Daten:** {'✅' if st.session_state.ev_file else '❌'}")

        ready = all([st.session_state.pv_file, st.session_state.ev_file])
        if st.button('▶️ Eigenverbrauch optimieren', disabled=not ready):
            try:
                # Daten laden
                pv_df = load_pv_df(st.session_state.pv_file)
                ev_df = load_ev_df(st.session_state.ev_file)

                valid, msg = validate_data_ev(pv_df, ev_df)
                if not valid:
                    st.error(f"❌ Validierungsfehler: {msg}")
                    st.stop()

                ts = pv_df['Zeitstempel']
                pv = pv_df['PV_kWh'].to_numpy() * pv_scale
                ev = ev_df['EV_kWh'].to_numpy() * ev_scale

                if len(pv) != len(ev):
                    st.error('❌ Datenreihen haben unterschiedliche Längen')
                    st.stop()

                try:
                    interval_h = ts.diff().dropna().mode()[0].total_seconds()/3600.0
                    if not (0 < interval_h <= 24):
                        raise ValueError
                except Exception:
                    interval_h = 1.0
                    st.warning('⚠️ Intervall unbekannt – setze 1h')

                st.info(f'📊 Optimierung für {len(pv)} Schritte @ {interval_h:.2f}h')

                # Optimierung
                res = solve_ev_only(pv, ev, cfg, grid_kw, interval_h, set_progress)
                if res.get('status') != 'Optimal':
                    st.error(f"Optimierung nicht optimal: {res.get('status')}")
                    st.stop()

                # Ergebnisse
                pv_ev      = res['pv_ev']
                pv_sur     = res['pv_sur']
                ev_res     = res['ev_res']
                ch_pv      = res['ch_pv']
                dh_ev      = res['dh_ev']
                pv_grid    = res['pv_grid']
                grid_ev    = res['grid_ev']
                pv_curt    = res['pv_curt']
                ev_short   = res['ev_short']
                soc        = res['soc']

                # Physikalische Unterdeckungs-Untergrenze (nur durch Limits pmax+gmax bedingt)
                pmax = cfg['bat_kw'] * interval_h
                gmax = grid_kw * interval_h
                ev_short_lb = np.maximum(0, ev_res - (gmax + pmax))
                ev_short_lb_tot = ev_short_lb.sum()

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
                pv_selfuse = ev_from_pv + ch_pv.sum()
                pv_selfuse_q = (pv_selfuse / pv_total * 100) if pv_total>0 else 0.0

                # Unterschreitung (falls Netzlimit + Batteriepower/SoC nicht reichen)
                ev_short_tot = ev_short.sum()
                short_steps = int((ev_short > 1e-6).sum())

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
                with m10: st.metric('Zyklen genutzt', f"{res['cycles']:.1f}")

                if ev_short_tot > 1e-6:
                    st.error(f"⚠️ EV nicht vollständig gedeckt: {ev_short_tot/1000:.2f} MWh in {short_steps} Zeitschritten (Netzlimit + Batteriepower/SoC reichen dort nicht).")
                if ev_short_lb_tot > 1e-6:
                    st.info(f"ℹ️ Physikalische Unterdeckungs-Untergrenze: mind. {ev_short_lb_tot/1000:.2f} MWh, weil EV_Rest > pmax+gmax in manchen Schritten.")

                # Zeitreihen-DF
                results_df = pd.DataFrame({
                    'Zeit': ts,
                    'PV_kWh': pv,
                    'EV_kWh': ev,
                    'PV→EV_kWh': pv_ev,
                    'EV_Rest_kWh': ev_res,
                    'PV_Überschuss_kWh': pv_sur,
                    'ch_pv_kWh': ch_pv,
                    'dh_ev_kWh': dh_ev,
                    'PV_Export_kWh': pv_grid,
                    'EV_aus_Netz_kWh': grid_ev,
                    'PV_Curtail_kWh': pv_curt,
                    'EV_unterdeckt_kWh': ev_short,
                    'EV_unterdeckt_LB_kWh': ev_short_lb,
                    'SoC_kWh': soc
                })

                st.markdown('#### Zeitreihen (Ausschnitt)')
                st.dataframe(results_df.head(200))

                # Quartalsaggregation (Zusammenfassung)
                results_df['Quartal'] = results_df['Zeit'].dt.to_period('Q')
                q = results_df.groupby('Quartal').agg({
                    'EV_aus_Netz_kWh': 'sum',
                    'PV_Export_kWh': 'sum',
                    'PV_Curtail_kWh': 'sum',
                    'EV_unterdeckt_kWh': 'sum',
                    'EV_unterdeckt_LB_kWh': 'sum'
                }).reset_index()
                if not q.empty:
                    q['Quartal_str'] = q['Quartal'].astype(str)
                    st.markdown('#### Quartalsbilanz (MWh) – Netzbezug / Export / Curtail / Unterdeckung / Unterdeckung-LB')
                    chart_df = q.set_index('Quartal_str')/1000.0
                    st.bar_chart(chart_df[['EV_aus_Netz_kWh','PV_Export_kWh','PV_Curtail_kWh','EV_unterdeckt_kWh','EV_unterdeckt_LB_kWh']], height=300)

                # Export
                st.markdown('#### Export')
                buf = BytesIO()
                results_df.to_excel(buf, index=False, engine='openpyxl')
                buf.seek(0)
                st.download_button('📥 Excel-Export', buf,
                    file_name=f"BESS_Single_Eigenverbrauch_{pd.Timestamp.now():%Y%m%d_%H%M%S}.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
