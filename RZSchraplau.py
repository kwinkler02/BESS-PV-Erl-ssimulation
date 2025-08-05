import streamlit as st
import pandas as pd
import numpy as np
import pulp
from io import BytesIO
import locale

# ── Fortschrittsanzeige ─────────────────────────────────────────────────────
progress_bar = st.sidebar.progress(0)
progress_text = st.sidebar.empty()
def set_progress(pct: int):
    progress_bar.progress(pct)
    progress_text.markdown(f"**Fortschritt:** {pct}%")

# ── Euro-Format ──────────────────────────────────────────────────────────────
try:
    locale.setlocale(locale.LC_ALL, "de_DE.UTF-8")
    def fmt_euro(x): return locale.currency(x, symbol=True, grouping=True)
except locale.Error:
    def fmt_euro(x):
        s = f"{x:,.2f}".replace(",","X").replace(".",",").replace("X",".")
        return s + " €"

# ── Daten-Lader ──────────────────────────────────────────────────────────────
def load_time_series(upl, usecols, names):
    if upl.name.lower().endswith('.csv'):
        df = pd.read_csv(upl, sep=';', decimal=',', usecols=usecols,
                         names=names, header=0)
    else:
        df = pd.read_excel(upl, usecols=usecols, names=names,
                           header=0, engine='openpyxl')
    df[names[0]] = parse_flexible_timestamp(df[names[0]])
    df[names[1]] = pd.to_numeric(df[names[1]], errors='raise')
    return df

def parse_flexible_timestamp(ts_series):
    parsed_ts = []
    for ts in ts_series:
        if pd.isna(ts):
            parsed_ts.append(pd.NaT)
            continue
        ts_str = str(ts).strip()
        try:
            parsed = pd.to_datetime(ts_str, dayfirst=True)
            parsed_ts.append(parsed)
        except:
            try:
                if '.' in ts_str and len(ts_str.split('.')[-1].split(' ')[0]) <= 2:
                    parts = ts_str.split('.')
                    if len(parts) >= 3:
                        year_part = parts[2].split(' ')[0]
                        if len(year_part) == 2:
                            year = int(year_part)
                            full_year = 2000 + year if year < 50 else 1900 + year
                            new_ts_str = ts_str.replace(f".{year_part} ", f".{full_year} ")
                            parsed = pd.to_datetime(new_ts_str, dayfirst=True)
                            parsed_ts.append(parsed)
                            continue
                parsed = pd.to_datetime(ts_str, infer_datetime_format=True)
                parsed_ts.append(parsed)
            except:
                parsed_ts.append(pd.NaT)
    return pd.Series(parsed_ts)

def load_price_df(upl): return load_time_series(upl, [0,1], ['Zeitstempel','Preis_€/MWh'])
def load_pv_df(upl):    return load_time_series(upl, [0,1], ['Zeitstempel','PV_kWh'])
def load_ev_df(upl):    return load_time_series(upl, [0,1], ['Zeitstempel','EV_kWh'])

# ── Zeitstempel-Ausrichtung ─────────────────────────────────────────────────
def align_timestamps(price_df, pv_df, ev_df):
    all_ts = pd.concat([price_df['Zeitstempel'], pv_df['Zeitstempel'], ev_df['Zeitstempel']]).drop_duplicates().sort_values().reset_index(drop=True)
    base = pd.DataFrame({'Zeitstempel': all_ts})
    price = base.merge(price_df, on='Zeitstempel', how='left').fillna(method='ffill').fillna(0)
    pv = base.merge(pv_df, on='Zeitstempel', how='left').fillna(method='ffill').fillna(0)
    ev = base.merge(ev_df, on='Zeitstempel', how='left').fillna(method='ffill').fillna(0)
    return price, pv, ev

# ── Datenvalidierung ────────────────────────────────────────────────────────
def validate_data(p_df, pv_df, ev_df):
    if p_df.empty or pv_df.empty or ev_df.empty:
        return False, "Leere Datei"
    if p_df['Zeitstempel'].isna().any(): return False, "Preis Zeitstempel fehlerhaft"
    if pv_df['Zeitstempel'].isna().any(): return False, "PV Zeitstempel fehlerhaft"
    if ev_df['Zeitstempel'].isna().any(): return False, "EV Zeitstempel fehlerhaft"
    if (pv_df['PV_kWh'] < 0).any() or (ev_df['EV_kWh'] < 0).any(): return False, "Neg. Last"
    return True, "OK"

# ── Simulation ───────────────────────────────────────────────────────────────
def run_sim():
    set_progress(0)
    p_df = load_price_df(price_file)
    pv_df = load_pv_df(pv_file)
    ev_df = load_ev_df(ev_file)
    valid, msg = validate_data(p_df, pv_df, ev_df)
    if not valid: st.error(msg); st.stop()
    try:
        p_df, pv_df, ev_df = align_timestamps(p_df, pv_df, ev_df)
        st.info(f"Daten aligned: {len(p_df)} Zeitpunkte")
    except Exception as e:
        st.error(e); st.stop()

    ts = p_df['Zeitstempel']
    prices = p_df['Preis_€/MWh'].to_numpy() / 1000.0
    pv = pv_df['PV_kWh'].to_numpy()
    ev = ev_df['EV_KWh'].to_numpy()
    interval_h = ts.diff().dropna().mode()[0].total_seconds() / 3600.0
    st.info(f"Intervall: {interval_h} h")

    for cfg in configs:
        if cfg['cap'] <= 0 or cfg['bat_kw'] <= 0 or cfg['eff_pct'] <= 0 or cfg['eff_pct'] > 1:
            st.error("Parameterfehler"); st.stop()
    if grid_kw <= 0: st.error("Grid Fehler"); st.stop()

    free = [solve_battery(prices, pv, ev, cfg, grid_kw, interval_h, set_progress) for cfg in configs]
    joint = solve_joint(prices, pv, ev, configs, grid_kw, interval_h, set_progress)
    return ts, prices, pv, ev, free, joint, interval_h

# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("BESS: Single vs. Multi Use & Grid Constraint")

price_file = st.sidebar.file_uploader("Preise", type=["csv", "xls", "xlsx"])
pv_file = st.sidebar.file_uploader("PV-Last", type=["csv", "xls", "xlsx"])
ev_file = st.sidebar.file_uploader("EV-Last", type=["csv", "xls", "xlsx"])

enable2 = st.sidebar.checkbox("Zweite Batterie", True)
configs = []
for i in (1, 2):
    if i == 1 or enable2:
        st.sidebar.markdown(f"**Batterie {i}**")
        mode = st.sidebar.selectbox(f"Modus{i}", ["Single Use", "Multi Use"], key=f"m{i}")
        start = st.sidebar.number_input(f"StartSoC{i} (kWh)", 0.0, 1e6, 0.0, key=f"s{i}")
        cap = st.sidebar.number_input(f"Kap{i} (kWh)", 0.1, 1e6, 4472.0, key=f"c{i}")
        bkw = st.sidebar.number_input(f"Leistung{i} (kW)", 0.1, 1e6, 559.0, key=f"p{i}")
        eff = st.sidebar.number_input(f"Eff{i} (%)", 1.0, 100.0, 91.0, key=f"e{i}") / 100.0
        cyc = st.sidebar.number_input(f"Zyklen{i}", 0.0, 1e4, 548.0, key=f"y{i}")
        configs.append({"mode": mode, "start_soc": start, "cap": cap, "bat_kw": bkw, "eff_pct": eff, "max_cycles": cyc})

grid_kw = st.sidebar.number_input("Netzanschluss (kW)", 0.1, 1e6, 37000.0)

if st.sidebar.button("▶️ Simulation starten"):
    if not (price_file and pv_file and ev_file):
        st.sidebar.error("Bitte alle Dateien hochladen.")
    else:
        try:
            st.session_state['res'] = run_sim()
            st.success("Fertig!")
        except Exception as e:
            st.error(e)

if 'res' not in st.session_state:
    st.info("Bitte Dateien hochladen und starten.")
    st.stop()

(ts, prices, pv, ev, free_res, jres, iv) = st.session_state['res']
obj_joint, chs_joint, dhs_joint = jres

col1, col2 = st.columns(2)
with col1:
    st.subheader("Einzeloptimierung")
    tot_free = 0
    for i, (cfg, (obj, _, _)) in enumerate(zip(configs, free_res), start=1):
        st.metric(f"B{i} ({cfg['mode']})", fmt_euro(obj))
        tot_free += obj
    st.metric("Gesamt Free", fmt_euro(tot_free))

with col2:
    st.subheader("Gemeinsam")
    st.metric("Gesamt", fmt_euro(obj_joint))
    for i, cfg in enumerate(configs, 1):
        ind = float(np.dot(prices, dhs_joint[i-1] - chs_joint[i-1]))
        st.metric(f"B{i} Anteil", fmt_euro(ind))

imp = obj_joint - tot_free
pct = (imp / abs(tot_free) * 100 if tot_free != 0 else 0)

st.subheader("Vergleich")
col1, col2, col3 = st.columns(3)
col1.metric("∆ absolut", fmt_euro(imp))
col2.metric("∆ %", f"{pct:.2f}%")
col3.metric("Netzauslastung", "{:.1f}%".format(np.mean((pv + ev + sum(chs_joint) - sum(dhs_joint)) / (grid_kw * iv)) * 100))

st.subheader("Ergebnisse")
out = pd.DataFrame({'Zeit': ts, 'Preis_€/kWh': prices, 'PV_kWh': pv, 'EV_kWh': ev})
for i in range(len(configs)):
    out[f'B{i+1}_Laden_kWh'] = chs_joint[i]
    out[f'B{i+1}_Entladen_kWh'] = dhs_joint[i]

tot_load = pv + ev + sum(chs_joint) - sum(dhs_joint)
out['Netzlast_kWh'] = tot_load
out['Netzlast_%'] = (tot_load / (grid_kw * iv)) * 100
viol = (tot_load > (grid_kw * iv)).sum()
if viol > 0:
    st.warning(f"⚠️ {viol} Überlastungen")

st.dataframe(out)
buf = BytesIO()
out.to_excel(buf, index=False, engine='openpyxl')
buf.seek(0)
st.download_button("📥 Laden", buf, f"res_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl")
