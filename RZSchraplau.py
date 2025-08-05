import streamlit as st
import pandas as pd
import numpy as np
import pulp
from io import BytesIO
import locale
import traceback

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
        return s + " €"

# ── Zeitstempel-Parsing ──────────────────────────────────────────────────────
def parse_flexible_timestamp(ts_series):
    parsed_ts = []
    for ts in ts_series:
        if pd.isna(ts):
            parsed_ts.append(pd.NaT)
            continue
        ts_str = str(ts).strip()
        for fmt in [None]:  # try default parse
            try:
                parsed = pd.to_datetime(ts_str, dayfirst=True, infer_datetime_format=True)
                parsed_ts.append(parsed)
                break
            except:
                parsed_ts.append(pd.NaT)
                break
    return pd.Series(parsed_ts)

# ── Generischer CSV/XLSX Loader ─────────────────────────────────────────────
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

# ── Spezifische Loader ───────────────────────────────────────────────────────
def load_price_df(upl):
    return load_time_series(upl, [0,1], ['Zeitstempel','Preis_€/MWh'])

def load_pv_df(upl):
    return load_time_series(upl, [0,1], ['Zeitstempel','PV_kWh'])

def load_ev_df(upl):
    return load_time_series(upl, [0,1], ['Zeitstempel','EV_kWh'])

# ── Einzelbatterie-Optimierung ───────────────────────────────────────────────
def solve_battery(prices, pv_vec, ev_vec, cfg, grid_kw, interval_h, progress_callback=None):
    T = len(prices)
    batt_max = cfg['bat_kw'] * interval_h
    grid_max = grid_kw * interval_h
    cap = cfg['cap']
    eff = np.sqrt(cfg['eff_pct'])
    max_cyc = cfg['max_cycles']

    m = pulp.LpProblem("BESS", pulp.LpMaximize)
    c = pulp.LpVariable.dicts("c", range(T), cat="Binary")
    d = pulp.LpVariable.dicts("d", range(T), cat="Binary")
    ch = pulp.LpVariable.dicts("ch", range(T), lowBound=0, upBound=batt_max)
    dh = pulp.LpVariable.dicts("dh", range(T), lowBound=0, upBound=batt_max)
    soc= pulp.LpVariable.dicts("soc",range(T), lowBound=0, upBound=cap)

    # Objective: maximize arbitrage revenue
    m += pulp.lpSum(prices[t] * (dh[t] - ch[t]) for t in range(T))

    for t in range(T):
        # no simultaneous charge/discharge
        m += c[t] + d[t] <= 1
        # power limits
        m += ch[t] <= batt_max * c[t]
        m += dh[t] <= batt_max * d[t]
        # min durations for single use
        if cfg['mode'] == 'Single Use':
            m += ch[t] >= interval_h * c[t]
            m += dh[t] >= interval_h * d[t]
        # SOC balance
        prev = cfg['start_soc'] if t == 0 else soc[t-1]
        m += soc[t] == prev + eff * ch[t] - dh[t] / eff
        # grid constraint: PV + EV + charging - discharging <= grid_max
        m += pv_vec[t] + ev_vec[t] + ch[t] - dh[t] <= grid_max
        # progress callback
        if progress_callback and t % max(1, T//50) == 0:
            progress_callback(5 + int(45 * t / T))

    # cycle constraint
    m += pulp.lpSum((ch[t] + dh[t]) / (2 * cap) for t in range(T)) <= max_cyc

    if progress_callback: progress_callback(50)
    status = pulp.PULP_CBC_CMD(msg=False, timeLimit=120).solve(m)
    if progress_callback: progress_callback(90)

    if status != pulp.LpStatusOptimal:
        st.warning(f"Single Use Solver Status: {pulp.LpStatus[status]}")
        return 0.0, np.zeros(T), np.zeros(T)
    obj = pulp.value(m.objective) or 0.0
    ch_v = np.array([ch[t].value() or 0.0 for t in range(T)])
    dh_v = np.array([dh[t].value() or 0.0 for t in range(T)])
    if progress_callback: progress_callback(100)
    return obj, ch_v, dh_v

# ── Gemeinsame Optimierung ───────────────────────────────────────────────────
def solve_joint(prices, pv_vec, ev_vec, cfgs, grid_kw, interval_h, progress_callback=None):
    n = len(cfgs)
    T = len(prices)
    grid_max = grid_kw * interval_h
    effs = [np.sqrt(c['eff_pct']) for c in cfgs]
    batt_maxs = [c['bat_kw'] * interval_h for c in cfgs]
    caps = [c['cap'] for c in cfgs]
    starts = [c['start_soc'] for c in cfgs]
    max_cycs = [c['max_cycles'] for c in cfgs]

    m = pulp.LpProblem("BESS_Joint", pulp.LpMaximize)
    c_vars, d_vars, ch_vars, dh_vars, soc_vars = {}, {}, {}, {}, {}
    for i in range(n):
        c_vars[i] = pulp.LpVariable.dicts(f"c{i}", range(T), cat="Binary")
        d_vars[i] = pulp.LpVariable.dicts(f"d{i}", range(T), cat="Binary")
        ch_vars[i] = pulp.LpVariable.dicts(f"ch{i}", range(T), lowBound=0, upBound=batt_maxs[i])
        dh_vars[i] = pulp.LpVariable.dicts(f"dh{i}", range(T), lowBound=0, upBound=batt_maxs[i])
        soc_vars[i]= pulp.LpVariable.dicts(f"soc{i}",range(T), lowBound=0, upBound=caps[i])

    # objective
    m += pulp.lpSum(prices[t] * pulp.lpSum(dh_vars[i][t] - ch_vars[i][t] for i in range(n))
                    for t in range(T))

    for t in range(T):
        # individual battery constraints
        for i in range(n):
            m += c_vars[i][t] + d_vars[i][t] <= 1
            m += ch_vars[i][t] <= batt_maxs[i] * c_vars[i][t]
            m += dh_vars[i][t] <= batt_maxs[i] * d_vars[i][t]
            if cfgs[i]['mode'] == 'Single Use':
                m += ch_vars[i][t] >= interval_h * c_vars[i][t]
                m += dh_vars[i][t] >= interval_h * d_vars[i][t]
            prev = starts[i] if t == 0 else soc_vars[i][t-1]
            m += soc_vars[i][t] == prev + effs[i] * ch_vars[i][t] - dh_vars[i][t] / effs[i]

        # grid hierarchy: SU first, MU uses remainder
        su_idx = [i for i,c in enumerate(cfgs) if c['mode']=='Single Use']
        mu_idx = [i for i,c in enumerate(cfgs) if c['mode']=='Multi Use']
        su_flow = pulp.lpSum(dh_vars[i][t] - ch_vars[i][t] for i in su_idx)
        mu_flow = pulp.lpSum(dh_vars[i][t] - ch_vars[i][t] for i in mu_idx)
        m += su_flow <= grid_max
        m += mu_flow <= grid_max - su_flow

        if progress_callback and t % max(1, T//50) == 0:
            progress_callback(5 + int(45 * t / T))

    # cycle constraints
    for i in range(n):
        m += pulp.lpSum((ch_vars[i][t] + dh_vars[i][t])/(2*caps[i]) for t in range(T)) <= max_cycs[i]

    if progress_callback: progress_callback(50)
    status = pulp.PULP_CBC_CMD(msg=False, timeLimit=120).solve(m)
    if progress_callback: progress_callback(90)

    if status != pulp.LpStatusOptimal:
        st.warning(f"Joint Solver Status: {pulp.LpStatus[status]}")
        return 0.0, [np.zeros(T) for _ in range(n)], [np.zeros(T) for _ in range(n)]

    obj = pulp.value(m.objective) or 0.0
    chs = [np.array([ch_vars[i][t].value() or 0.0 for t in range(T)]) for i in range(n)]
    dhs = [np.array([dh_vars[i][t].value() or 0.0 for t in range(T)]) for i in range(n)]
    if progress_callback: progress_callback(100)
    return obj, chs, dhs

# ── Hilfsfunktionen: Align & Validate ─────────────────────────────────────────
def align_timestamps(price_df, pv_df, ev_df):
    all_ts = pd.concat([price_df['Zeitstempel'], pv_df['Zeitstempel'], ev_df['Zeitstempel']])
    idx = all_ts.drop_duplicates().sort_values().reset_index(drop=True)
    base = pd.DataFrame({'Zeitstempel': idx})
    p = base.merge(price_df, on='Zeitstempel', how='left').fillna(method='ffill').fillna(0)
    v = base.merge(pv_df, on='Zeitstempel', how='left').fillna(method='ffill').fillna(0)
    e = base.merge(ev_df, on='Zeitstempel', how='left').fillna(method='ffill').fillna(0)
    return p, v, e

def validate_data(df, name):
    if df['Zeitstempel'].isna().any():
        return False, f"Fehler in {name}: ungültige Zeitstempel"
    return True, None

# ── Simulation Wrapper ──────────────────────────────────────────────────────
def run_sim():
    price_df = load_price_df(price_file)
    pv_df = load_pv_df(pv_file)
    ev_df = load_ev_df(ev_file)

    for df, nm in [(price_df,'Preis'), (pv_df,'PV'), (ev_df,'EV')]:
        ok, msg = validate_data(df, nm)
        if not ok:
            st.error(msg)
            st.stop()

    price_df, pv_df, ev_df = align_timestamps(price_df, pv_df, ev_df)

    ts = price_df['Zeitstempel']
    prices = price_df['Preis_€/MWh'].to_numpy() / 1000.0
    pv = pv_df['PV_kWh'].to_numpy()
    ev = ev_df['EV_kWh'].to_numpy()
    interval_h = ts.diff().mode()[0].total_seconds() / 3600.0

    free_res = []
    for cfg in configs:
        free_res.append(solve_battery(prices, pv, ev, cfg, grid_kw, interval_h, set_progress))

    joint_res = solve_joint(prices, pv, ev, configs, grid_kw, interval_h, set_progress)
    return ts, prices, pv, ev, free_res, joint_res, interval_h

# ── Streamlit-UI ─────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("BESS: Single vs. Multi Use & Grid Constraint")

price_file = st.sidebar.file_uploader("Preise (Zeit;€/MWh)", type=["csv","xls","xlsx"])
pv_file    = st.sidebar.file_uploader("PV-Lastgang (Zeit;kWh)", type=["csv","xls","xlsx"])
ev_file    = st.sidebar.file_uploader("EV-Lastgang (Zeit;kWh)", type=["csv","xls","xlsx"])
enable2    = st.sidebar.checkbox("Zweite Batterie aktivieren", True)

configs = []
def make_cfg(idx):
    st.sidebar.markdown(f"**Batterie {idx}**")
    mode  = st.sidebar.selectbox(f"Modus {idx}",["Single Use","Multi Use"], key=f"m{idx}")
    start = st.sidebar.number_input(f"StartSoC {idx} (kWh)", 0.0, 1e6, 0.0, key=f"s{idx}")
    cap   = st.sidebar.number_input(f"Kapazität {idx} (kWh)", 0.1, 1e6, 4472.0, key=f"c{idx}")
    bkw   = st.sidebar.number_input(f"Leistung {idx} (kW)", 0.1, 1e6, 559.0, key=f"p{idx}")
    eff   = st.sidebar.number_input(f"RT-Eff {idx} (%)", 1.0, 100.0, 91.0, key=f"e{idx}")/100.0
    cyc   = st.sidebar.number_input(f"Zyklen/Jahr {idx}", 0.0,1e4,548.0, key=f"y{idx}")
    return {"mode":mode, "start_soc":start, "cap":cap, "bat_kw":bkw, "eff_pct":eff, "max_cycles":cyc}

configs.append(make_cfg(1))
if enable2:
    configs.append(make_cfg(2))

grid_kw = st.sidebar.number_input("Netzanschluss (kW)", 0.1, 1e6, 757.5)

if st.sidebar.button("▶️ Simulation starten"):
    if not (price_file and pv_file and ev_file):
        st.sidebar.error("Bitte alle Dateien hochladen.")
    else:
        try:
            with st.spinner("Simulation läuft…"):
                st.session_state['res'] = run_sim()
            st.success("Simulation abgeschlossen!")
        except Exception as e:
            st.error(f"Fehler in Simulation: {e}")
            st.text(traceback.format_exc())

if 'res' not in st.session_state:
    st.info("Bitte Dateien hochladen und Simulation starten.")
    st.stop()

# Ergebnisse extrahieren
(ts, prices, pv, ev, free_res, joint_res, iv) = st.session_state['res']
tj, jchs, jdhs = joint_res

# Ergebnisanzeigen
col1, col2 = st.columns(2)
with col1:
    st.subheader("Einzeloptimierung (Free)")
    total_free = sum(obj for obj,_,_ in free_res)
    for i, (cfg, (obj,_,_)) in enumerate(zip(configs, free_res), 1):
        st.metric(f"Batterie {i} ({cfg['mode']}) Gewinn", fmt_euro(obj))
    st.metric("Gesamt (Free)", fmt_euro(total_free))

with col2:
    st.subheader("Gemeinsame Optimierung (Joint)")
    st.metric("Gesamtgewinn", fmt_euro(tj))
    for i, cfg in enumerate(configs, 1):
        indiv = float(np.dot(prices, jdhs[i-1] - jchs[i-1]))
        st.metric(f"Batterie {i} Anteil", fmt_euro(indiv))

# Vergleichsmesswerte
improvement = tj - total_free
improvement_pct = (improvement / abs(total_free) * 100) if total_free != 0 else 0
st.subheader("Vergleich")
metric_cols = st.columns(3)
metric_cols[0].metric("Verbesserung (absolut)", fmt_euro(improvement))
metric_cols[1].metric("Verbesserung (%)", f"{improvement_pct:.2f}%")
util = np.mean((pv + ev + sum(jchs) - sum(jdhs)) / (grid_kw * iv)) * 100
metric_cols[2].metric("Netzauslastung (Ø)", f"{util:.1f}%")

# Detaillierte Tabelle
out = pd.DataFrame({'Zeit': ts, 'PV_kWh': pv, 'EV_kWh': ev})
for i in range(len(configs)):
    out[f'Ch{i+1}'] = jchs[i]
    out[f'Dh{i+1}'] = jdhs[i]
out['Netzlast_kWh'] = pv + ev + sum(jchs) - sum(jdhs)
out['Netzlast_%'] = out['Netzlast_kWh'] / (grid_kw * iv) * 100

violations = (out['Netzlast_kWh'] > grid_kw * iv).sum()
if violations > 0:
    st.warning(f"⚠️ {violations} Zeitpunkte mit Netzüberlastung gefunden!")

st.subheader("Ergebnis-Tabelle")
st.dataframe(out)

# Download
buf = BytesIO()
out.to_excel(buf, index=False, engine='openpyxl')
buf.seek(0)
st.download_button(
    "📥 Ergebnisse herunterladen",
    data=buf,
    file_name=f"bess_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    key="download_results"
)
