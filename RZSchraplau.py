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

# ── Daten-Lader (generisch) ─────────────────────────────────────────────────────────
def load_time_series(upl, usecols, names):
    """
    Universeller Loader für CSV/XLS[X]:
      upl: UploadedFile
      usecols: Liste der Spaltenindices oder Namen
      names: Liste von neuen Spaltennamen (Zeit, Wert)
    """
    if upl.name.lower().endswith('.csv'):
        df = pd.read_csv(upl, sep=';', decimal=',', usecols=usecols,
                         names=names, header=0)
    else:
        df = pd.read_excel(upl, usecols=usecols, names=names,
                           header=0, engine='openpyxl')
    df[names[0]] = pd.to_datetime(df[names[0]], dayfirst=True)
    df[names[1]] = pd.to_numeric(df[names[1]], errors='raise')
    return df

# Spezifische Loader bauen auf generisch auf
def load_price_df(upl):
    return load_time_series(upl, usecols=[0,1], names=['Zeitstempel','Preis_€/MWh'])

def load_pv_df(upl):
    return load_time_series(upl, usecols=[0,1], names=['Zeitstempel','PV_kWh'])

def load_ev_df(upl):
    return load_time_series(upl, usecols=[0,1], names=['Zeitstempel','EV_kWh'])

# ── Einzelbatterie Solver ─────────────────────────────────────────────────────
def solve_battery(prices, pv_vec, ev_vec, cfg, grid_kw, interval_h, progress_callback=None):
    T = len(prices)
    batt_max = cfg['bat_kw'] * interval_h
    grid_max = grid_kw * interval_h
    cap = cfg['cap']
    eff = cfg['eff_pct']**0.5
    max_cyc = cfg['max_cycles']

    m = pulp.LpProblem("BESS", pulp.LpMaximize)
    c = pulp.LpVariable.dicts("c", range(T), cat="Binary")
    d = pulp.LpVariable.dicts("d", range(T), cat="Binary")
    ch = pulp.LpVariable.dicts("ch", range(T), lowBound=0, upBound=batt_max)
    dh = pulp.LpVariable.dicts("dh", range(T), lowBound=0, upBound=batt_max)
    soc= pulp.LpVariable.dicts("soc",range(T), lowBound=0, upBound=cap)

    m += pulp.lpSum(prices[t]*(dh[t]-ch[t]) for t in range(T))
    
    for t in range(T):
        # Binary constraint: can't charge and discharge simultaneously
        m += c[t]+d[t] <= 1
        
        # Charge constraints
        m += ch[t] <= batt_max*c[t]
        m += ch[t] >= interval_h*c[t] if cfg['mode'] == 'Single Use' else 0
        
        # Discharge constraints
        m += dh[t] <= batt_max*d[t]
        m += dh[t] >= interval_h*d[t] if cfg['mode'] == 'Single Use' else 0
        
        # Grid constraint: PV + EV + battery operations <= grid capacity
        # For single battery: only consider its own operations
        m += pv_vec[t] + ev_vec[t] + ch[t] <= grid_max
        
        # SOC dynamics
        prev = cfg['start_soc'] if t==0 else soc[t-1]
        m += soc[t] == prev + eff*ch[t] - dh[t]/eff
        
        if progress_callback and t%(max(1,T//50))==0:
            progress_callback(5+int(45*t/T))
    
    # Cycle constraint
    m += pulp.lpSum((ch[t]+dh[t])/(2*cap) for t in range(T)) <= max_cyc
    
    if progress_callback: progress_callback(50)
    
    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=120)
    status = solver.solve(m)
    
    if progress_callback: progress_callback(90)
    
    # Check if solution was found
    if status != pulp.LpStatusOptimal:
        st.warning(f"Solver Status: {pulp.LpStatus[status]} - möglicherweise keine optimale Lösung gefunden")
        obj = 0.0
        ch_v = np.zeros(T)
        dh_v = np.zeros(T)
    else:
        obj = pulp.value(m.objective) or 0.0
        ch_v = np.array([ch[t].value() or 0.0 for t in range(T)])
        dh_v = np.array([dh[t].value() or 0.0 for t in range(T)])
    
    if progress_callback: progress_callback(100)
    return obj, ch_v, dh_v

# ── Gemeinsamer Solver ───────────────────────────────────────────────────────
def solve_joint(prices, pv_vec, ev_vec, cfgs, grid_kw, interval_h, progress_callback=None):
    n = len(cfgs)
    T = len(prices)
    grid_max = grid_kw * interval_h
    effs = [c['eff_pct']**0.5 for c in cfgs]
    batt_maxs = [c['bat_kw']*interval_h for c in cfgs]
    caps = [c['cap'] for c in cfgs]
    starts = [c['start_soc'] for c in cfgs]
    max_cycs = [c['max_cycles'] for c in cfgs]

    m = pulp.LpProblem("BESS_Joint", pulp.LpMaximize)
    
    # Variables for each battery
    c_vars={}; d_vars={}; ch_vars={}; dh_vars={}; soc_vars={}
    for i in range(n):
        c_vars[i] = pulp.LpVariable.dicts(f"c{i}", range(T), cat="Binary")
        d_vars[i] = pulp.LpVariable.dicts(f"d{i}", range(T), cat="Binary")
        ch_vars[i] = pulp.LpVariable.dicts(f"ch{i}", range(T), lowBound=0, upBound=batt_maxs[i])
        dh_vars[i] = pulp.LpVariable.dicts(f"dh{i}", range(T), lowBound=0, upBound=batt_maxs[i])
        soc_vars[i] = pulp.LpVariable.dicts(f"soc{i}", range(T), lowBound=0, upBound=caps[i])
    
    # Objective: maximize total revenue
    m += pulp.lpSum(prices[t] * pulp.lpSum(dh_vars[i][t] - ch_vars[i][t] for i in range(n)) for t in range(T))

    for t in range(T):
        for i in range(n):
            # Binary constraint: can't charge and discharge simultaneously
            m += c_vars[i][t] + d_vars[i][t] <= 1
            
            # Charge constraints
            m += ch_vars[i][t] <= batt_maxs[i] * c_vars[i][t]
            if cfgs[i]['mode'] == 'Single Use':
                m += ch_vars[i][t] >= interval_h * c_vars[i][t]
            
            # Discharge constraints
            m += dh_vars[i][t] <= batt_maxs[i] * d_vars[i][t]
            if cfgs[i]['mode'] == 'Single Use':
                m += dh_vars[i][t] >= interval_h * d_vars[i][t]
            
            # SOC dynamics
            prev = starts[i] if t==0 else soc_vars[i][t-1]
            m += soc_vars[i][t] == prev + effs[i]*ch_vars[i][t] - dh_vars[i][t]/effs[i]
        
        # CORRECTED: Grid constraint including PV and EV loads
        m += pv_vec[t] + ev_vec[t] + pulp.lpSum(ch_vars[i][t] for i in range(n)) <= grid_max
        
        if progress_callback and t%(max(1,T//50))==0:
            progress_callback(5+int(45*t/T))
    
    # Cycle constraints for each battery
    for i in range(n):
        m += pulp.lpSum((ch_vars[i][t]+dh_vars[i][t])/(2*caps[i]) for t in range(T)) <= max_cycs[i]
    
    if progress_callback: progress_callback(50)
    
    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=120)
    status = solver.solve(m)
    
    if progress_callback: progress_callback(90)
    
    # Check if solution was found
    if status != pulp.LpStatusOptimal:
        st.warning(f"Joint Solver Status: {pulp.LpStatus[status]} - möglicherweise keine optimale Lösung gefunden")
        obj = 0.0
        chs = [np.zeros(T) for _ in range(n)]
        dhs = [np.zeros(T) for _ in range(n)]
    else:
        obj = pulp.value(m.objective) or 0.0
        chs = [np.array([ch_vars[i][t].value() or 0.0 for t in range(T)]) for i in range(n)]
        dhs = [np.array([dh_vars[i][t].value() or 0.0 for t in range(T)]) for i in range(n)]
    
    if progress_callback: progress_callback(100)
    return obj, chs, dhs

# ── Simulation Wrapper ──────────────────────────────────────────────────────
def validate_data(price_df, pv_df, ev_df):
    """Validate that all dataframes have consistent timestamps"""
    if not all(df['Zeitstempel'].equals(price_df['Zeitstempel']) for df in [pv_df, ev_df]):
        return False, "Zeitstempel in den Dateien sind nicht identisch!"
    
    if not (len(price_df) == len(pv_df) == len(ev_df)):
        return False, "Ungleiche Zeitreihenlängen."
    
    # Check for negative values where they shouldn't be
    if (price_df['Preis_€/MWh'] < -1000).any():
        return False, "Unrealistisch niedrige Preise gefunden."
    
    if (pv_df['PV_kWh'] < 0).any() or (ev_df['EV_kWh'] < 0).any():
        return False, "Negative PV- oder EV-Werte gefunden."
    
    return True, "OK"

def run_sim():
    # Reset progress
    set_progress(0)
    
    # Load data
    price_df = load_price_df(price_file)
    pv_df = load_pv_df(pv_file)
    ev_df = load_ev_df(ev_file)
    
    # Validate data
    is_valid, error_msg = validate_data(price_df, pv_df, ev_df)
    if not is_valid:
        st.error(error_msg)
        st.stop()
    
    # Extract time series
    ts = price_df['Zeitstempel']
    prices = price_df['Preis_€/MWh'].to_numpy() / 1000.0  # Convert to €/kWh
    pv = pv_df['PV_kWh'].to_numpy()
    ev = ev_df['EV_kWh'].to_numpy()
    
    # Calculate interval
    interval_h = (ts[1] - ts[0]).total_seconds() / 3600.0
    
    # Validate configurations
    for i, cfg in enumerate(configs):
        if cfg['cap'] <= 0 or cfg['bat_kw'] <= 0:
            st.error(f"Batterie {i+1}: Kapazität und Leistung müssen positiv sein.")
            st.stop()
        if cfg['eff_pct'] <= 0 or cfg['eff_pct'] > 1:
            st.error(f"Batterie {i+1}: Effizienz muss zwischen 0 und 100% liegen.")
            st.stop()
    
    if grid_kw <= 0:
        st.error("Netzanschluss muss positiv sein.")
        st.stop()
    
    # Run individual optimizations
    free_res = []
    for i, cfg in enumerate(configs):
        set_progress(0)
        result = solve_battery(prices, pv, ev, cfg, grid_kw, interval_h, set_progress)
        free_res.append(result)
    
    # Run joint optimization
    set_progress(0)
    joint_res = solve_joint(prices, pv, ev, configs, grid_kw, interval_h, set_progress)
    
    return ts, prices, pv, ev, free_res, joint_res, interval_h

# ── Streamlit-UI ─────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("BESS: Single vs. Multi Use & Grid Constraint")

st.sidebar.markdown("## Dateien")
price_file = st.sidebar.file_uploader("Preise (Zeit;€/MWh)", type=["csv","xls","xlsx"])
pv_file = st.sidebar.file_uploader("PV-Lastgang (Zeit;kWh)", type=["csv","xls","xlsx"])
ev_file = st.sidebar.file_uploader("EV-Lastgang (Zeit;kWh)", type=["csv","xls","xlsx"])

st.sidebar.markdown("## Parameter")
enable2 = st.sidebar.checkbox("Zweite Batterie aktivieren", True)

configs = []
def make_cfg(idx):
    st.sidebar.markdown(f"**Batterie {idx}**")
    mode = st.sidebar.selectbox(
        f"Modus {idx}", 
        ["Single Use", "Multi Use"], 
        key=f"m{idx}",
        help="Single Use: Batterie muss mindestens 1h laden/entladen. Multi Use: Beliebige Leistung möglich."
    )
    start = st.sidebar.number_input(f"StartSoC {idx} (kWh)", 0.0, 1e6, 0.0, key=f"s{idx}")
    cap = st.sidebar.number_input(f"Kapazität {idx} (kWh)", 0.1, 1e6, 4472.0, key=f"c{idx}")
    bkw = st.sidebar.number_input(f"Leistung {idx} (kW)", 0.1, 1e6, 559.0, key=f"p{idx}")
    eff = st.sidebar.number_input(f"RT-Eff {idx} (%)", 1.0, 100.0, 91.0, key=f"e{idx}") / 100.0
    cyc = st.sidebar.number_input(f"Zyklen/Jahr {idx}", 0.0, 1e4, 548.0, key=f"y{idx}")
    return {
        "mode": mode, 
        "start_soc": start, 
        "cap": cap, 
        "bat_kw": bkw, 
        "eff_pct": eff, 
        "max_cycles": cyc
    }

configs.append(make_cfg(1))
if enable2: 
    configs.append(make_cfg(2))

grid_kw = st.sidebar.number_input("Netzanschluss (kW)", 0.1, 1e6, 757.5)

if st.sidebar.button("▶️ Simulation starten"):
    if not (price_file and pv_file and ev_file):
        st.sidebar.error("Bitte alle Dateien hochladen.")
    else:
        try:
            with st.spinner("Simulation läuft..."):
                st.session_state['res'] = run_sim()
            st.success("Simulation erfolgreich abgeschlossen!")
        except Exception as e:
            st.error(f"Fehler in Simulation: {e}")
            import traceback
            st.text(traceback.format_exc())

if 'res' not in st.session_state:
    st.info("Bitte laden Sie alle Dateien hoch und starten Sie die Simulation.")
    st.stop()

# Extract results
(ts, prices, pv, ev, free_res, jres, iv) = st.session_state['res']
total_joint_obj, joint_chs, joint_dhs = jres

# Display results
col1, col2 = st.columns(2)

with col1:
    st.subheader("Einzeloptimierung (Free)")
    total_free = 0
    for i, (cfg, (obj, _, _)) in enumerate(zip(configs, free_res), start=1):
        st.metric(f"Batterie {i} ({cfg['mode']})", fmt_euro(obj))
        total_free += obj
    st.metric("**Gesamt (Free)**", fmt_euro(total_free))

with col2:
    st.subheader("Gemeinsame Optimierung (Joint)")
    st.metric("**Gesamtgewinn**", fmt_euro(total_joint_obj))
    for i, cfg in enumerate(configs, start=1):
        individual_profit = float(np.dot(prices, joint_dhs[i-1] - joint_chs[i-1]))
        st.metric(f"Batterie {i} Anteil", fmt_euro(individual_profit))

# Improvement calculation
improvement = total_joint_obj - total_free
improvement_pct = (improvement / abs(total_free) * 100) if total_free != 0 else 0

st.subheader("Vergleich")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Verbesserung (absolut)", fmt_euro(improvement))
with col2:
    st.metric("Verbesserung (%)", f"{improvement_pct:.2f}%")
with col3:
    grid_utilization = np.mean((pv + ev + sum(joint_chs) - sum(joint_dhs)) / (grid_kw * iv)) * 100
    st.metric("Netzauslastung (Ø)", f"{grid_utilization:.1f}%")

# Create results dataframe
st.subheader("Detaillierte Ergebnisse")
out = pd.DataFrame({
    'Zeit': ts,
    'Preis_€/kWh': prices,
    'PV_kWh': pv,
    'EV_kWh': ev
})

for i in range(len(configs)):
    out[f'B{i+1}_Laden_kWh'] = joint_chs[i]
    out[f'B{i+1}_Entladen_kWh'] = joint_dhs[i]

# Add grid load calculation
total_grid_load = pv + ev + sum(joint_chs) - sum(joint_dhs)
out['Netzlast_kWh'] = total_grid_load
out['Netzlast_%'] = (total_grid_load / (grid_kw * iv)) * 100

st.dataframe(out)

# Download button
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
