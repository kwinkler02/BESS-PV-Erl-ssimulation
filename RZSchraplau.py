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

# ── Einzelbatterie Solver ───────────────────────────────────────────────────── ─────────────────────────────────────────────────────
def solve_battery(prices, pv_vec, cfg, grid_kw, interval_h, progress_callback=None):
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
        m += c[t]+d[t] <= 1
        m += ch[t] <= batt_max*c[t]
        m += ch[t] >= interval_h*c[t]
        m += dh[t] <= batt_max*d[t]
        m += dh[t] >= interval_h*d[t]
        m += pv_vec[t] + ch[t]+dh[t] <= grid_max
        prev = cfg['start_soc'] if t==0 else soc[t-1]
        m += soc[t] == prev + eff*ch[t] - dh[t]/eff
        if progress_callback and t%(max(1,T//50))==0:
            progress_callback(5+int(45*t/T))
    m += pulp.lpSum((ch[t]+dh[t])/(2*cap) for t in range(T)) <= max_cyc
    if progress_callback: progress_callback(50)
    pulp.PULP_CBC_CMD(msg=False,timeLimit=120).solve(m)
    if progress_callback: progress_callback(90)
    obj = pulp.value(m.objective) or 0.0
    ch_v = np.array([ch[t].value() for t in range(T)])
    dh_v = np.array([dh[t].value() for t in range(T)])
    if progress_callback: progress_callback(100)
    return obj, ch_v, dh_v

# ── Gemeinsamer Solver ───────────────────────────────────────────────────────
def solve_joint(prices, pv_vecs, ev_cons, cfgs, grid_kw, interval_h, progress_callback=None):
    n = len(cfgs); T = len(prices)
    grid_max = grid_kw*interval_h
    effs = [c['eff_pct']**0.5 for c in cfgs]
    batt_maxs = [c['bat_kw']*interval_h for c in cfgs]
    caps = [c['cap'] for c in cfgs]
    starts = [c['start_soc'] for c in cfgs]
    max_cycs = [c['max_cycles'] for c in cfgs]

    m = pulp.LpProblem("BESS_Joint",pulp.LpMaximize)
    c_vars={}; d_vars={}; ch_vars={}; dh_vars={}; soc_vars={}
    for i in range(n):
        c_vars[i]=pulp.LpVariable.dicts(f"c{i}",range(T),cat="Binary")
        d_vars[i]=pulp.LpVariable.dicts(f"d{i}",range(T),cat="Binary")
        ch_vars[i]=pulp.LpVariable.dicts(f"ch{i}",range(T),lowBound=0,upBound=batt_maxs[i])
        dh_vars[i]=pulp.LpVariable.dicts(f"dh{i}",range(T),lowBound=0,upBound=batt_maxs[i])
        soc_vars[i]=pulp.LpVariable.dicts(f"soc{i}",range(T),lowBound=0,upBound=caps[i])
    m += pulp.lpSum(prices[t]*pulp.lpSum(dh_vars[i][t]-ch_vars[i][t] for i in range(n)) for t in range(T))

    for t in range(T):
        for i in range(n):
            m += c_vars[i][t]+d_vars[i][t] <= 1
            m += ch_vars[i][t] <= batt_maxs[i]*c_vars[i][t]
            m += ch_vars[i][t] >= interval_h*c_vars[i][t]
            m += dh_vars[i][t] <= batt_maxs[i]*d_vars[i][t]
            m += dh_vars[i][t] >= interval_h*d_vars[i][t]
            prev = starts[i] if t==0 else soc_vars[i][t-1]
            m += soc_vars[i][t] == prev + effs[i]*ch_vars[i][t] - dh_vars[i][t]/effs[i]
        m += pulp.lpSum(ch_vars[i][t]+dh_vars[i][t] for i in range(n)) <= grid_max
        if progress_callback and t%(max(1,T//50))==0:
            progress_callback(5+int(45*t/T))
    for i in range(n):
        m += pulp.lpSum((ch_vars[i][t]+dh_vars[i][t])/(2*caps[i]) for t in range(T)) <= max_cycs[i]
    if progress_callback: progress_callback(50)
    pulp.PULP_CBC_CMD(msg=False,timeLimit=120).solve(m)
    if progress_callback: progress_callback(90)
    obj = pulp.value(m.objective) or 0.0
    chs = [np.array([ch_vars[i][t].value() for t in range(T)]) for i in range(n)]
    dhs = [np.array([dh_vars[i][t].value() for t in range(T)]) for i in range(n)]
    if progress_callback: progress_callback(100)
    return obj, chs, dhs

# ── Simulation Wrapper ──────────────────────────────────────────────────────
def run_sim():
    price_df = load_price_df(price_file)
    pv_df = load_pv_df(pv_file)
    ev_df = load_ev_df(ev_file)
    ts = price_df['Zeitstempel']
    prices=price_df['Preis_€/MWh'].to_numpy()/1000.0
    pv = pv_df['PV_kWh'].to_numpy()
    ev = ev_df['EV_kWh'].to_numpy()
    if not(len(pv)==len(prices)==len(ev)):
        st.error("Ungleiche Zeitreihenlängen.")
        st.stop()
    interval_h = (ts[1]-ts[0]).total_seconds()/3600.0
    free_res=[solve_battery(prices,pv,cfg,grid_kw,interval_h,set_progress) for cfg in configs]
    joint_res=solve_joint(prices,[pv]*len(configs),ev,configs,grid_kw,interval_h,set_progress)
    return ts,prices,pv,ev,free_res,joint_res,interval_h

# ── Streamlit-UI ─────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("BESS: Single vs. Multi Use & Grid Constraint")

st.sidebar.markdown("## Dateien")
price_file=st.sidebar.file_uploader("Preise (Zeit;€/MWh)",type=["csv","xls","xlsx"])
pv_file   =st.sidebar.file_uploader("PV-Lastgang (Zeit;kWh)",type=["csv","xls","xlsx"])
ev_file   =st.sidebar.file_uploader("EV-Lastgang (Zeit;kWh)",type=["csv","xls","xlsx"])

st.sidebar.markdown("## Parameter")
enable2=st.sidebar.checkbox("Zweite Batterie aktivieren",True)
configs=[]
def make_cfg(idx):
    mode=st.sidebar.selectbox(f"Modus{idx}",["Single Use","Multi Use"], key=f"m{idx}")
    start=st.sidebar.number_input(f"StartSoC{idx} (kWh)",0.0,1e6,0.0,key=f"s{idx}")
    cap  =st.sidebar.number_input(f"Kapazität{idx} (kWh)",0.0,1e6,4472.0,key=f"c{idx}")
    bkw  =st.sidebar.number_input(f"Leistung{idx} (kW)",0.0,1e6,559.0,key=f"p{idx}")
    eff  =st.sidebar.number_input(f"RT-Eff{idx} (%)",0.0,100.0,91.0,key=f"e{idx}")/100.0
    cyc  =st.sidebar.number_input(f"Zyklen/Jahr{idx}",0.0,1e4,548.0,key=f"y{idx}")
    return {"mode":mode,"start_soc":start,"cap":cap,"bat_kw":bkw,"eff_pct":eff,"max_cycles":cyc}
configs.append(make_cfg(1))
if enable2: configs.append(make_cfg(2))
grid_kw=st.sidebar.number_input("Netzanschluss (kW)",0.0,1e6,757.5)

if st.sidebar.button("▶️ Simulation starten"):
    if not(price_file and pv_file and ev_file): st.sidebar.error("Bitte alle Dateien hochladen.")
    else: st.session_state['res']=run_sim()

if 'res' not in st.session_state:
    st.info("Bitte Simulation starten.")
    st.stop()
(ts,prices,pv,ev,free_res,jres,iv)=st.session_state['res']
total_joint_obj,joint_chs,joint_dhs=jres

st.subheader("Free-Ergebnisse")
for i,(cfg,(obj,_,_)) in enumerate(zip(configs,free_res),start=1): st.metric(f"B{i} ({cfg['mode']}) Gewinn",fmt_euro(obj))

st.subheader("Joint-Ergebnisse")
st.metric("Gesamtgewinn",fmt_euro(total_joint_obj))
for i,cfg in enumerate(configs,start=1):
    ind=float(np.dot(prices,joint_dhs[i-1]-joint_chs[i-1]))
    st.metric(f"B{i} Joint Gewinn",fmt_euro(ind))

out=pd.DataFrame({'Zeit':ts,'PV_kWh':pv,'EV_kWh':ev})
for i in range(len(configs)):
    out[f'Ch{i+1}']=joint_chs[i]
    out[f'Dh{i+1}']=joint_dhs[i]

# MU/SU-Detaillierung & GridLoad wie gehabt

st.subheader("Ergebnis-Tabelle")
st.dataframe(out)

buf=BytesIO(); out.to_excel(buf,index=False,engine='openpyxl'); buf.seek(0)
st.download_button("📥 Ergebnis",data=buf,file_name="res.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key="dl")
