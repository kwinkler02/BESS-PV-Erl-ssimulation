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

# Spezifische Loader

def load_price_df(upl): return load_time_series(upl, [0,1], ['Zeitstempel','Preis_€/MWh'])
def load_pv_df(upl):    return load_time_series(upl, [0,1], ['Zeitstempel','PV_kWh'])
def load_ev_df(upl):    return load_time_series(upl, [0,1], ['Zeitstempel','EV_kWh'])

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
        m += c[t]+d[t] <= 1
        m += ch[t] <= batt_max*c[t]
        if cfg['mode']=='Single Use': m += ch[t]>=interval_h*c[t]
        m += dh[t] <= batt_max*d[t]
        if cfg['mode']=='Single Use': m += dh[t]>=interval_h*d[t]
        m += pv_vec[t] + ev_vec[t] + ch[t] - dh[t] <= grid_max
        prev = cfg['start_soc'] if t==0 else soc[t-1]
        m += soc[t] == prev + eff*ch[t] - dh[t]/eff
        if progress_callback and t%(max(1,T//50))==0:
            progress_callback(5+int(45*t/T))
    m += pulp.lpSum((ch[t]+dh[t])/(2*cap) for t in range(T)) <= max_cyc
    if progress_callback: progress_callback(50)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=120)
    status = solver.solve(m)
    if progress_callback: progress_callback(90)
    if status!=pulp.LpStatusOptimal:
        st.warning(f"Solver Status: {pulp.LpStatus[status]}")
        obj, ch_v, dh_v = 0.0, np.zeros(T), np.zeros(T)
    else:
        obj = pulp.value(m.objective) or 0.0
        ch_v = np.array([ch[t].value() or 0.0 for t in range(T)])
        dh_v = np.array([dh[t].value() or 0.0 for t in range(T)])
    if progress_callback: progress_callback(100)
    return obj, ch_v, dh_v

# ── Gemeinsamer Solver ───────────────────────────────────────────────────────
def solve_joint(prices, pv_vec, ev_vec, cfgs, grid_kw, interval_h, progress_callback=None):
    n,lenp = len(cfgs), len(prices)
    grid_max = grid_kw*interval_h
    effs = [c['eff_pct']**0.5 for c in cfgs]
    batt_maxs=[c['bat_kw']*interval_h for c in cfgs]
    caps = [c['cap'] for c in cfgs]
    starts=[c['start_soc'] for c in cfgs]
    max_cyc = [c['max_cycles'] for c in cfgs]

    m = pulp.LpProblem("BESS_Joint", pulp.LpMaximize)
    c_vars,d_vars,ch_vars,dh_vars,soc_vars = {},{},{},{},{}
    for i in range(n):
        c_vars[i]=pulp.LpVariable.dicts(f"c{i}", range(lenp), cat="Binary")
        d_vars[i]=pulp.LpVariable.dicts(f"d{i}", range(lenp), cat="Binary")
        ch_vars[i]=pulp.LpVariable.dicts(f"ch{i}", range(lenp), lowBound=0, upBound=batt_maxs[i])
        dh_vars[i]=pulp.LpVariable.dicts(f"dh{i}", range(lenp), lowBound=0, upBound=batt_maxs[i])
        soc_vars[i]=pulp.LpVariable.dicts(f"so{i}", range(lenp), lowBound=0, upBound=caps[i])

    m+= pulp.lpSum(prices[t]*pulp.lpSum(dh_vars[i][t]-ch_vars[i][t] for i in range(n)) for t in range(lenp))
    for t in range(lenp):
        for i in range(n):
            m+= c_vars[i][t]+d_vars[i][t]<=1
            m+= ch_vars[i][t]<=batt_maxs[i]*c_vars[i][t]
            if cfgs[i]['mode']=='Single Use': m+= ch_vars[i][t]>=interval_h*c_vars[i][t]
            m+= dh_vars[i][t]<=batt_maxs[i]*d_vars[i][t]
            if cfgs[i]['mode']=='Single Use': m+= dh_vars[i][t]>=interval_h*d_vars[i][t]
            prev=starts[i] if t==0 else soc_vars[i][t-1]
            m+= soc_vars[i][t]==prev+effs[i]*ch_vars[i][t]-dh_vars[i][t]/effs[i]
        total_batt = pulp.lpSum(ch_vars[i][t]-dh_vars[i][t] for i in range(n))
        m += pv_vec[t] + ev_vec[t] + total_batt <= grid_max
        if progress_callback and t%(max(1,lenp//50))==0:
            progress_callback(5+int(45*t/lenp))
    for i in range(n):
        m+= pulp.lpSum((ch_vars[i][t]+dh_vars[i][t])/(2*caps[i]) for t in range(lenp))<=max_cyc[i]
    if progress_callback: progress_callback(50)
    solver = pulp.PULP_CBC_CMD(msg=False,timeLimit=120)
    status=solver.solve(m)
    if progress_callback: progress_callback(90)
    if status!=pulp.LpStatusOptimal:
        st.warning(f"Joint Solver Status: {pulp.LpStatus[status]}")
        obj, chs, dhs = 0.0, [np.zeros(lenp) for _ in range(n)], [np.zeros(lenp) for _ in range(n)]
    else:
        obj=pulp.value(m.objective) or 0.0
        chs=[np.array([ch_vars[i][t].value() or 0.0 for t in range(lenp)]) for i in range(n)]
        dhs=[np.array([dh_vars[i][t].value() or 0.0 for t in range(lenp)]) for i in range(n)]
    if progress_callback: progress_callback(100)
    return obj, chs, dhs

# ── Simulation Wrapper & UI ─────────────────────────────────────────────────
def align_timestamps(price_df,pv_df,ev_df):
    all_ts=pd.concat([price_df['Zeitstempel'],pv_df['Zeitstempel'],ev_df['Zeitstempel']]).drop_duplicates().sort_values().reset_index(drop=True)
    base=pd.DataFrame({'Zeitstempel':all_ts})
    price=base.merge(price_df,on='Zeitstempel',how='left').fillna(method='ffill').fillna(0)
    pv=base.merge(pv_df,on='Zeitstempel',how='left').fillna(method='ffill').fillna(0)
    ev=base.merge(ev_df,on='Zeitstempel',how='left').fillna(method='ffill').fillna(0)
    return price,pv,ev

def validate_data(p_df,pv_df,ev_df):
    if p_df.empty or pv_df.empty or ev_df.empty: return False,"Leere Datei"
    if p_df['Zeitstempel'].isna().any(): return False,"Preis Zeitstempel fehlerhaft"
    if pv_df['Zeitstempel'].isna().any(): return False,"PV Zeitstempel fehlerhaft"
    if ev_df['Zeitstempel'].isna().any(): return False,"EV Zeitstempel fehlerhaft"
    if (pv_df['PV_kWh']<0).any() or (ev_df['EV_kWh']<0).any(): return False,"Neg. Last"
    return True,"OK"

def run_sim():
    set_progress(0)
    p_df=load_price_df(price_file)
    pv_df=load_pv_df(pv_file)
    ev_df=load_ev_df(ev_file)
    valid,msg=validate_data(p_df,pv_df,ev_df)
    if not valid: st.error(msg); st.stop()
    try:
        p_df,pv_df,ev_df=align_timestamps(p_df,pv_df,ev_df)
        st.info(f"Daten aligned: {len(p_df)} Zeitpunkte")
    except Exception as e:
        st.error(e); st.stop()
    ts=p_df['Zeitstempel']
    prices=p_df['Preis_€/MWh'].to_numpy()/1000.0
    pv=pv_df['PV_kWh'].to_numpy()
    ev=ev_df['EV_kWh'].to_numpy()
    interval_h=ts.diff().dropna().mode()[0].total_seconds()/3600.0
    st.info(f"Intervall: {interval_h} h")
    for i,cfg in enumerate(configs):
        if cfg['cap']<=0 or cfg['bat_kw']<=0 or cfg['eff_pct']<=0 or cfg['eff_pct']>1: st.error("Param Fehler"); st.stop()
    if grid_kw<=0: st.error("Grid Fehler"); st.stop()
    free=[]
    for cfg in configs: free.append(solve_battery(prices,pv,ev,cfg,grid_kw,interval_h,set_progress))
    joint=solve_joint(prices,pv,ev,configs,grid_kw,interval_h,set_progress)
    return ts,prices,pv,ev,free,joint,interval_h

st.set_page_config(layout="wide")
st.title("BESS: Single vs. Multi Use & Grid Constraint")
price_file=st.sidebar.file_uploader("Preise",type=["csv","xls","xlsx"])
pv_file=st.sidebar.file_uploader("PV-Last",type=["csv","xls","xlsx"])
ev_file=st.sidebar.file_uploader("EV-Last",type=["csv","xls","xlsx"])
enable2=st.sidebar.checkbox("Zweite Batterie",True)
configs=[]
for i in (1,2):
    if i==1 or enable2:
        st.sidebar.markdown(f"**Batterie {i}**")
        mode=st.sidebar.selectbox(f"Modus{i}",["Single Use","Multi Use"],key=f"m{i}")
        start=st.sidebar.number_input(f"StartSoC{i} (kWh)",0.0,1e6,0.0,key=f"s{i}")
        cap=st.sidebar.number_input(f"Kap{i} (kWh)",0.1,1e6,4472.0,key=f"c{i}")
        bkw=st.sidebar.number_input(f"Leistung{i} (kW)",0.1,1e6,559.0,key=f"p{i}")
        eff=st.sidebar.number_input(f"Eff{i} (%)",1.0,100.0,91.0,key=f"e{i}")/100.0
        cyc=st.sidebar.number_input(f"Zyklen{i}",0.0,1e4,548.0,key=f"y{i}")
        configs.append({"mode":mode,"start_soc":start,"cap":cap,"bat_kw":bkw,"eff_pct":eff,"max_cycles":cyc})
grid_kw=st.sidebar.number_input("Netzanschluss (kW)",0.1,1e6,757.5)
if st.sidebar.button("▶️ Simulation starten"):
    if not(price_file and pv_file and ev_file): st.sidebar.error("Bitte alle Dateien hochladen.")
    else:
        try: st.session_state['res']=run_sim(); st.success("Fertig!")
        except Exception as e: st.error(e)
if 'res' not in st.session_state: st.info("Bitte Dateien hochladen und starten."); st.stop()
(ts,prices,pv,ev,free_res,jres,iv)=st.session_state['res']
obj_joint,chs_joint,dhs_joint=jres
col1,col2=st.columns(2)
with col1:
    st.subheader("Einzeloptimierung")
    tot_free=0
    for i,(cfg,(obj,_,_)) in enumerate(zip(configs,free_res),start=1): st.metric(f"B{i} ({cfg['mode']})",fmt_euro(obj)); tot_free+=obj
    st.metric("Gesamt Free",fmt_euro(tot_free))
with col2:
    st.subheader("Gemeinsam")
    st.metric("Gesamt",fmt_euro(obj_joint))
    for i,cfg in enumerate(configs,1): ind=float(np.dot(prices,dhs_joint[i-1]-chs_joint[i-1])); st.metric(f"B{i} Anteil",fmt_euro(ind))
imp=obj_joint-tot_free; pct=(imp/abs(tot_free)*100 if tot_free!=0 else 0)
st.subheader("Vergleich")
col1,col2,col3=st.columns(3)
col1.metric("∆ absolut",fmt_euro(imp))
col2.metric("∆ %",f"{pct:.2f}%")
col3.metric("Netzauslastung","{:.1f}%".format(np.mean((pv+ev+sum(chs_joint)-sum(dhs_joint))/(grid_kw*iv))*100))
st.subheader("Ergebnisse")
out=pd.DataFrame({'Zeit':ts,'Preis_€/kWh':prices,'PV_kWh':pv,'EV_kWh':ev})
for i in range(len(configs)):
    out[f'B{i+1}_Laden_kWh']=chs_joint[i]
    out[f'B{i+1}_Entladen_kWh']=dhs_joint[i]
tot_load=pv+ev+sum(chs_joint)-sum(dhs_joint)
out['Netzlast_kWh']=tot_load
out['Netzlast_%']=(tot_load/(grid_kw*iv))*100
viol=(tot_load>(grid_kw*iv)).sum()
if viol>0: st.warning(f"⚠️ {viol} Überlastungen")
st.dataframe(out)
buf=BytesIO(); out.to_excel(buf,index=False,engine='openpyxl'); buf.seek(0)
st.download_button("📥 Laden",buf,f"res_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl")
