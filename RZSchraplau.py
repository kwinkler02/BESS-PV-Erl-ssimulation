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

# ── Flexibler Zeitstempel-Parser ───────────────────────────────────────────────
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
                    y = parts[2].split(' ')[0]
                    if len(y) == 2:
                        yy = int(y)
                        full = 2000+yy if yy<50 else 1900+yy
                        s = s.replace(f".{y} ", f".{full} ")
                parsed.append(pd.to_datetime(s, infer_datetime_format=True))
            except:
                parsed.append(pd.NaT)
    return pd.Series(parsed)

# ── Generischer Daten-Loader (erste 2 Spalten) ─────────────────────────────────
def load_generic_series(upl, col_name):
    if upl.name.lower().endswith('.csv'):
        df = pd.read_csv(upl, sep=';', decimal=',', header=0, usecols=[0,1])
    else:
        df = pd.read_excel(upl, header=0, usecols=[0,1], engine='openpyxl')
    df.columns = ['Zeitstempel', col_name]
    df['Zeitstempel'] = parse_flexible_timestamp(df['Zeitstempel'])
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
    return df

def load_price_df(upl): return load_generic_series(upl, 'Preis_€/MWh')
def load_pv_df(upl):    return load_generic_series(upl, 'PV_kWh')
def load_ev_df(upl):    return load_generic_series(upl, 'EV_kWh')

# ── Datenvalidierung ───────────────────────────────────────────────────────────
def validate_data(p_df, pv_df, ev_df):
    if p_df.empty or pv_df.empty or ev_df.empty:
        return False, 'Leere Datei'
    # Zeitstempel validieren
    for df, name in [(p_df,'Preis'), (pv_df,'PV'), (ev_df,'EV')]:
        if df['Zeitstempel'].isna().any():
            return False, f"{name} Zeitstempel fehlerhaft"
    # Negative Werte nur für PV und EV prüfen, Preise dürfen negativ sein
    for df, name in [(pv_df,'PV'), (ev_df,'EV')]:
        if (df.iloc[:,1] < 0).any():
            return False, f"Neg. Last in {name} Datei"
    return True, 'OK'

# ── Solver-Funktionen ─────────────────────────────────────────────────────────
def solve_battery(prices, pv, ev, cfg, grid_kw, interval_h, progress=None):
    T = len(prices)
    batt_max = cfg['bat_kw']*interval_h
    grid_max = grid_kw*interval_h
    cap, eff_sqrt, max_cyc = cfg['cap'], cfg['eff_pct']**0.5, cfg['max_cycles']

    m = pulp.LpProblem('BESS', pulp.LpMaximize)
    c = pulp.LpVariable.dicts('c', range(T), cat='Binary')
    d = pulp.LpVariable.dicts('d', range(T), cat='Binary')
    ch = pulp.LpVariable.dicts('ch', range(T), lowBound=0, upBound=batt_max)
    dh = pulp.LpVariable.dicts('dh', range(T), lowBound=0, upBound=batt_max)
    soc = pulp.LpVariable.dicts('soc', range(T), lowBound=0, upBound=cap)

    m += pulp.lpSum(prices[t]*(dh[t]-ch[t]) for t in range(T))
    for t in range(T):
        m += c[t]+d[t] <= 1
        m += ch[t] <= batt_max*c[t]
        m += dh[t] <= batt_max*d[t]
        m += pv[t] + ev[t] + ch[t] - dh[t] <= grid_max
        prev = cfg['start_soc'] if t==0 else soc[t-1]
        m += soc[t] == prev + eff_sqrt*ch[t] - dh[t]/eff_sqrt
        if progress and t%(max(1,T//50))==0:
            progress(5+int(45*t/T))
    m += pulp.lpSum((ch[t]+dh[t])/(2*cap) for t in range(T)) <= max_cyc
    if progress: progress(50)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=120)
    status = solver.solve(m)
    if progress: progress(90)
    if status!=pulp.LpStatusOptimal:
        st.warning(f"Solver Status: {pulp.LpStatus[status]}")
        return 0.0, np.zeros(T), np.zeros(T)
    obj = pulp.value(m.objective) or 0
    ch_v = np.array([ch[t].value() or 0 for t in range(T)])
    dh_v = np.array([dh[t].value() or 0 for t in range(T)])
    if progress: progress(100)
    return obj, ch_v, dh_v

def solve_joint(prices, pv, ev, cfgs, grid_kw, interval_h, progress=None):
    n,T = len(cfgs), len(prices)
    grid_max = grid_kw*interval_h
    effs = [c['eff_pct']**0.5 for c in cfgs]
    batt_maxs=[c['bat_kw']*interval_h for c in cfgs]
    caps=[c['cap'] for c in cfgs]
    starts=[c['start_soc'] for c in cfgs]
    max_cyc=[c['max_cycles'] for c in cfgs]

    m = pulp.LpProblem('BESS_Joint', pulp.LpMaximize)
    c_vars,d_vars,ch_vars,dh_vars,soc_vars = {},{},{},{},{}
    for i in range(n):
        c_vars[i] = pulp.LpVariable.dicts(f'c{i}',range(T),cat='Binary')
        d_vars[i] = pulp.LpVariable.dicts(f'd{i}',range(T),cat='Binary')
        ch_vars[i] = pulp.LpVariable.dicts(f'ch{i}',range(T),lowBound=0,upBound=batt_maxs[i])
        dh_vars[i] = pulp.LpVariable.dicts(f'dh{i}',range(T),lowBound=0,upBound=batt_maxs[i])
        soc_vars[i] = pulp.LpVariable.dicts(f'so{i}',range(T),lowBound=0,upBound=caps[i])
    m += pulp.lpSum(prices[t]*pulp.lpSum(dh_vars[i][t]-ch_vars[i][t] for i in range(n)) for t in range(T))
    for t in range(T):
        for i in range(n):
            m += c_vars[i][t]+d_vars[i][t] <= 1
            m += ch_vars[i][t] <= batt_maxs[i]*c_vars[i][t]
            m += dh_vars[i][t] <= batt_maxs[i]*d_vars[i][t]
            prev = starts[i] if t==0 else soc_vars[i][t-1]
            m += soc_vars[i][t] == prev + effs[i]*ch_vars[i][t] - dh_vars[i][t]/effs[i]
        total = pulp.lpSum(ch_vars[i][t]-dh_vars[i][t] for i in range(n))
        m += pv[t] + ev[t] + total <= grid_max
        if progress and t%(max(1,T//50))==0:
            progress(5+int(45*t/T))
    for i in range(n):
        m += pulp.lpSum((ch_vars[i][t]+dh_vars[i][t])/(2*caps[i]) for t in range(T)) <= max_cyc[i]
    if progress: progress(50)
    solver = pulp.PULP_CBC_CMD(msg=False,timeLimit=120)
    status = solver.solve(m)
    if progress: progress(90)
    if status!=pulp.LpStatusOptimal:
        st.warning(f"Joint Solver Status: {pulp.LpStatus[status]}")
        return 0.0, [np.zeros(T) for _ in range(n)], [np.zeros(T) for _ in range(n)]
    obj = pulp.value(m.objective) or 0
    chs = [np.array([ch_vars[i][t].value() or 0 for t in range(T)]) for i in range(n)]
    dhs = [np.array([dh_vars[i][t].value() or 0 for t in range(T)]) for i in range(n)]
    if progress: progress(100)
    return obj, chs, dhs

# ── Simulation & UI ───────────────────────────────────────────────────────────
st.set_page_config(layout='wide')
st.title('BESS: Single vs. Multi Use & Grid Constraint')
price_file = st.sidebar.file_uploader('Preise', type=['csv','xls','xlsx'])
pv_file    = st.sidebar.file_uploader('PV-Last', type=['csv','xls','xlsx'])
ev_file    = st.sidebar.file_uploader('EV-Last', type=['csv','xls','xlsx'])
enable2    = st.sidebar.checkbox('Zweite Batterie', True)
configs    = []
for i in (1,2):
    if i==1 or enable2:
        st.sidebar.markdown(f'**Batterie {i}**')
        mode = st.sidebar.selectbox(f'Modus{i}', ['Single Use','Multi Use'], key=f'm{i}')
        start= st.sidebar.number_input(f'StartSoC{i} (kWh)', 0.0,1e6,0.0, key=f's{i}')
        cap  = st.sidebar.number_input(f'Kap{i} (kWh)', 0.1,1e6,4472.0,key=f'c{i}')
        bkw  = st.sidebar.number_input(f'Leistung{i} (kW)', 0.1,1e6,559.0,key=f'p{i}')
        eff  = st.sidebar.number_input(f'Eff{i} (%)', 1.0,100.0,91.0, key=f'e{i}')/100.0
        cyc  = st.sidebar.number_input(f'Zyklen{i}', 0.0,1e4,548.0,key=f'y{i}')
        configs.append({'mode':mode,'start_soc':start,'cap':cap,'bat_kw':bkw,'eff_pct':eff,'max_cycles':cyc})
grid_kw = st.sidebar.number_input('Netzanschluss (kW)',0.1,1e6,37000.0)
if st.sidebar.button('▶️ Simulation starten'):
    if not(price_file and pv_file and ev_file):
        st.sidebar.error('Bitte alle Dateien hochladen.')
    else:
        # Laden & Validieren
        p_df = load_price_df(price_file)
        pv_df= load_pv_df(pv_file)
        ev_df= load_ev_df(ev_file)
        valid,msg = validate_data(p_df,pv_df,ev_df)
        if not valid:
            st.error(msg); st.stop()
        # Arrays
        ts     = p_df['Zeitstempel']
        prices = p_df['Preis_€/MWh'].to_numpy()/1000.0
        pv     = pv_df['PV_kWh'].to_numpy()
        ev     = ev_df['EV_kWh'].to_numpy()
        interval_h = ts.diff().dropna().mode()[0].total_seconds()/3600.0
        st.info(f'Intervall: {interval_h} h')
        # Solver
        free = [solve_battery(prices,pv,ev,c,grid_kw,interval_h,set_progress) for c in configs]
        obj_joint, chs_joint, dhs_joint = solve_joint(prices,pv,ev,configs,grid_kw,interval_h,set_progress)
        # Einzeloptimierung anzeigen
        st.subheader('Einzeloptimierung')
        tot_free = 0
        for idx,(cfg,(obj,_,_)) in enumerate(zip(configs,free), start=1):
            st.metric(f"B{idx} ({cfg['mode']})", fmt_euro(obj))
            tot_free += obj
        st.metric('Gesamt Free', fmt_euro(tot_free))
        # Gemeinsame Optimierung
        st.subheader('Gemeinsam')
        st.metric('Gesamt', fmt_euro(obj_joint))
        for idx,cfg in enumerate(configs, start=1):
            ind = float(np.dot(prices, dhs_joint[idx-1] - chs_joint[idx-1]))
            st.metric(f"B{idx} Anteil", fmt_euro(ind))
        # Vergleich
        imp = obj_joint - tot_free
        pct = (imp / abs(tot_free) * 100) if tot_free!=0 else 0
        st.subheader('Vergleich')
        c1,c2,c3 = st.columns(3)
        c1.metric('Δ absolut', fmt_euro(imp))
        c2.metric('Δ %', f"{pct:.2f}%")
        net_load = pv + ev + sum(chs_joint) - sum(dhs_joint)
        net_util = np.mean(net_load/(grid_kw*interval_h))*100
        c3.metric('Netzauslastung', f"{net_util:.1f}%")
        # Ergebnisse als Tabelle & Download
        st.subheader('Ergebnisse')
        out = pd.DataFrame({'Zeit':ts, 'Preis_€/kWh':prices, 'PV_kWh':pv, 'EV_kWh':ev})
        for idx in range(len(configs)):
            out[f'B{idx+1}_Laden_kWh'] = chs_joint[idx]
            out[f'B{idx+1}_Entladen_kWh'] = dhs_joint[idx]
        out['Netzlast_kWh'] = net_load
        out['Netzlast_%'] = net_load/(grid_kw*interval_h)*100
        viol = (net_load > grid_kw*interval_h).sum()
        if viol>0:
            st.warning(f"⚠️ {viol} Überlastungen")
        st.dataframe(out)
        buf = BytesIO()
        out.to_excel(buf, index=False, engine='openpyxl')
        buf.seek(0)
        st.download_button('📥 Excel-Export', buf, file_name=f'res_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
