# BESS – Eigenverbrauchsmaximierung (ohne Day-Ahead)
# -------------------------------------------------
# Streamlit-App zur reinen Eigenverbrauchsoptimierung (EV-first)
# - Keine Preis- oder Arbitrage-Logik
# - Priorität: PV -> EV, dann Batterie (Laden/Entladen), Rest optional Export/Spill
# - Robuste Zeitreihen-Verarbeitung (15-min / 60-min etc.)
# - KPI-Ausgabe + Zeitreihen + Excel-Export (inkl. SoC)
#
# Hinweise zu Einheiten:
# - PV, EV (Last) in kWh pro Zeitschritt
# - Batterie: E_max in kWh, P_max in kW
# - Zeitschritt Δt wird aus Zeitstempel abgeleitet (Median der Differenzen)
# - RTE wird in η_c = η_d = sqrt(RTE) aufgeteilt
#
# Benötigte Dateien (jeweils .xlsx oder .csv):
# - PV-Zeitreihe mit Spalten [timestamp, value]
# - EV-Zeitreihe (Eigenverbrauchslastgang) mit Spalten [timestamp, value]

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import math

# ────────────────────────────────────────────────────────────────────────────────
# UI Setup
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="BESS – EV-Only", page_icon="🔋", layout="wide")

# Session-State für Fortschritt
if "progress_bar" not in st.session_state:
    st.session_state.progress_bar = st.progress(0)
if "progress_text" not in st.session_state:
    st.session_state.progress_text = st.empty()

def set_progress(pct: int):
    st.session_state.progress_bar.progress(min(max(int(pct), 0), 100))
    st.session_state.progress_text.markdown(f"**Fortschritt:** {min(max(int(pct), 0), 100)}%")

# ────────────────────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────────────────────

def read_timeseries(uploaded_file) -> pd.DataFrame:
    """Liest .xlsx oder .csv mit Spalten [timestamp, value].
    - Erkennt Zeitstempel-Spalte (erste Datum/Zeit-ähnliche Spalte) automatisch
    - Erkennt Werte-Spalte (erste numerische Spalte) automatisch
    - Normalisiert Spaltennamen auf ['ts', 'val'] und sortiert
    """
    if uploaded_file is None:
        return pd.DataFrame(columns=["ts", "val"])  # leer

    name = uploaded_file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")

    # Spalten auto-erkennen
    ts_col = None
    val_col = None
    for col in df.columns:
        if ts_col is None:
            # Versuche, als Datum zu parsen
            try:
                pd.to_datetime(df[col])
                ts_col = col
            except Exception:
                pass
        if val_col is None and np.issubdtype(df[col].dtype, np.number):
            val_col = col

    if ts_col is None:
        # Fallback: erste Spalte parsen
        ts_col = df.columns[0]
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    else:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    if val_col is None:
        # Fallback: zweite Spalte
        candidates = [c for c in df.columns if c != ts_col]
        if not candidates:
            raise ValueError("Keine Werte-Spalte gefunden.")
        val_col = candidates[0]

    df = df[[ts_col, val_col]].rename(columns={ts_col: "ts", val_col: "val"})
    df = df.dropna(subset=["ts"]).copy()
    # numerisch erzwingen, negative Rauschen cleanen
    df["val"] = pd.to_numeric(df["val"], errors="coerce").fillna(0.0)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def align_series(pv: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Inner Join auf Zeitstempel, NaNs -> 0. Rückgabe: DataFrame mit Spalten pv, ev."""
    if pv.empty or ev.empty:
        return pd.DataFrame(columns=["ts", "pv", "ev"])  # leer
    df = pd.merge(pv, ev, on="ts", how="inner", suffixes=("_pv", "_ev"))
    df = df.rename(columns={"val_pv": "pv", "val_ev": "ev"})
    df = df[["ts", "pv", "ev"]].copy()
    df["pv"] = df["pv"].clip(lower=0)
    df["ev"] = df["ev"].clip(lower=0)
    return df


def infer_dt_hours(ts: pd.Series) -> float:
    """Δt in Stunden basierend auf Median der Zeitdifferenzen (robust ggü. DST)."""
    if len(ts) < 2:
        return 1.0  # Default 1h
    d = pd.Series(ts).diff().dropna().dt.total_seconds() / 3600.0
    med = float(d.median())
    # Num. Stabilisierung (z. B. 0.249999 -> 0.25)
    for target in [1/96, 0.25, 0.5, 1.0]:
        if abs(med - target) < 1e-3:
            return target
    return med


def simulate_ev_only(df: pd.DataFrame, E_kWh: float, P_kW: float, rte: float,
                      soc0_perc: float, grid_imp_kW: float, grid_exp_kW: float,
                      ts_is_period_end: bool = True) -> pd.DataFrame:
    """Vorwärts-Simulation EV-first ohne Preise.

    Args:
        df: DataFrame mit Spalten [ts, pv, ev] (Energien je Intervall, kWh)
        E_kWh: Batteriekapazität [kWh]
        P_kW: max. Lade- und Entladeleistung [kW] (symmetrisch)
        rte: Round-Trip-Effizienz in Dezimal (z. B. 0.92)
        soc0_perc: Anfangs-SoC in % [0..100]
        grid_imp_kW: Importlimit [kW]
        grid_exp_kW: Exportlimit [kW]
        ts_is_period_end: True, wenn Zeitstempel das Intervallende markieren

    Returns:
        DataFrame mit Flüssen und SoC.
    """
    df = df.copy()
    if df.empty:
        return df

    dt_h = infer_dt_hours(df["ts"])  # Stunden
    eta_c = eta_d = math.sqrt(max(min(rte, 0.999999), 0.0))  # Sicherheitskappung

    # Ergebnis-Arrays
    n = len(df)
    pv_to_ev = np.zeros(n)
    pv_to_ch = np.zeros(n)
    pv_to_exp = np.zeros(n)
    pv_spill  = np.zeros(n)
    dis_to_ev = np.zeros(n)
    imp_to_ev = np.zeros(n)
    soc_arr   = np.zeros(n+1)

    soc = E_kWh * soc0_perc / 100.0
    soc_arr[0] = soc

    imp_limit_e = grid_imp_kW * dt_h
    exp_limit_e = grid_exp_kW * dt_h
    ch_limit_e  = P_kW * dt_h
    dis_limit_e = P_kW * dt_h

    for i, row in enumerate(df.itertuples(index=False)):
        pv = float(row.pv)
        ev = float(row.ev)

        # 1) PV -> EV
        pv_ev = min(pv, ev)
        pv -= pv_ev
        ev -= pv_ev

        # 2) Batterie -> EV (so viel wie möglich decken)
        # Max. AC-Energie aus Batterie = soc * eta_d
        batt_to_ev_max = min(dis_limit_e, soc * eta_d)
        dis_ev = min(ev, batt_to_ev_max)
        ev -= dis_ev
        soc -= dis_ev / eta_d  # SoC-Abnahme (AC -> DC)

        # 3) Rest-Last per Netzimport (begrenzt)
        imp_ev = min(ev, imp_limit_e)
        ev -= imp_ev
        # ev ist jetzt unversorgter Rest (shed); wir erfassen ihn nicht in KPIs,
        # zeigen aber einen Warnhinweis am Ende, falls > 0 auftritt.

        # 4) PV-Überschuss -> Batterie (Laden)
        # Max. AC-Energie, die wir laden können, begrenzt durch Platz (DC/η_c)
        room_ac = max((E_kWh - soc) / eta_c, 0.0)
        ch_e = min(pv, ch_limit_e, room_ac)
        pv -= ch_e
        soc += ch_e * eta_c  # SoC-Zunahme (AC -> DC)

        # 5) PV-Rest -> Export (begrenzt) und ggf. Spill
        exp_e = min(pv, exp_limit_e)
        pv -= exp_e
        spill = pv  # verbleibender Rest (Curtailment)

        # Ergebnisse schreiben
        pv_to_ev[i] = pv_ev
        dis_to_ev[i] = dis_ev
        imp_to_ev[i] = imp_ev
        pv_to_ch[i] = ch_e
        pv_to_exp[i] = exp_e
        pv_spill[i]  = spill
        soc_arr[i+1] = soc

    out = df.copy()
    out["pv_to_ev"] = pv_to_ev
    out["dis_to_ev"] = dis_to_ev
    out["imp_to_ev"] = imp_to_ev
    out["pv_to_ch"] = pv_to_ch
    out["pv_to_exp"] = pv_to_exp
    out["pv_spill"]  = pv_spill
    out["soc_kwh"]   = soc_arr[1:]

    # KPIs
    out["self_consumed_pv"] = out["pv_to_ev"] + out["pv_to_ch"]  # PV, die vor Ort genutzt wird
    out["onsite_supply"] = out["pv_to_ev"] + out["dis_to_ev"]     # EV, die vor Ort gedeckt wurde

    return out


def kpi_box(out: pd.DataFrame):
    if out.empty:
        return
    total_pv = out["pv"].sum()
    total_ev = out["ev"].sum()
    total_import = out["imp_to_ev"].sum()
    total_export = out["pv_to_exp"].sum()
    total_spill  = out["pv_spill"].sum()
    total_batt_throughput = out["pv_to_ch"].sum() + out["dis_to_ev"].sum()

    # Kennzahlen
    self_consumed = out["self_consumed_pv"].sum()
    onsite = out["onsite_supply"].sum()

    scr = (self_consumed / total_pv * 100) if total_pv > 0 else 0.0  # Eigenverbrauchsquote PV
    aut = (onsite / total_ev * 100) if total_ev > 0 else 0.0          # Autarkiegrad
    cycles = (out["pv_to_ch"].sum() * 1.0 + out["dis_to_ev"].sum() * 1.0) / (2 * total_pv + 1e-9)
    # Alternativer Zyklen-Schätzer (konservativ): Durchsatz / (2 * E_kWh)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("PV gesamt [kWh]", f"{total_pv:,.0f}".replace(",", "."))
    c2.metric("Last gesamt [kWh]", f"{total_ev:,.0f}".replace(",", "."))
    c3.metric("Eigenverbrauchsquote PV", f"{scr:.1f} %")
    c4.metric("Autarkiegrad", f"{aut:.1f} %")
    c5.metric("Import / Export [kWh]", f"{total_import:,.0f} / {total_export:,.0f}".replace(",", "."))

    if total_spill > 1e-6:
        st.info(f"PV-Abregelung (Spill): {total_spill:,.0f} kWh".replace(",", "."))


# ────────────────────────────────────────────────────────────────────────────────
# Sidebar – Konfiguration
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Konfiguration")

    colA, colB = st.columns(2)
    E_kWh = colA.number_input("Batteriekapazität E_max [kWh]", 0.1, 10_000.0, 20.0, step=1.0)
    P_kW = colB.number_input("Batterieleistung P_max [kW]", 0.1, 10_000.0, 10.0, step=0.5)

    colC, colD = st.columns(2)
    rte_pct = colC.slider("Round-Trip-Effizienz RTE [%]", 60, 100, 92)
    soc0 = colD.slider("Start-SoC [%]", 0, 100, 50)

    colE, colF = st.columns(2)
    grid_imp = colE.number_input("Netz-Importlimit [kW]", 0.0, 100_000.0, 1_000.0, step=10.0)
    grid_exp = colF.number_input("Netz-Exportlimit [kW]", 0.0, 100_000.0, 1_000.0, step=10.0)

    ts_end = st.checkbox("Zeitstempel sind Periodenende (empfohlen)", value=True)

# ────────────────────────────────────────────────────────────────────────────────
# Main – Upload & Simulation
# ────────────────────────────────────────────────────────────────────────────────
st.title("BESS – Single Battery · Eigenverbrauchsmaximierung")

st.subheader("Upload")
col1, col2 = st.columns(2)
with col1:
    pv_file = st.file_uploader("PV-Zeitreihe (xlsx/csv)", type=["xlsx", "xls", "csv"], key="pv")
with col2:
    ev_file = st.file_uploader("EV-/Last-Zeitreihe (xlsx/csv)", type=["xlsx", "xls", "csv"], key="ev")

pv_df = read_timeseries(pv_file)
ev_df = read_timeseries(ev_file)

if not pv_df.empty and not ev_df.empty:
    set_progress(10)
    data = align_series(pv_df, ev_df)
    if data.empty:
        st.error("Keine überlappenden Zeitstempel zwischen PV und Last gefunden.")
    else:
        st.success(f"Daten geladen: {len(data):,} Schritte".replace(",", "."))
        dt_h = infer_dt_hours(data["ts"]) 
        st.caption(f"Erkannter Zeitschritt Δt = {dt_h:.4f} h")

        st.subheader("Simulation")
        if st.button("Eigenverbrauch optimieren", type="primary"):
            set_progress(40)
            out = simulate_ev_only(
                data, E_kWh=E_kWh, P_kW=P_kW, rte=rte_pct/100.0,
                soc0_perc=soc0, grid_imp_kW=grid_imp, grid_exp_kW=grid_exp,
                ts_is_period_end=ts_end,
            )
            set_progress(70)

            # KPIs
            kpi_box(out)

            # Charts
            st.subheader("Zeitreihen")
            show_cols = [
                "pv", "ev", "pv_to_ev", "dis_to_ev", "imp_to_ev", "pv_to_ch", "pv_to_exp", "pv_spill"
            ]
            st.line_chart(out.set_index("ts")[show_cols])
            st.area_chart(out.set_index("ts")["soc_kwh"], height=140, use_container_width=True)

            # Excel-Export
            st.subheader("Export")
            with BytesIO() as buffer:
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    out.to_excel(writer, index=False, sheet_name="EV_only")
                st.download_button(
                    label="Excel herunterladen (inkl. SoC)",
                    data=buffer.getvalue(),
                    file_name="ev_only_self_consumption.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            # Warnung bei unversorgter Last (shed)
            # Hier wird indirekt geprüft: wenn ev nach Schritt 3 > 0 blieb, hätten wir shed.
            # Wir rekonstruieren es:
            shed = (out["ev"] - out["pv_to_ev"] - out["dis_to_ev"] - out["imp_to_ev"]).clip(lower=0).sum()
            if shed > 1e-6:
                st.warning(
                    "Es gab unversorgte Last (Netzlimit zu niedrig oder Batterie zu klein).\n"
                    "Erhöhe ggf. Importlimit oder Kapazitäten."
                )

            set_progress(100)
else:
    st.info("Bitte PV- und Last-Datei hochladen.")
    set_progress(0)
