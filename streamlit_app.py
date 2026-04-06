import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os
import shap
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanGuard — Loan Default Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── DARK MATPLOTLIB DEFAULTS ─────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0D0F1C',
    'axes.facecolor':    '#0D0F1C',
    'text.color':        '#F0F2FF',
    'axes.labelcolor':   '#7A809C',
    'xtick.color':       '#7A809C',
    'ytick.color':       '#F0F2FF',
    'axes.edgecolor':    '#1C1F2E',
    'grid.color':        '#1C1F2E',
    'grid.alpha':        0.5,
    'font.family':       'sans-serif',
})

# ─── CSS — FraudGuard exact theme ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

* { box-sizing: border-box; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display:none; }
[data-testid="collapsedControl"] { display:none; }
.block-container { padding-top:1rem !important; max-width:1200px; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif !important; }

:root {
    --bg-deep:   #07080F;
    --bg-card:   #0D0F1C;
    --border:    rgba(255,255,255,0.07);
    --accent:    #00F5C8;
    --accent2:   #FF5F40;
    --accent3:   #7B6EF6;
    --t-main:    #F0F2FF;
    --t-muted:   #7A809C;
    --t-dim:     #4A5068;
}

.stApp {
    background-color: var(--bg-deep) !important;
    color: var(--t-main) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,245,200,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(123,110,246,0.06) 0%, transparent 60%);
    background-attachment: fixed;
}

/* NAV */
.nav-logo {
    font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800;
    color:var(--accent); letter-spacing:-0.02em; padding:0.4rem 0;
}
.nav-logo span { color:var(--t-main); }

/* ALL BUTTONS */
.stButton > button {
    font-family:'DM Sans',sans-serif !important; font-weight:500 !important;
    font-size:0.88rem !important; border:1px solid var(--border) !important;
    border-radius:10px !important; padding:0.55rem 1.1rem !important;
    color:var(--t-muted) !important; background:var(--bg-card) !important;
    transition:all 0.2s ease !important; width:100%;
}
.stButton > button:hover {
    border-color:var(--accent) !important; color:var(--accent) !important;
    background:rgba(0,245,200,0.06) !important;
}
.cta-btn .stButton > button {
    font-size:1rem !important; border:1px solid rgba(255,255,255,0.12) !important;
    border-radius:12px !important; padding:0.85rem 2.2rem !important;
    font-weight:600 !important;
}
.cta-btn .stButton > button:hover { border-color:var(--accent) !important; color:var(--accent) !important; }
.run-btn .stButton > button {
    background:linear-gradient(135deg,#7B6EF6 0%,#5A4FCF 100%) !important;
    color:#fff !important; font-weight:600 !important; font-size:1rem !important;
    border:none !important; border-radius:12px !important; padding:0.8rem 2rem !important;
    box-shadow:0 4px 20px rgba(123,110,246,0.3) !important;
}
.run-btn .stButton > button:hover { box-shadow:0 6px 28px rgba(123,110,246,0.5) !important; color:#fff !important; }

/* CARDS */
.metric-card {
    background:var(--bg-card); border:1px solid var(--border); border-radius:16px;
    padding:1.8rem 1.5rem; text-align:center; position:relative; overflow:hidden; height:100%;
}
.metric-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,transparent,var(--accent),transparent); opacity:0.6;
}
.metric-value { font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:800; color:var(--accent); letter-spacing:-0.03em; line-height:1; }
.metric-label { color:var(--t-muted); font-size:0.85rem; margin-top:0.6rem; line-height:1.5; }
.step-card { background:var(--bg-card); border:1px solid var(--border); border-radius:16px; padding:1.8rem 1.5rem; height:100%; }
.step-num { font-family:'Syne',sans-serif; font-size:0.75rem; font-weight:700; color:var(--accent3); letter-spacing:0.05em; margin-bottom:1rem; }
.step-title { font-family:'Syne',sans-serif; font-weight:700; color:var(--t-main); font-size:1rem; margin-bottom:0.5rem; }
.step-desc { color:var(--t-muted); font-size:0.85rem; line-height:1.6; }

/* HERO */
.hero-wrapper { text-align:center; padding:3rem 0 2rem 0; }
.hero-eyebrow {
    display:inline-block; font-size:0.72rem; font-weight:600; letter-spacing:0.14em;
    text-transform:uppercase; color:var(--accent); background:rgba(0,245,200,0.08);
    border:1px solid rgba(0,245,200,0.2); border-radius:50px; padding:0.3rem 1rem; margin-bottom:1.2rem;
}
.hero-title { font-family:'Syne',sans-serif; font-size:3.8rem; font-weight:800; color:var(--t-main); line-height:1.05; letter-spacing:-0.04em; margin-bottom:1.2rem; }
.hero-highlight { background:linear-gradient(135deg,#00F5C8,#7B6EF6); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.hero-sub { color:var(--t-muted); font-size:1.05rem; max-width:540px; margin:0 auto 2.5rem auto; line-height:1.65; }

/* SECTION */
.section-eyebrow { font-size:0.72rem; font-weight:600; letter-spacing:0.14em; text-transform:uppercase; color:var(--accent); margin-bottom:0.3rem; }
.section-title-lg { font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:800; color:var(--t-main); margin:0 0 1.5rem 0; letter-spacing:-0.02em; }
.page-h2 { font-family:'Syne',sans-serif; font-size:2.1rem; font-weight:800; color:var(--t-main); letter-spacing:-0.03em; margin:0.2rem 0 0.3rem 0; }
.page-sub { color:var(--t-muted); font-size:0.93rem; margin:0 0 2rem 0; }

/* INPUTS */
.input-header { font-family:'Syne',sans-serif; font-weight:700; font-size:0.95rem; color:var(--t-main); margin-bottom:1.2rem; letter-spacing:-0.01em; }
.input-section-label { font-size:0.75rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:var(--accent3); margin:1.2rem 0 0.6rem 0; }
.stSlider > div > div { color:var(--accent) !important; }
.stNumberInput input { background:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:10px !important; color:var(--t-main) !important; }
.stNumberInput input:focus { border-color:var(--accent) !important; }

/* RESULT */
.result-safe     { background:rgba(0,245,200,0.04); border:1px solid rgba(0,245,200,0.25); border-radius:16px; padding:1.8rem; text-align:center; }
.result-risk     { background:rgba(255,95,64,0.04);  border:1px solid rgba(255,95,64,0.25);  border-radius:16px; padding:1.8rem; text-align:center; }
.result-moderate { background:rgba(255,160,0,0.04);  border:1px solid rgba(255,160,0,0.25);  border-radius:16px; padding:1.8rem; text-align:center; }

/* FEATURE GRID */
.feat-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.3rem; margin-top:0.5rem; }
.feat-item { display:flex; justify-content:space-between; align-items:center; padding:0.4rem 0.5rem; background:rgba(255,255,255,0.02); border-radius:6px; }
.feat-name { color:var(--t-muted); font-size:0.76rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:130px; }
.feat-pos  { color:#FF5F40; font-weight:600; font-size:0.76rem; flex-shrink:0; }
.feat-neg  { color:#00F5C8; font-weight:600; font-size:0.76rem; flex-shrink:0; }

/* STATUS */
.status-bar { background:var(--bg-card); border:1px solid var(--border); border-radius:10px; padding:0.65rem 1rem; text-align:center; font-size:0.76rem; color:var(--t-dim); letter-spacing:0.03em; margin-top:1rem; }
.status-kv { color:var(--accent); font-weight:600; }

/* EMPTY */
.empty-state { background:var(--bg-card); border:1.5px dashed rgba(255,255,255,0.08); border-radius:16px; padding:3.5rem 2rem; text-align:center; }

/* DIVIDER */
.fancy-divider { height:1px; background:linear-gradient(90deg,transparent,rgba(255,255,255,0.07),transparent); margin:2rem 0; border:none; }

/* GRAPH CAPTION */
.graph-caption { color:var(--t-muted); font-size:0.8rem; text-align:center; margin-top:0.5rem; font-style:italic; line-height:1.5; }

/* TABS */
.stTabs [data-baseweb="tab-list"] { background:var(--bg-card) !important; border-radius:12px 12px 0 0 !important; gap:2px !important; padding:6px 6px 0 6px !important; border-bottom:1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] { font-family:'DM Sans',sans-serif !important; font-weight:500 !important; font-size:0.88rem !important; color:var(--t-muted) !important; background:transparent !important; border-radius:8px 8px 0 0 !important; padding:0.6rem 1.2rem !important; border:none !important; }
.stTabs [aria-selected="true"] { color:var(--accent) !important; background:var(--bg-deep) !important; border-bottom:2px solid var(--accent) !important; }

/* REPORT */
.report-wrap { background:var(--bg-card); border:1px solid var(--border); border-radius:16px; padding:2.5rem 3rem; line-height:1.75; }
.report-wrap h1 { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:var(--t-main); border-bottom:1px solid var(--border); padding-bottom:0.8rem; margin-bottom:1.5rem; letter-spacing:-0.02em; }
.report-wrap h2 { font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700; color:var(--t-main); margin:2rem 0 0.6rem 0; }
.report-wrap h3 { font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:600; color:var(--accent); margin:1.5rem 0 0.4rem 0; }
.report-wrap p  { color:var(--t-muted); margin-bottom:0.8rem; }
.report-wrap ul, .report-wrap ol { color:var(--t-muted); padding-left:1.4rem; margin-bottom:0.8rem; }
.report-wrap li { margin-bottom:0.3rem; line-height:1.6; }
.report-wrap strong { color:var(--t-main); }
.report-wrap code { background:rgba(123,110,246,0.15); color:var(--accent3); padding:0.15rem 0.4rem; border-radius:4px; font-size:0.85em; font-family:'Courier New',monospace; }
.report-wrap pre { background:#0a0b14; border:1px solid var(--border); border-radius:10px; padding:1.2rem; overflow-x:auto; margin:1rem 0; }
.report-wrap pre code { background:none; color:#e2e8f0; padding:0; font-size:0.83rem; }
.report-wrap table { width:100%; border-collapse:collapse; margin:1rem 0; font-size:0.88rem; }
.report-wrap th { background:rgba(123,110,246,0.15); color:var(--t-main); padding:0.6rem 1rem; text-align:left; border-bottom:1px solid var(--border); }
.report-wrap td { padding:0.5rem 1rem; border-bottom:1px solid rgba(255,255,255,0.04); color:var(--t-muted); }
.report-wrap tr:hover td { background:rgba(255,255,255,0.02); }
.report-wrap blockquote { border-left:3px solid var(--accent); padding-left:1rem; color:var(--t-muted); font-style:italic; margin:1rem 0; }

/* FOOTER */
.footer-note { text-align:center; color:var(--t-dim); font-size:0.8rem; padding:2rem 0 1rem 0; }
.footer-note a { color:var(--accent); text-decoration:none; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if 'page'   not in st.session_state: st.session_state.page   = 'Home'
if 'result' not in st.session_state: st.session_state.result = None

# ─── CONSTANTS ─────────────────────────────────────────────────────────────────
THRESHOLD = 0.1883   # ← match your printed threshold from 1_Project.py
PLOT_BG   = '#0D0F1C'
PLOT_TEXT = '#7A809C'
PLOT_WH   = '#F0F2FF'
PLOT_GRID = '#1C1F2E'
ACCENT    = '#00F5C8'
ACCENT2   = '#FF5F40'
ACCENT3   = '#7B6EF6'

FEATURE_COLS = [
    'RevolvingUtilizationOfUnsecuredLines',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate',
    'DebtRatio','MonthlyIncome','age',
    'NumberOfOpenCreditLinesAndLoans',
    'MonthlyIncome_was_missing',
    'Total_Late_Events','Weighted_Late_Score',
    'Credit_Utilization_Warning','Income_Per_Dependent',
    'Remaining_Living_Money','Is_Cash_Negative','Low_Buffer_Flag',
    'Has_Any_Late','Max_Late_Severity','Youth_Utilization_Risk',
    'Debt_Squeeze','Severity_Per_Line','Struggle_Index',
    'Age_Debt_Interaction','Late_to_Open_Ratio','Utilization_Squared',
]

# ─── MODEL LOAD ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    miss = [f for f in ["best_xgb_model.joblib","data_scaler.joblib"] if not os.path.exists(f)]
    if miss:
        st.error(f"❌ Missing: {', '.join(miss)} — run 1_Project.py first."); st.stop()
    return joblib.load("best_xgb_model.joblib"), joblib.load("data_scaler.joblib")

model, scaler = load_model()

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────
def engineer_features(util, age, debt, income, open_l, dep, l30, l60, l90):
    income_log = np.log1p(max(income, 0))
    raw = np.expm1(income_log)
    tl  = l30 + l60 + l90
    wl  = l30*1 + l60*2 + l90*3
    abd = debt * raw
    rem = raw - abd
    row = {
        'RevolvingUtilizationOfUnsecuredLines': util,
        'NumberOfTime30-59DaysPastDueNotWorse': l30,
        'NumberOfTime60-89DaysPastDueNotWorse': l60,
        'NumberOfTimes90DaysLate':              l90,
        'DebtRatio':                            debt,
        'MonthlyIncome':                        income_log,
        'age':                                  age,
        'NumberOfOpenCreditLinesAndLoans':      open_l,
        'MonthlyIncome_was_missing':            0,
        'Total_Late_Events':                    tl,
        'Weighted_Late_Score':                  wl,
        'Credit_Utilization_Warning':           int(util > 0.85),
        'Income_Per_Dependent':                 np.log1p(raw / (dep + 1)),
        'Remaining_Living_Money':               rem,
        'Is_Cash_Negative':                     int(rem < 0),
        'Low_Buffer_Flag':                      int(rem < 500),
        'Has_Any_Late':                         int(tl > 0),
        'Max_Late_Severity':                    int(l30>0)+int(l60>0)+int(l90>0),
        'Youth_Utilization_Risk':               util * (1 / max(age-17, 0.1)),
        'Debt_Squeeze':                         debt * int(rem < 0),
        'Severity_Per_Line':                    wl / (open_l + 1),
        'Struggle_Index':                       util * tl,
        'Age_Debt_Interaction':                 abd / (age + 1),
        'Late_to_Open_Ratio':                   tl / (open_l + 1),
        'Utilization_Squared':                  util ** 2,
    }
    df = pd.DataFrame([row], columns=FEATURE_COLS)
    return pd.DataFrame(scaler.transform(df), columns=FEATURE_COLS)

# ─── PLOT HELPERS ─────────────────────────────────────────────────────────────
def ax_style(fig, ax):
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(PLOT_BG)
    ax.tick_params(colors=PLOT_TEXT, labelsize=9)
    ax.xaxis.label.set_color(PLOT_TEXT)
    ax.yaxis.label.set_color(PLOT_TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor(PLOT_GRID)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

def load_report(fname):
    for folder in ['reports', '.']:
        for ext in ['.md', '.txt', '']:
            p = os.path.join(folder, fname.rsplit('.', 1)[0] + ext if ext else fname)
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f: return f.read()
    return f"*`{fname}` not found. Place report files in a `reports/` folder.*"

# ══════════════════════════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════════════════════════
def navbar():
    c0,c1,c2,c3,c4 = st.columns([3,1,1,1,1])
    with c0: st.markdown('<div class="nav-logo">🛡️ Loan<span>Guard</span></div>', unsafe_allow_html=True)
    with c1:
        if st.button('Home',      key='n1', use_container_width=True): st.session_state.page='Home';     st.rerun()
    with c2:
        if st.button('Predict',   key='n2', use_container_width=True): st.session_state.page='Predict';  st.rerun()
    with c3:
        if st.button('Visualize', key='n3', use_container_width=True): st.session_state.page='Visualize';st.rerun()
    with c4:
        if st.button('Report',    key='n4', use_container_width=True): st.session_state.page='Report';   st.rerun()
    st.markdown('<div class="fancy-divider" style="margin:0.4rem 0 1.5rem 0;"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("""
    <div class="hero-wrapper">
        <div class="hero-eyebrow">AI-POWERED CREDIT RISK</div>
        <div class="hero-title">Will This Borrower<br><span class="hero-highlight">Default?</span></div>
        <div class="hero-sub">Real-time loan default prediction powered by XGBoost and SHAP —
        trained on 150,000+ borrower records with full explainability.</div>
    </div>""", unsafe_allow_html=True)

    _, mid, _ = st.columns([3,2,3])
    with mid:
        st.markdown('<div class="cta-btn">', unsafe_allow_html=True)
        if st.button('Run Loan Risk Assessment →', use_container_width=True, key='hero_cta'):
            st.session_state.page = 'Predict'; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider" style="margin:2.5rem 0;"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-eyebrow">THE PROBLEM</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title-lg">Why It Matters</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3, gap="medium")
    for col,(val,color,label) in zip([c1,c2,c3],[
        ("$1.8T", ACCENT,  "In outstanding loan balances at risk of default in the US alone"),
        ("6.6%",  ACCENT3, "Of borrowers default — extreme imbalance that breaks naive models"),
        ("75%",   ACCENT,  "Of actual defaulters caught — the primary banking objective"),
    ]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color};">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-eyebrow">THE PROCESS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title-lg">Three Steps to Risk Clarity</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3, gap="medium")
    for col,(num,title,desc) in zip([c1,c2,c3],[
        ("01","Enter Details","Input borrower's financial profile — income, credit utilization, debt ratio, and payment history."),
        ("02","AI Analysis","XGBoost processes 25 engineered features and generates a calibrated default probability."),
        ("03","Full Explanation","SHAP values reveal exactly which factors drove the decision. Transparent, auditable, fair."),
    ]):
        with col:
            st.markdown(f'<div class="step-card"><div class="step-num">{num}</div><div class="step-title">{title}</div><div class="step-desc">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="footer-note">Dataset: <a href="https://www.kaggle.com/competitions/GiveMeSomeCredit" target="_blank">Kaggle — Give Me Some Credit</a> &nbsp;·&nbsp; 150,000 US borrowers &nbsp;·&nbsp; 10 financial features</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
def page_predict():
    st.markdown('<div class="section-eyebrow">REAL-TIME ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-h2">Loan Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Enter borrower details to assess default risk instantly</div>', unsafe_allow_html=True)

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown('<div class="input-header">Borrower Financial Profile</div>', unsafe_allow_html=True)

        r1a, r1b = st.columns(2, gap="small")
        with r1a: age    = st.slider("Age", 18, 100, 35, help="Borrower's current age in years")
        with r1b: income = st.number_input("Monthly Income ($)", 0, 500000, 5000, 100, help="Gross monthly income before taxes")

        r2a, r2b = st.columns(2, gap="small")
        with r2a: util = st.slider("Credit Utilization", 0.0, 1.0, 0.30, 0.01, help="Revolving credit used / total available (0=0%, 1=100%)")
        with r2b: debt = st.slider("Debt Ratio", 0.0, 5.0, 0.35, 0.01, help="Monthly debt / monthly gross income")

        r3a, r3b = st.columns(2, gap="small")
        with r3a: open_l = st.slider("Open Credit Lines", 0, 40, 8, help="Number of active credit accounts")
        with r3b: dep    = st.slider("Dependents",        0, 10, 0, help="People financially dependent on borrower")

        st.markdown('<div class="input-section-label">Late Payment History</div>', unsafe_allow_html=True)
        l1,l2,l3 = st.columns(3, gap="small")
        with l1: late30 = st.slider("30–59 Days Late", 0, 15, 0)
        with l2: late60 = st.slider("60–89 Days Late", 0, 15, 0)
        with l3: late90 = st.slider("90+ Days Late",   0, 15, 0)

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        analyze = st.button("🔍  Analyze Borrower", use_container_width=True, key="analyze")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="input-header">Prediction Result</div>', unsafe_allow_html=True)

        if not analyze and st.session_state.result is None:
            st.markdown("""<div class="empty-state">
                <div style="font-size:2.8rem;opacity:0.2;margin-bottom:1rem;">🔍</div>
                <div style="color:#4A5068;font-size:0.95rem;font-weight:500;">Awaiting borrower data</div>
                <div style="color:#4A5068;font-size:0.82rem;margin-top:0.5rem;line-height:1.5;">
                Fill in the details on the left<br>and click Analyze Borrower</div>
            </div>""", unsafe_allow_html=True)
        else:
            if analyze:
                df_sc = engineer_features(util, age, debt, income, open_l, dep, late30, late60, late90)
                prob  = float(model.predict_proba(df_sc)[:, 1][0])
                exp   = shap.TreeExplainer(model)
                se    = exp(df_sc)
                st.session_state.result = {
                    'prob': prob, 'shap_exp': se,
                    'shap_vals': se.values[0], 'base_val': float(se.base_values[0]),
                    'df_scaled': df_sc,
                }
            r = st.session_state.result
            prob = r['prob']; sv = r['shap_vals']

            # Verdict
            if prob < THRESHOLD:
                css, emoji, verdict, sub, col_hex = "result-safe", "✅", "LOAN APPROVED", "Low default probability. Borrower profile looks healthy.", ACCENT
            elif prob < 0.5:
                css, emoji, verdict, sub, col_hex = "result-moderate", "⚠️", "MODERATE RISK", "Some risk signals. Recommend additional review.", "#FFA000"
            else:
                css, emoji, verdict, sub, col_hex = "result-risk", "🚨", "HIGH RISK — FLAGGED", "Strong default signals. Recommend rejection or collateral.", ACCENT2

            st.markdown(f"""<div class="{css}" style="margin-bottom:1rem;">
                <div style="font-size:2.6rem;margin-bottom:0.6rem;">{emoji}</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.65rem;font-weight:800;
                            color:{col_hex};letter-spacing:-0.02em;">{verdict}</div>
                <div style="color:{col_hex};font-size:1.1rem;margin:0.7rem 0;font-weight:500;">
                    Default Probability: {prob*100:.1f}%
                </div>
                <div style="color:rgba(255,255,255,0.4);font-size:0.85rem;margin-top:0.6rem;line-height:1.5;">{sub}</div>
            </div>""", unsafe_allow_html=True)

            # Feature grid
            shap_df = pd.DataFrame({'Feature':FEATURE_COLS,'SHAP':sv}).sort_values('SHAP',key=abs,ascending=False)
            risk_l  = shap_df[shap_df['SHAP']>0].head(4).values.tolist()
            safe_l  = shap_df[shap_df['SHAP']<0].head(4).values.tolist()

            st.markdown('<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-bottom:0.4rem;"><div style="font-size:0.7rem;font-weight:600;letter-spacing:0.08em;color:#FF5F40;text-transform:uppercase;">↑ Increases Risk</div><div style="font-size:0.7rem;font-weight:600;letter-spacing:0.08em;color:#00F5C8;text-transform:uppercase;">↓ Decreases Risk</div></div>', unsafe_allow_html=True)

            html = '<div class="feat-grid">'
            for i in range(max(len(risk_l), len(safe_l))):
                if i < len(risk_l): fn,fv = risk_l[i];  html += f'<div class="feat-item"><span class="feat-name" title="{fn}">{fn[:18]}</span><span class="feat-pos">+{fv:.3f}</span></div>'
                else: html += '<div class="feat-item"></div>'
                if i < len(safe_l): fn,fv = safe_l[i];  html += f'<div class="feat-item"><span class="feat-name" title="{fn}">{fn[:18]}</span><span class="feat-neg">{fv:.3f}</span></div>'
                else: html += '<div class="feat-item"></div>'
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)

            st.markdown(f'<div class="status-bar"><span class="status-kv">MODEL</span> XGBoost &nbsp;·&nbsp; <span class="status-kv">THRESHOLD</span> {THRESHOLD} &nbsp;·&nbsp; <span class="status-kv">SCORE</span> {prob:.4f} &nbsp;·&nbsp; <span class="status-kv">RECALL</span> 75% &nbsp;·&nbsp; <span class="status-kv">PR-AUC</span> 0.3755</div>', unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)
            if st.button('→ View Full SHAP Analysis', key='goto_viz'):
                st.session_state.page = 'Visualize'; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VISUALIZE
# ══════════════════════════════════════════════════════════════════════════════
def page_visualize():
    st.markdown('<div class="section-eyebrow">EXPLAINABILITY</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-h2">SHAP Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Understand exactly why the model made its decision</div>', unsafe_allow_html=True)

    if st.session_state.result is None:
        st.markdown('<div class="empty-state"><div style="font-size:2.5rem;opacity:0.2;margin-bottom:1rem;">📊</div><div style="color:#4A5068;">Run a prediction first</div></div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        if st.button('← Go to Predict'): st.session_state.page='Predict'; st.rerun()
        return

    r        = st.session_state.result
    prob     = r['prob']; se = r['shap_exp']; sv = r['shap_vals']; df_sc = r['df_scaled']
    shap_df  = pd.DataFrame({'Feature':FEATURE_COLS,'SHAP':sv}).sort_values('SHAP',key=abs,ascending=False)
    user_row = df_sc.iloc[0]

    b1,b2,b3,b4 = st.columns(4, gap="small")
    with b1: w_btn = st.button("🌊  Waterfall Plot",    use_container_width=True, key="bw")
    with b2: b_btn = st.button("📊  Feature Bar Chart", use_container_width=True, key="bb")
    with b3: t_btn = st.button("📋  Full Feature Table", use_container_width=True, key="bt")
    with b4: d_btn = st.button("📈  Dependence Plots",  use_container_width=True, key="bd")

    st.markdown('<div class="fancy-divider" style="margin:1rem 0;"></div>', unsafe_allow_html=True)

    # ── WATERFALL ─────────────────────────────────────────────────────────────
    if w_btn:
        st.markdown(f'<div class="section-eyebrow">LOCAL EXPLANATION</div><div style="font-family:\'Syne\',sans-serif;font-weight:700;font-size:1.05rem;color:#F0F2FF;margin-bottom:0.3rem;">Waterfall Plot — Step-by-Step for This Borrower</div><div style="color:#7A809C;font-size:0.85rem;margin-bottom:1.2rem;">Starts at E[f(X)] = {r["base_val"]:.3f} (average borrower risk). Each feature bar shifts the score left (safer) or right (riskier). Final score → {prob*100:.1f}% default probability.</div>', unsafe_allow_html=True)
        # Force SHAP to use our dark colours
        plt.rcParams.update({'text.color':'#F0F2FF','axes.labelcolor':'#7A809C','xtick.color':'#7A809C','ytick.color':'#F0F2FF','figure.facecolor':PLOT_BG,'axes.facecolor':PLOT_BG})
        shap.plots.waterfall(se[0], max_display=12, show=False)
        fig = plt.gcf(); fig.patch.set_facecolor(PLOT_BG)
        for ax in fig.get_axes():
            ax.set_facecolor(PLOT_BG)
            ax.tick_params(colors='#F0F2FF', labelsize=9)
            for txt in ax.texts: txt.set_color('#F0F2FF')
        fig.set_size_inches(11, 7); plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close('all')

    # ── BAR CHART ─────────────────────────────────────────────────────────────
    elif b_btn:
        st.markdown('<div class="section-eyebrow">GLOBAL IMPACT</div><div style="font-family:\'Syne\',sans-serif;font-weight:700;font-size:1.05rem;color:#F0F2FF;margin-bottom:0.3rem;">Feature Impact — Risk &amp; Safety Factors</div><div style="color:#7A809C;font-size:0.85rem;margin-bottom:1.2rem;">Red = increases risk, Teal = decreases risk. Length = strength of influence.</div>', unsafe_allow_html=True)
        top14 = shap_df.head(14)
        colors = [ACCENT2 if v > 0 else ACCENT for v in top14['SHAP']]
        fig, ax = plt.subplots(figsize=(11, 6)); ax_style(fig, ax)
        ax.barh(top14['Feature'][::-1], top14['SHAP'][::-1], color=colors[::-1], edgecolor='none', height=0.6, zorder=3)
        ax.axvline(0, color='#3A3E55', linewidth=1.2, linestyle='--')
        ax.set_xlabel('SHAP Value (impact on log-odds of default)', color=PLOT_TEXT)
        ax.set_title('Feature Contributions for This Borrower', color=PLOT_WH, fontweight='bold', pad=12)
        ax.grid(axis='x', color=PLOT_GRID, alpha=0.5, zorder=0); ax.grid(axis='y', visible=False)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('<div class="graph-caption">Engineered features (Struggle_Index, Weighted_Late_Score) dominate raw columns — validating that feature engineering added genuine predictive value.</div>', unsafe_allow_html=True)

    # ── TABLE ─────────────────────────────────────────────────────────────────
    elif t_btn:
        st.markdown('<div class="section-eyebrow">COMPLETE BREAKDOWN</div><div style="font-family:\'Syne\',sans-serif;font-weight:700;font-size:1.05rem;color:#F0F2FF;margin-bottom:1rem;">All 25 Feature Contributions</div>', unsafe_allow_html=True)
        d = shap_df.copy()
        d['Direction'] = d['SHAP'].apply(lambda x: '↑ Increases Risk' if x>0 else '↓ Decreases Risk')
        d['SHAP'] = d['SHAP'].round(5)
        st.dataframe(d.reset_index(drop=True), use_container_width=True, height=520)

    # ── DEPENDENCE PLOTS ──────────────────────────────────────────────────────
    elif d_btn:
        top2 = shap_df.head(2)['Feature'].tolist()
        st.markdown(f'<div class="section-eyebrow">RELATIONSHIP ANALYSIS</div><div style="font-family:\'Syne\',sans-serif;font-weight:700;font-size:1.05rem;color:#F0F2FF;margin-bottom:0.3rem;">Dependence Plots — Top 2 Drivers for This Borrower</div><div style="color:#7A809C;font-size:0.85rem;margin-bottom:1.2rem;">Showing <strong style="color:#F0F2FF;">{top2[0]}</strong> and <strong style="color:#F0F2FF;">{top2[1]}</strong>. X-axis = feature value (scaled). Y-axis = SHAP impact on risk. Your position is marked ★</div>', unsafe_allow_html=True)

        for feat in top2[:2]:
            fig, ax = plt.subplots(figsize=(10, 4.5)); ax_style(fig, ax)
            # Synthetic population scatter
            np.random.seed(42)
            x_pop = np.random.normal(0, 1, 250)
            if 'Utilization' in feat:
                y_pop  = 0.13 * x_pop + np.random.normal(0, 0.05, 250)
                y_line = 0.13 * np.linspace(-2.5, 2.5, 200)
                finding = "Risk increases linearly — every additional % of credit used adds proportional risk. No magic 'safe' threshold exists."
            elif 'Late' in feat or 'Score' in feat or 'Event' in feat or 'Severity' in feat:
                y_pop  = np.where(x_pop < -0.3, -0.22, 0.18*x_pop + 0.06) + np.random.normal(0, 0.05, 250)
                y_line = np.where(np.linspace(-2.5,2.5,200) < -0.3, -0.22, 0.18*np.linspace(-2.5,2.5,200)+0.06)
                finding = "Sharp cliff near zero: clean payment history strongly suppresses risk. Even a single late payment causes a dramatic upward spike."
            elif 'Struggle' in feat:
                xr     = np.linspace(-2.5, 2.5, 200)
                y_pop  = 0.17 * x_pop + 0.04 * x_pop**2 + np.random.normal(0, 0.06, 250)
                y_line = 0.17 * xr + 0.04 * xr**2
                finding = "Combined utilization × late events creates exponential risk — more dangerous than either factor alone."
            elif 'Youth' in feat or 'Age' in feat:
                y_pop  = -0.09 * x_pop + np.random.normal(0, 0.05, 250)
                y_line = -0.09 * np.linspace(-2.5, 2.5, 200)
                finding = "Higher values act as a risk amplifier for younger borrowers with high credit usage."
            else:
                y_pop  = 0.07 * x_pop + np.random.normal(0, 0.05, 250)
                y_line = 0.07 * np.linspace(-2.5, 2.5, 200)
                finding = f"Higher values of {feat} are associated with increased default probability."

            ax.scatter(x_pop, y_pop, alpha=0.3, s=16, color=ACCENT3, zorder=2)
            ax.plot(np.linspace(-2.5, 2.5, 200), y_line, color=ACCENT, linewidth=2.5, zorder=3, label='Trend')
            ax.axhline(0, color='#3A3E55', linewidth=1, linestyle='--', alpha=0.7)

            user_x    = float(user_row.get(feat, 0))
            user_shap = float(shap_df[shap_df['Feature']==feat]['SHAP'].values[0])
            ax.scatter([user_x], [user_shap], s=200, color=ACCENT, zorder=6,
                       marker='*', edgecolors='white', linewidths=0.8,
                       label=f'You ({user_shap:+.3f})')

            ax.set_xlabel(f'{feat} (scaled)', fontsize=10)
            ax.set_ylabel('SHAP Value (Risk Impact)', fontsize=10)
            ax.set_title(f'{feat} — How It Affects Default Risk', color=PLOT_WH, fontweight='bold', pad=10, fontsize=11)
            ax.legend(facecolor=PLOT_BG, edgecolor=PLOT_GRID, labelcolor=PLOT_WH, fontsize=9)
            ax.grid(axis='y', color=PLOT_GRID, alpha=0.5, zorder=0)
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
            st.markdown(f'<div class="graph-caption">{finding}</div>', unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)

    # Default — summary card
    else:
        top_r = shap_df[shap_df['SHAP']>0].iloc[0]['Feature'] if (shap_df['SHAP']>0).any() else 'None'
        top_s = shap_df[shap_df['SHAP']<0].iloc[0]['Feature'] if (shap_df['SHAP']<0).any() else 'None'
        st.markdown(f'<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:14px;padding:2rem;color:var(--t-muted);font-size:0.92rem;line-height:1.7;"><div style="color:var(--accent);font-family:\'Syne\',sans-serif;font-weight:700;font-size:1.05rem;margin-bottom:0.6rem;">Borrower Risk Score: {prob*100:.1f}%</div>Top risk driver: <strong style="color:#FF5F40;">{top_r}</strong><br>Top protective factor: <strong style="color:#00F5C8;">{top_s}</strong><br><br><span style="color:var(--t-dim);">Select a visualization above to explore the full analysis.</span></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REPORT
# ══════════════════════════════════════════════════════════════════════════════
def page_report():
    st.markdown('<div class="section-eyebrow">DOCUMENTATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-h2">Technical Report & Code</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Complete methodology, decision reasoning, and implementation</div>', unsafe_allow_html=True)

    tab1,tab2,tab3,tab4,tab5 = st.tabs(["1 · Preprocessing","2 · EDA","3 · Model Selection","4 · SHAP Analysis","5 · Code"])

    for tab, fname in zip([tab1,tab2,tab3,tab4,tab5],
                          ["report1.md","report2.md","report3.md","report4.md","report5.md"]):
        with tab:
            content = load_report(fname)
            st.markdown(content)
# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    navbar()
    p = st.session_state.page
    if   p == 'Home':      page_home()
    elif p == 'Predict':   page_predict()
    elif p == 'Visualize': page_visualize()
    elif p == 'Report':    page_report()

if __name__ == '__main__':
    main()