"""
Lab-4: Interactive Statistical Modeling Dashboard
Dataset: Medical Insurance Costs (medical_insurance.csv)
 
Run with:  streamlit run app.py
"""
 
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
 

# Page setup
st.set_page_config(
    page_title="Insurance Charges Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
BG = "#0B1220"
PANEL = "#121B2E"
SIDEBAR = "#161F36"
BORDER = "rgba(255,255,255,0.08)"
TEAL = "#2DD4BF"
INDIGO = "#6366F1"
CORAL = "#FB7185"
AMBER = "#F5B54C"
TEXT = "#E7ECF5"
MUTED = "#8B95AC"
GRADIENT = f"linear-gradient(135deg, {TEAL} 0%, {INDIGO} 100%)"
 

# CSS
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
 
<style>
:root {{
    --bg: {BG}; --panel: {PANEL}; --sidebar: {SIDEBAR}; --border: {BORDER};
    --teal: {TEAL}; --indigo: {INDIGO}; --coral: {CORAL}; --amber: {AMBER};
    --text: {TEXT}; --muted: {MUTED};
}}
 
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
h1, h2, h3, h4, .app-title {{ font-family: 'Space Grotesk', sans-serif; }}
 
.stApp {{ background: var(--bg); color: var(--text); }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{
    background: var(--sidebar);
    border-right: 1px solid var(--border);
}}
[data-testid="stSidebar"] > div:first-child {{ padding-top: 1.2rem; }}
 
/* ---- One orchestrated entrance for the header, nothing else fades ---- */
@keyframes riseIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.hero {{ animation: riseIn 0.6s ease-out; }}
 
/* ---- Sidebar brand block ---- */
.brand {{
    padding: 0.4rem 0.2rem 1.1rem 0.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}}
.brand-mark {{
    width: 38px; height: 38px; border-radius: 10px;
    background: {GRADIENT};
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; margin-bottom: 0.6rem;
}}
.brand-title {{ font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.05rem; color: var(--text); line-height: 1.25; }}
.brand-sub {{ font-size: 0.78rem; color: var(--muted); margin-top: 0.15rem; }}
 
/* ---- Vertical nav pills (the app's controller) ---- */
[data-testid="stSidebar"] .stRadio > label {{ display: none; }}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] {{ gap: 0.35rem; }}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 0.55rem 0.8rem;
    transition: background 0.15s ease, border-color 0.15s ease;
    width: 100%;
}}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{
    background: rgba(255,255,255,0.04);
    border-color: var(--border);
}}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-checked="true"] {{
    background: linear-gradient(135deg, rgba(45,212,191,0.16), rgba(99,102,241,0.16));
    border-color: rgba(45,212,191,0.35);
}}
 
.control-label {{
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.01em;
    color: var(--muted); margin: 1.3rem 0 0.4rem 0.1rem;
}}
 
/* ---- KPI cards ---- */
.kpi {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-top: 3px solid;
    border-image: {GRADIENT} 1;
    border-radius: 12px;
    padding: 0.95rem 1.1rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.kpi:hover {{ transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.35); }}
.kpi-label {{ font-size: 0.78rem; color: var(--muted); margin-bottom: 0.25rem; }}
.kpi-value {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.55rem; font-weight: 600; color: var(--text); }}
 
/* ---- Panels ---- */
.panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 1rem;
}}
.panel h4 {{ margin-top: 0; }}
 
/* ---- Verdict banners ---- */
.verdict {{
    border-radius: 10px; padding: 0.8rem 1rem; font-size: 0.92rem;
    border: 1px solid var(--border);
}}
.verdict.reject {{ background: rgba(251,113,133,0.10); border-color: rgba(251,113,133,0.35); }}
.verdict.fail {{ background: rgba(45,212,191,0.10); border-color: rgba(45,212,191,0.35); }}
 
/* ---- Misc widget theming ---- */
.stSlider [data-baseweb="slider"] > div > div {{ background: {GRADIENT} !important; }}
.stButton > button {{
    background: {GRADIENT}; color: #0B1220; font-weight: 600;
    border: none; border-radius: 8px; transition: filter 0.15s ease;
}}
.stButton > button:hover {{ filter: brightness(1.08); }}
hr {{ border-color: var(--border); }}
::-webkit-scrollbar {{ width: 8px; }}
::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.15); border-radius: 8px; }}
</style>
""", unsafe_allow_html=True)
 
 
def kpi_card(label, value):
    st.markdown(f"""<div class="kpi"><div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div></div>""", unsafe_allow_html=True)
 
 
def verdict_banner(reject, text):
    cls = "reject" if reject else "fail"
    icon = "🔴" if reject else "🟢"
    st.markdown(f'<div class="verdict {cls}">{icon} &nbsp;{text}</div>', unsafe_allow_html=True)
 
 
def themed(fig, title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(family="Inter", color=TEXT, size=12),
        title=dict(text=title, font=dict(family="Space Grotesk", size=15)) if title else None,
        colorway=[TEAL, INDIGO, CORAL, AMBER],
        margin=dict(t=50 if title else 20, l=10, r=10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig
 
 

# Data + model (cached)
@st.cache_data
def load_data():
    df = pd.read_csv("medical_insurance.csv")
    return df.drop_duplicates().reset_index(drop=True)
 
 
@st.cache_resource
def fit_model(df):
    return smf.ols("charges ~ age + bmi + children + C(sex) + C(smoker) + C(region)", data=df).fit()
 
 
df = load_data()
model = fit_model(df)

# Sidebar — (nav + contextual controls)
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-mark">📈</div>
        <div class="brand-title">Insurance Charges Analysis</div>
        <div class="brand-sub">Statistical modeling & diagnostics</div>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown('<div class="control-label">VIEW</div>', unsafe_allow_html=True)
    view = st.radio(
        "view", ["📊  Data Exploration", "🧪  Hypothesis Testing", "🔮  Live Prediction"],
        label_visibility="collapsed",
    )
 
    st.markdown('<div class="control-label">CONTROLS</div>', unsafe_allow_html=True)
 
    if view.endswith("Exploration"):
        age_range = st.slider("Age range", int(df.age.min()), int(df.age.max()),
                               (int(df.age.min()), int(df.age.max())))
        bmi_range = st.slider("BMI range", float(df.bmi.min()), float(df.bmi.max()),
                               (float(df.bmi.min()), float(df.bmi.max())))
        regions = st.multiselect("Region(s)", sorted(df.region.unique()), default=sorted(df.region.unique()))
        smoker_filter = st.multiselect("Smoker status", sorted(df.smoker.unique()), default=sorted(df.smoker.unique()))
        sex_filter = st.multiselect("Sex", sorted(df.sex.unique()), default=sorted(df.sex.unique()))
 
    elif view.endswith("Testing"):
        cat_cols = ["sex", "smoker", "region"]
        num_cols = ["age", "bmi", "children", "charges"]
        factor = st.selectbox("Categorical factor", cat_cols, index=1)
        metric = st.selectbox("Numeric metric", num_cols, index=3)
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)
        st.markdown('<div class="control-label">CHI-SQUARE PAIR</div>', unsafe_allow_html=True)
        cat1 = st.selectbox("Factor 1", cat_cols, index=1, key="chi1")
        cat2 = st.selectbox("Factor 2", [c for c in cat_cols if c != cat1], index=0, key="chi2")
 
    else:
        in_age = st.slider("Age", 18, 64, 35)
        in_bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.0, step=0.1)
        in_children = st.slider("Children", 0, 5, 0)
        in_sex = st.selectbox("Sex", sorted(df.sex.unique()))
        in_smoker = st.selectbox("Smoker", sorted(df.smoker.unique()))
        in_region = st.selectbox("Region", sorted(df.region.unique()))
 
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"{len(df):,} de-duplicated records · OLS R² = {model.rsquared:.3f}")

# Header

st.markdown(f"""
<div class="hero" style="margin-bottom:1.4rem;">
    <div style="font-family:'Space Grotesk',sans-serif; font-size:1.9rem; font-weight:700;
        background:{GRADIENT}; -webkit-background-clip:text; background-clip:text; color:transparent;
        display:inline-block;">
        {view.split("  ")[1] if "  " in view else view}
    </div>
    <div style="color:{MUTED}; font-size:0.95rem; margin-top:0.15rem;">
        Medical insurance charges · exploration, testing, and live prediction in one control panel
    </div>
</div>
""", unsafe_allow_html=True)
 
# VIEW 1 — Data Exploration
if view.endswith("Exploration"):
    fdf = df[
        df.age.between(*age_range) & df.bmi.between(*bmi_range)
        & df.region.isin(regions) & df.smoker.isin(smoker_filter) & df.sex.isin(sex_filter)
    ]
 
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Rows matched", f"{len(fdf):,}")
    with c2: kpi_card("Mean charge", f"${fdf.charges.mean():,.0f}")
    with c3: kpi_card("Median charge", f"${fdf.charges.median():,.0f}")
    with c4: kpi_card("Mean BMI", f"{fdf.bmi.mean():.1f}")
 
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        fig = px.histogram(fdf, x="charges", color="smoker", nbins=40, marginal="box")
        st.plotly_chart(themed(fig, "Charges distribution by smoker status"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        fig2 = px.scatter(fdf, x="bmi", y="charges", color="smoker", size="age",
                           hover_data=["age", "children", "region"])
        st.plotly_chart(themed(fig2, "BMI vs. charges (bubble size = age)"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
 
    col3, col4 = st.columns([1.3, 1])
    with col3:
        st.markdown('<div class="panel"><h4>Summary statistics</h4>', unsafe_allow_html=True)
        st.dataframe(fdf.describe().round(2), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        corr = fdf[["age", "bmi", "children", "charges"]].corr()
        fig3 = px.imshow(corr, text_auto=".2f", color_continuous_scale=[TEAL, PANEL, INDIGO], zmin=-1, zmax=1)
        st.plotly_chart(themed(fig3, "Correlation heatmap"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# VIEW 2 — Hypothesis Testing Lab
elif view.endswith("Testing"):
    groups = df[factor].unique()
 
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f"#### {metric} across **{factor}** groups")
 
    if len(groups) == 2:
        g1_name, g2_name = groups
        g1 = df.loc[df[factor] == g1_name, metric]
        g2 = df.loc[df[factor] == g2_name, metric]
 
        sw1 = stats.shapiro(g1.sample(min(500, len(g1)), random_state=1))
        sw2 = stats.shapiro(g2.sample(min(500, len(g2)), random_state=1))
        lev = stats.levene(g1, g2)
        normal = (sw1.pvalue > alpha) and (sw2.pvalue > alpha)
        equal_var = lev.pvalue > alpha
 
        cA, cB, cC = st.columns(3)
        with cA: kpi_card(f"Shapiro p ({g1_name})", f"{sw1.pvalue:.2e}")
        with cB: kpi_card(f"Shapiro p ({g2_name})", f"{sw2.pvalue:.2e}")
        with cC: kpi_card("Levene p", f"{lev.pvalue:.2e}")
 
        st.caption(f"Normal? {'✅ yes' if normal else '❌ no'}  ·  Equal variance? {'✅ yes' if equal_var else '❌ no'}")
 
        if normal:
            stat, pval = stats.ttest_ind(g1, g2, equal_var=equal_var)
            test_used = "Two-sample t-test" if equal_var else "Welch's t-test"
        else:
            stat, pval = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            test_used = "Mann-Whitney U test"
 
        st.write("")
        r1, r2, r3 = st.columns(3)
        with r1: kpi_card("Test used", test_used)
        with r2: kpi_card("Statistic", f"{stat:.3f}")
        with r3: kpi_card("p-value", f"{pval:.3e}")
 
        st.write("")
        verdict_banner(pval < alpha,
                        f"{'Reject' if pval < alpha else 'Fail to reject'} H0 at α = {alpha} — "
                        f"{'a significant difference was found' if pval < alpha else 'no significant difference detected'} "
                        f"between {g1_name} and {g2_name} on {metric}.")
 
        fig = px.box(df, x=factor, y=metric, color=factor, points="all")
        st.plotly_chart(themed(fig, f"{metric} by {factor}"), use_container_width=True)
 
    else:
        region_groups = [df.loc[df[factor] == g, metric].values for g in groups]
        f_stat, p_anova = stats.f_oneway(*region_groups)
        lev = stats.levene(*region_groups)
        kw_stat, kw_p = stats.kruskal(*region_groups)
 
        r1, r2, r3 = st.columns(3)
        with r1: kpi_card("ANOVA F", f"{f_stat:.3f}")
        with r2: kpi_card("ANOVA p", f"{p_anova:.4f}")
        with r3: kpi_card("Levene p", f"{lev.pvalue:.4f}")
 
        st.write("")
        verdict_banner(p_anova < alpha, f"{'Reject' if p_anova < alpha else 'Fail to reject'} H0 at α = {alpha} (One-Way ANOVA).")
        st.caption(f"Robustness check — Kruskal-Wallis (non-parametric): stat={kw_stat:.3f}, p={kw_p:.4f}")
 
        fig = px.box(df, x=factor, y=metric, color=factor, points="all")
        st.plotly_chart(themed(fig, f"{metric} by {factor}"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
 
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f"#### Chi-square: **{cat1}** × **{cat2}**")
    contingency = pd.crosstab(df[cat1], df[cat2])
    chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
    cX, cY = st.columns([1, 1.4])
    with cX:
        st.dataframe(contingency, use_container_width=True)
    with cY:
        r1, r2 = st.columns(2)
        with r1: kpi_card("Chi-square", f"{chi2:.3f}")
        with r2: kpi_card("p-value", f"{chi_p:.4f}")
        verdict_banner(chi_p < alpha, f"{'Reject' if chi_p < alpha else 'Fail to reject'} H0 — "
                        f"{'associated' if chi_p < alpha else 'independent'} at α = {alpha}.")
    st.markdown('</div>', unsafe_allow_html=True)
 
# VIEW 3 — Live Prediction & Diagnostics

else:
    new_point = pd.DataFrame([{"age": in_age, "bmi": in_bmi, "children": in_children,
                                "sex": in_sex, "smoker": in_smoker, "region": in_region}])
    pred = model.get_prediction(new_point)
    ps = pred.summary_frame(alpha=0.05)
 
    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Predicted charge", f"${ps['mean'][0]:,.0f}")
    with c2: kpi_card("95% CI (mean)", f"${ps['mean_ci_lower'][0]:,.0f} – ${ps['mean_ci_upper'][0]:,.0f}")
    with c3: kpi_card("95% prediction interval", f"${ps['obs_ci_lower'][0]:,.0f} – ${ps['obs_ci_upper'][0]:,.0f}")
 
    st.caption("The confidence interval bounds the **average** charge for this profile. "
               "The wider prediction interval bounds where a **single new person's** actual charge is likely to fall.")
 
    st.write("")
    fitted, resid = model.fittedvalues, model.resid
    d1, d2 = st.columns(2)
    with d1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        fig = px.scatter(x=fitted, y=resid, labels={"x": "Fitted values", "y": "Residuals"})
        fig.add_hline(y=0, line_dash="dash", line_color=CORAL)
        st.plotly_chart(themed(fig, "Residuals vs. fitted"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with d2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        qq = sm.ProbPlot(resid)
        tq, sq = qq.theoretical_quantiles, qq.sample_quantiles
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=tq, y=sq, mode="markers", name="Residuals",
                                   marker=dict(color=TEAL, size=5, opacity=0.6)))
        line_x = np.array([tq.min(), tq.max()])
        fig2.add_trace(go.Scatter(x=line_x, y=line_x * resid.std() + resid.mean(),
                                   mode="lines", name="Reference", line=dict(color=CORAL, dash="dash")))
        st.plotly_chart(themed(fig2, "Q-Q plot of residuals"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
 
    st.markdown('<div class="panel"><h4>Variance Inflation Factor</h4>', unsafe_allow_html=True)
    X = sm.add_constant(df[["age", "bmi", "children"]])
    vif_df = pd.DataFrame({"feature": X.columns,
                            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
    v1, v2 = st.columns([1, 2])
    with v1:
        st.dataframe(vif_df, use_container_width=True, hide_index=True)
    with v2:
        st.caption("Rule of thumb: VIF above 5–10 signals a multicollinearity concern. "
                    "All three predictors sit near 1.0, so none is flagged here.")
    st.markdown('</div>', unsafe_allow_html=True)
 
    with st.expander("Full OLS regression summary"):
        st.text(str(model.summary()))
 
        