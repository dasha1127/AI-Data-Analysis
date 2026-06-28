import os
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from agent import analyze, auto_insights
from data_loader import load_combined, most_improved

load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY", "")

st.set_page_config(
    page_title="DataMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# Teaching note:
#   @keyframes  = defines an animation (like a loop of CSS states)
#   gradient    = smooth blend between colors
#   glassmorphism = frosted-glass card using backdrop-filter + semi-transparent bg
#   transition  = smooth change when hovering
#   box-shadow with color = glowing border effect
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Animated gradient background on the app ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a0a1a 100%);
    background-size: 400% 400%;
    animation: bgShift 12s ease infinite;
}
@keyframes bgShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Sidebar glassmorphism ── */
[data-testid="stSidebar"] {
    background: rgba(15, 20, 40, 0.85) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(79, 139, 249, 0.2);
}

/* ── Animated gradient title text ── */
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #4F8BF9, #a78bfa, #38bdf8, #4F8BF9);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: textShine 4s linear infinite;
    margin: 0;
    letter-spacing: -1px;
}
@keyframes textShine {
    0%   { background-position: 0% center; }
    100% { background-position: 300% center; }
}

.hero-sub {
    color: #7ca3e0;
    font-size: 1.05rem;
    margin-top: 4px;
    margin-bottom: 24px;
}

/* ── Glowing metric cards ── */
.metric-card {
    background: rgba(79, 139, 249, 0.08);
    border: 1px solid rgba(79, 139, 249, 0.3);
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    transition: all 0.3s ease;
    animation: fadeUp 0.5s ease both;
}
.metric-card:hover {
    background: rgba(79, 139, 249, 0.18);
    box-shadow: 0 0 20px rgba(79, 139, 249, 0.35);
    transform: translateY(-3px);
}
.metric-num  { font-size: 2rem; font-weight: 800; color: #4F8BF9; }
.metric-label{ font-size: 0.75rem; color: #7ca3e0; text-transform: uppercase; letter-spacing: 1px; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Chat bubbles ── */
.user-bubble {
    background: linear-gradient(135deg, #1e3a5f, #1a2a4a);
    border: 1px solid rgba(79,139,249,0.4);
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    margin: 10px 0 10px 60px;
    color: #e0eaff;
    animation: slideLeft 0.3s ease;
}
.agent-bubble {
    background: linear-gradient(135deg, #0d2416, #0a1f12);
    border: 1px solid rgba(52, 211, 153, 0.35);
    border-radius: 18px 18px 18px 4px;
    padding: 12px 18px;
    margin: 10px 60px 10px 0;
    color: #d1fae5;
    animation: slideRight 0.3s ease;
}
@keyframes slideLeft  { from { opacity:0; transform:translateX(20px); } to { opacity:1; transform:translateX(0); } }
@keyframes slideRight { from { opacity:0; transform:translateX(-20px);} to { opacity:1; transform:translateX(0); } }

.bubble-label { font-size: 0.7rem; opacity: 0.5; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 1px; }

/* ── Glassy info boxes in sidebar ── */
.info-card {
    background: rgba(79,139,249,0.07);
    border: 1px solid rgba(79,139,249,0.2);
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 0.82rem;
    color: #a0b8e0;
    line-height: 1.7;
}
.info-card b { color: #7eb8f9; }

/* ── Status pill ── */
.pill-on  { background:#064e3b; color:#6ee7b7; border:1px solid #059669;
            border-radius:20px; padding:3px 12px; font-size:0.75rem; display:inline-block; }
.pill-off { background:#1e1b4b; color:#a5b4fc; border:1px solid #4338ca;
            border-radius:20px; padding:3px 12px; font-size:0.75rem; display:inline-block; }

/* ── Tab styling ── */
[data-testid="stTabs"] button {
    color: #7ca3e0 !important;
    font-weight: 600;
    font-size: 0.9rem;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #4F8BF9 !important;
    border-bottom: 2px solid #4F8BF9 !important;
}

/* ── Plotly chart cards ── */
[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(79,139,249,0.15);
    border-radius: 14px;
    padding: 4px;
}

/* ── Hide default Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 DataMind AI")

    # LLM status pill — reads key from .env only (not shown in UI for security)
    if GROQ_KEY:
        st.markdown("<span class='pill-on'>● LLM ON — Llama 3.3 70B</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='pill-off'>○ Rule-based mode</span>", unsafe_allow_html=True)
        st.caption("Add GROQ_API_KEY to .env to enable LLM")

    st.divider()
    st.markdown("**📂 Load Data**")

    upload_tab, sample_tab = st.tabs(["Upload", "Samples"])
    df = None
    dataset_name = ""

    with upload_tab:
        uploaded = st.file_uploader("Drop any CSV", type=["csv", "xlsx"], label_visibility="collapsed")
        if uploaded:
            try:
                df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
                dataset_name = uploaded.name
                st.success(f"{df.shape[0]:,} rows loaded")
            except Exception as e:
                st.error(str(e))

    with sample_tab:
        sample = st.selectbox("Dataset", [
            "None",
            "World Happiness 2015–2022",
            "Tips",
            "Iris",
        ], label_visibility="collapsed")
        if sample == "World Happiness 2015–2022":
            try:
                df = load_combined()
                dataset_name = "World Happiness 2015–2022"
                st.success(f"{df.shape[0]:,} rows · 8 years")
            except FileNotFoundError:
                st.error("Download the dataset first (see README)")
        elif sample == "Tips":
            df, dataset_name = px.data.tips(), "Tips"
            st.success("244 rows loaded")
        elif sample == "Iris":
            df, dataset_name = px.data.iris(), "Iris"
            st.success("150 rows loaded")

    if df is not None:
        st.divider()
        st.markdown("**📋 Schema**")
        for col in df.columns:
            dtype = str(df[col].dtype)
            icon = "🔢" if any(t in dtype for t in ("int", "float")) else "📝"
            st.caption(f"{icon} `{col}`")

    st.divider()
    is_happiness = df is not None and "happiness_score" in df.columns
    tips = [
        ("Happiness trend over years", is_happiness),
        ("Top 10 happiest countries", is_happiness),
        ("Correlation between GDP and happiness", is_happiness),
        ("Compare happiness by region", is_happiness),
        ("Show correlation heatmap", not is_happiness),
        ("Distribution of &lt;column&gt;", not is_happiness),
        ("Top 10 by &lt;column&gt;", not is_happiness),
        ("Outliers in &lt;column&gt;", not is_happiness),
    ]
    tip_lines = "<br>".join(f"• {t}" for t, show in tips if show)
    st.markdown(f"<div class='info-card'><b>💡 Try asking:</b><br>{tip_lines}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>🧠 DataMind AI</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Upload any dataset · Ask in plain English · Get instant AI-powered insights</div>", unsafe_allow_html=True)

if df is None:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown("""<div class='metric-card'>
        <div class='metric-num'>01</div>
        <div class='metric-label'>Load any CSV</div>
        <br><small style='color:#7ca3e0'>Upload from your machine or pick a built-in dataset from the sidebar</small>
    </div>""", unsafe_allow_html=True)
    c2.markdown("""<div class='metric-card'>
        <div class='metric-num'>02</div>
        <div class='metric-label'>Auto Insights</div>
        <br><small style='color:#7ca3e0'>Correlation heatmaps, distributions, outlier detection — all instant</small>
    </div>""", unsafe_allow_html=True)
    c3.markdown("""<div class='metric-card'>
        <div class='metric-num'>03</div>
        <div class='metric-label'>Ask in English</div>
        <br><small style='color:#7ca3e0'>Type a question, get a chart and a text answer powered by Llama 3.3</small>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────────────────────────────────────
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
is_happiness = "happiness_score" in df.columns

c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    (f"{df.shape[0]:,}", "Rows"),
    (df.shape[1], "Columns"),
    (len(num_cols), "Numeric"),
    (len(cat_cols), "Categorical"),
    (df["year"].nunique() if is_happiness else f"{df.isnull().sum().sum():,}", "Years" if is_happiness else "Missing"),
]
for col, (num, label) in zip([c1, c2, c3, c4, c5], cards):
    col.markdown(f"""<div class='metric-card'>
        <div class='metric-num'>{num}</div>
        <div class='metric-label'>{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_labels = ["📊 Auto-Insights", "🌍 Happiness Trends" if is_happiness else "📈 Trends", "💬 Chat", "🗂 Raw Data"]
tabs = st.tabs(tab_labels)


# ── Tab 1: Auto-Insights ─────────────────────────────────────────────────────
with tabs[0]:
    with st.spinner("Generating insights..."):
        insights = auto_insights(df)

    if not insights:
        st.info("No automatic charts. Try the Chat tab!")
    else:
        for i in range(0, len(insights), 2):
            cols = st.columns(2)
            for j, insight in enumerate(insights[i: i + 2]):
                with cols[j]:
                    st.plotly_chart(insight["fig"], use_container_width=True)

    st.markdown("#### Statistical Summary")
    st.dataframe(df.describe(include="all").T, use_container_width=True)


# ── Tab 2: Happiness Trends ──────────────────────────────────────────────────
with tabs[1]:
    if is_happiness:
        avg_by_year = df.groupby("year")["happiness_score"].mean().reset_index()
        fig1 = px.line(
            avg_by_year, x="year", y="happiness_score",
            title="Global Average Happiness Score (2015–2022)",
            markers=True, template="plotly_dark",
        )
        fig1.update_traces(line=dict(color="#4F8BF9", width=3))
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig1, use_container_width=True)

        col_a, col_b = st.columns(2)
        latest_year = df["year"].max()

        with col_a:
            top10 = df[df["year"] == latest_year].nlargest(10, "happiness_score")
            fig2 = px.bar(
                top10, x="happiness_score", y="country", orientation="h",
                title=f"Top 10 Happiest Countries ({latest_year})",
                color="happiness_score", color_continuous_scale="Blues",
                template="plotly_dark",
            )
            fig2.update_layout(
                yaxis={"categoryorder": "total ascending"},
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            region_year = (
                df[df["region"].notna()]
                .groupby(["year", "region"])["happiness_score"]
                .mean().reset_index()
            )
            fig3 = px.line(
                region_year, x="year", y="happiness_score", color="region",
                title="Happiness by Region Over Time", markers=True,
                template="plotly_dark",
            )
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.scatter(
            df[df["year"] == latest_year].dropna(subset=["gdp_per_capita", "happiness_score"]),
            x="gdp_per_capita", y="happiness_score", color="region",
            hover_name="country", size="happiness_score",
            title=f"GDP per Capita vs Happiness ({latest_year})",
            trendline="ols", template="plotly_dark",
        )
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            improved = most_improved(df, top_n=10)
            fig5 = px.bar(
                improved, x="change", y="country", orientation="h",
                title="Most Improved Countries (First → Last Year)",
                color="change", color_continuous_scale="Greens",
                template="plotly_dark",
            )
            fig5.update_layout(
                yaxis={"categoryorder": "total ascending"},
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig5, use_container_width=True)

        with col_d:
            countries = sorted(df["country"].unique())
            selected = st.multiselect(
                "Track countries over time",
                countries,
                default=["Finland", "United States", "India", "China"],
            )
            if selected:
                fig6 = px.line(
                    df[df["country"].isin(selected)],
                    x="year", y="happiness_score", color="country",
                    markers=True, title="Country Spotlight",
                    template="plotly_dark",
                )
                fig6.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("Load the World Happiness dataset to see trend analysis.")


# ── Tab 3: Chat ──────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### 💬 Chat with your data")
    mode = "🟢 Llama 3.3 70B" if GROQ_KEY else "🔵 Rule-based"
    st.caption(f"Mode: {mode}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""<div class='user-bubble'>
                <div class='bubble-label'>You</div>
                {msg['content']}
            </div>""", unsafe_allow_html=True)
        else:
            if msg.get("text"):
                st.markdown(f"""<div class='agent-bubble'>
                    <div class='bubble-label'>DataMind</div>
                    {msg['text']}
                </div>""", unsafe_allow_html=True)
            if msg.get("fig"):
                st.plotly_chart(msg["fig"], use_container_width=True)
            if msg.get("code") and st.session_state.get("show_code"):
                with st.expander("Generated code"):
                    st.code(msg["code"], language="python")

    c_left, c_right = st.columns([4, 1])
    with c_right:
        show_code = st.checkbox("Show code", key="show_code")

    question = st.chat_input("Ask anything about your data...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            result = analyze(question, df, GROQ_KEY or None)
        result["role"] = "assistant"
        st.session_state.messages.append(result)
        st.rerun()

    if st.session_state.messages:
        if st.button("🗑 Clear chat"):
            st.session_state.messages = []
            st.rerun()


# ── Tab 4: Raw Data ──────────────────────────────────────────────────────────
with tabs[3]:
    if is_happiness:
        year_range = st.select_slider(
            "Filter by year", options=sorted(df["year"].unique()), value=(2015, 2022)
        )
        view = df[df["year"].between(*year_range)]
    else:
        view = df

    st.dataframe(view, use_container_width=True, height=500)
    st.download_button(
        "⬇️ Download CSV",
        view.to_csv(index=False),
        "data.csv", "text/csv",
    )
