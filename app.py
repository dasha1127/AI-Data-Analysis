import os
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from agent import analyze, auto_insights
from data_loader import load_combined, most_improved

load_dotenv()

st.set_page_config(
    page_title="AI Data Analysis Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #4F8BF9; margin-bottom: 0; }
    .subtitle   { color: #888; font-size: 1rem; margin-top: 0; }
    .user-msg   { background: #2A2A3E; border-radius: 12px; padding: 12px 16px; margin: 8px 0; }
    .agent-msg  { background: #1A2A1A; border-left: 3px solid #4CAF50;
                  border-radius: 0 12px 12px 0; padding: 12px 16px; margin: 8px 0; }
    .tip-box    { background: #2A1A2E; border-left: 3px solid #9B59B6;
                  border-radius: 0 8px 8px 0; padding: 10px 14px; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Setup")

    groq_key = st.text_input(
        "Groq API Key (optional — enables LLM)",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        placeholder="gsk_...",
        help="Free at console.groq.com — no credit card needed",
    )
    st.caption("LLM mode: Llama 3.1 70B" if groq_key else "Rule-based mode active")

    st.divider()
    st.markdown("## 📂 Load Data")

    upload_tab, sample_tab = st.tabs(["Upload CSV", "Sample Data"])
    df = None
    dataset_name = ""

    with upload_tab:
        uploaded = st.file_uploader("Drop any CSV here", type=["csv", "xlsx"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
                dataset_name = uploaded.name
                st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} cols")
            except Exception as e:
                st.error(f"Error: {e}")

    with sample_tab:
        sample = st.selectbox("Pick a dataset", [
            "None",
            "World Happiness 2015–2022",
            "Tips (regression demo)",
            "Iris (classification demo)",
        ])
        if sample == "World Happiness 2015–2022":
            try:
                df = load_combined()
                dataset_name = "World Happiness Report (2015–2022)"
                st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} cols — 8 years!")
            except FileNotFoundError:
                st.error("Run: kaggle datasets download -d mathurinache/world-happiness-report --unzip -p ./data")
        elif sample == "Tips (regression demo)":
            df = px.data.tips()
            dataset_name = "Tips dataset"
            st.success(f"Loaded {df.shape[0]:,} rows")
        elif sample == "Iris (classification demo)":
            df = px.data.iris()
            dataset_name = "Iris dataset"
            st.success(f"Loaded {df.shape[0]:,} rows")

    if df is not None:
        st.divider()
        st.markdown("## 📋 Columns")
        for col in df.columns:
            dtype = str(df[col].dtype)
            icon = "🔢" if "int" in dtype or "float" in dtype else "📝"
            st.caption(f"{icon} `{col}`")

    st.divider()

    is_happiness = df is not None and "happiness_score" in df.columns
    if is_happiness:
        st.markdown("""
<div class='tip-box'>
<b>💡 Try asking:</b><br>
• Happiness trend over the years<br>
• Top 10 happiest countries in 2021<br>
• Which country improved the most?<br>
• Correlation between GDP and happiness<br>
• Compare happiness by region<br>
• Distribution of happiness scores<br>
• Outliers in happiness score
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class='tip-box'>
<b>💡 Try asking:</b><br>
• Show correlation heatmap<br>
• Distribution of &lt;column&gt;<br>
• Top 10 by &lt;column&gt;<br>
• Any missing values?<br>
• Compare &lt;col&gt; by group<br>
• Outliers in &lt;column&gt;
</div>""", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>🤖 AI Data Analysis Agent</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload any dataset → ask questions in plain English → get instant insights</div>", unsafe_allow_html=True)
st.markdown("")

if df is None:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown("### 1️⃣ Load Data\nUpload a CSV or pick **World Happiness 2015–2022** from the sidebar.")
    c2.markdown("### 2️⃣ Explore Insights\nAuto-generated charts appear instantly on load.")
    c3.markdown("### 3️⃣ Ask Questions\nChat in plain English — get charts + text answers.")
    st.stop()


# ── Stats bar ─────────────────────────────────────────────────────────────────
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
is_happiness = "happiness_score" in df.columns

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Columns", df.shape[1])
c3.metric("Numeric", len(num_cols))
c4.metric("Categorical", len(cat_cols))
if is_happiness:
    c5.metric("Years", df["year"].nunique())
else:
    c5.metric("Missing", f"{df.isnull().sum().sum():,}")

st.markdown("---")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📊 Auto-Insights", "🌍 Happiness Trends" if is_happiness else "📈 Trends", "💬 Chat with Data", "🗂 Raw Data"])


# ── Tab 1: Auto-Insights ──────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### Instant insights")
    with st.spinner("Generating..."):
        insights = auto_insights(df)

    if not insights:
        st.info("No automatic charts available. Try the Chat tab!")
    else:
        for i in range(0, len(insights), 2):
            cols = st.columns(2)
            for j, insight in enumerate(insights[i: i + 2]):
                with cols[j]:
                    st.plotly_chart(insight["fig"], use_container_width=True)

    st.markdown("#### Statistical Summary")
    st.dataframe(df.describe(include="all").T, use_container_width=True)


# ── Tab 2: Happiness Trends (or generic trends) ───────────────────────────────
with tabs[1]:
    if is_happiness:
        st.markdown("### World Happiness — 8 Years of Data")

        # ── Global avg happiness over time ────────────────────────────────────
        avg_by_year = df.groupby("year")["happiness_score"].mean().reset_index()
        fig1 = px.line(
            avg_by_year, x="year", y="happiness_score",
            title="Global Average Happiness Score (2015–2022)",
            markers=True, labels={"happiness_score": "Avg Happiness Score"},
        )
        fig1.update_traces(line_color="#4F8BF9", line_width=3)
        st.plotly_chart(fig1, use_container_width=True)

        col_a, col_b = st.columns(2)

        # ── Top 10 happiest (latest year) ─────────────────────────────────────
        with col_a:
            latest_year = df["year"].max()
            top10 = df[df["year"] == latest_year].nlargest(10, "happiness_score")
            fig2 = px.bar(
                top10, x="happiness_score", y="country", orientation="h",
                title=f"Top 10 Happiest Countries ({latest_year})",
                color="happiness_score", color_continuous_scale="Blues",
                labels={"happiness_score": "Score", "country": ""},
            )
            fig2.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # ── Happiness by region over time ─────────────────────────────────────
        with col_b:
            region_year = (
                df[df["region"].notna()]
                .groupby(["year", "region"])["happiness_score"]
                .mean()
                .reset_index()
            )
            fig3 = px.line(
                region_year, x="year", y="happiness_score", color="region",
                title="Happiness by Region Over Time",
                markers=True, labels={"happiness_score": "Avg Score"},
            )
            st.plotly_chart(fig3, use_container_width=True)

        # ── GDP vs Happiness scatter ───────────────────────────────────────────
        fig4 = px.scatter(
            df[df["year"] == latest_year].dropna(subset=["gdp_per_capita", "happiness_score"]),
            x="gdp_per_capita", y="happiness_score",
            color="region", hover_name="country", size="happiness_score",
            title=f"GDP per Capita vs Happiness ({latest_year})",
            labels={"gdp_per_capita": "GDP per Capita", "happiness_score": "Happiness Score"},
            trendline="ols",
        )
        st.plotly_chart(fig4, use_container_width=True)

        # ── Most improved countries ────────────────────────────────────────────
        col_c, col_d = st.columns(2)

        with col_c:
            improved = most_improved(df, top_n=10)
            fig5 = px.bar(
                improved, x="change", y="country", orientation="h",
                title="Most Improved Countries (First → Last Year)",
                color="change", color_continuous_scale="Greens",
                labels={"change": "Score Change", "country": ""},
            )
            fig5.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)

        # ── Country spotlight ──────────────────────────────────────────────────
        with col_d:
            countries = sorted(df["country"].unique())
            selected = st.multiselect(
                "Track specific countries over time",
                countries,
                default=["Finland", "United States", "India", "China"],
            )
            if selected:
                spotlight = df[df["country"].isin(selected)]
                fig6 = px.line(
                    spotlight, x="year", y="happiness_score", color="country",
                    markers=True, title="Country Happiness Trends",
                    labels={"happiness_score": "Score"},
                )
                st.plotly_chart(fig6, use_container_width=True)

    else:
        # Generic trend tab for non-happiness datasets
        if "year" in df.columns or any("date" in c.lower() or "year" in c.lower() for c in df.columns):
            time_col = "year" if "year" in df.columns else next(
                c for c in df.columns if "date" in c.lower() or "year" in c.lower()
            )
            num = num_cols[0] if num_cols else None
            if num:
                trend = df.groupby(time_col)[num].mean().reset_index()
                fig = px.line(trend, x=time_col, y=num, markers=True, title=f"{num} over time")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No time column detected. Use the Chat tab to ask trend questions.")


# ── Tab 3: Chat ───────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### Ask anything about your data")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'>👤 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            if msg.get("text"):
                st.markdown(f"<div class='agent-msg'>🤖 {msg['text']}</div>", unsafe_allow_html=True)
            if msg.get("fig"):
                st.plotly_chart(msg["fig"], use_container_width=True)
            if msg.get("code") and st.session_state.get("show_code"):
                with st.expander("Generated code"):
                    st.code(msg["code"], language="python")

    show_code = st.checkbox("Show generated code", value=False, key="show_code")

    question = st.chat_input("Ask a question about your data...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Analyzing..."):
            result = analyze(question, df, groq_key or None)
        result["role"] = "assistant"
        st.session_state.messages.append(result)
        st.rerun()

    if st.session_state.messages and st.button("🗑 Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ── Tab 4: Raw Data ───────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### Raw Data")

    if is_happiness:
        year_filter = st.select_slider("Filter by year", options=sorted(df["year"].unique()), value=(2015, 2022))
        filtered = df[df["year"].between(*year_filter)]
    else:
        filtered = df

    st.dataframe(filtered, use_container_width=True, height=500)
    st.download_button(
        "⬇️ Download as CSV",
        filtered.to_csv(index=False),
        "world_happiness_combined.csv" if is_happiness else "data.csv",
        "text/csv",
    )
