# AI Data Analysis Agent

An interactive AI-powered data analysis app — upload any CSV and ask questions in plain English to get instant charts, trends, and insights.

Built with **Streamlit + Plotly + Groq (Llama 3.1 70B)** and includes 8 years of World Happiness Report data (2015–2022) as a built-in demo.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red) ![Groq](https://img.shields.io/badge/LLM-Llama_3.1_70B-green)

---

## Features

- **Upload any CSV** — instant auto-insights on load
- **Natural language chat** — ask questions, get charts + text answers
- **LLM mode** — powered by Llama 3.1 70B via Groq (free API key)
- **Rule-based fallback** — works offline with zero API key
- **World Happiness 2015–2022** — 8 years of data merged and normalized, with dedicated trend dashboard

### Auto-generated insights
- Correlation heatmap
- Distributions with box plots
- Category comparisons
- Missing value analysis
- Outlier detection

### World Happiness dashboard
- Global happiness trend over 8 years
- Top 10 happiest countries per year
- Happiness by region over time
- GDP per capita vs Happiness (with trendline)
- Most improved countries (2015 → 2022)
- Interactive country spotlight

---

## Quick Start

```bash
git clone https://github.com/dasha1127/AI-Data-Analysis.git
cd AI-Data-Analysis
pip install -r requirements.txt
streamlit run app.py
```

### Enable LLM mode (free)
1. Get a free API key at [console.groq.com](https://console.groq.com) — no credit card needed
2. Copy `.env.example` → `.env` and add your key:
   ```
   GROQ_API_KEY=gsk_...
   ```
3. Restart the app — chat is now powered by **Llama 3.1 70B**

### Load the World Happiness dataset
```bash
pip install kaggle
kaggle auth login
kaggle datasets download -d mathurinache/world-happiness-report --unzip -p ./data
```
Then in the app: sidebar → Sample Data → **World Happiness 2015–2022**

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Charts | Plotly Express |
| LLM | Groq API — Llama 3.1 70B |
| Data | Pandas, NumPy |
| ML utils | scikit-learn, scipy |

## Project Structure

```
├── app.py           # Streamlit UI — tabs, chat, trend dashboard
├── agent.py         # AI agent — intent detection, code generation, safe exec
├── data_loader.py   # Normalizes 8 years of World Happiness CSVs into one schema
├── requirements.txt
└── data/            # Place Kaggle CSVs here (gitignored)
```

## Key Concepts Demonstrated

- **Agentic AI pattern** — LLM generates Python code, agent executes it safely
- **RAG-style context** — dataframe schema passed as context to the LLM
- **Intent classification** — NLP-based routing without an LLM (offline fallback)
- **Data normalization** — merging 8 inconsistently-named CSVs into one schema
- **Safe code execution** — sandboxed `exec()` for AI-generated code

---

*Built as part of an AI/ML engineering portfolio project.*
