# Autonomous AI Equity Analyst 📈

**Live Application:** https://autonomous-equity-analyst-dur8wbrzizh5s6t6ikvqre.streamlit.app  

**Video Demo:** [Paste Your Video Link Here - Optional but recommended]

## Overview
An autonomous, multi-agent financial research pipeline built to analyze equities, retrieve real-time market data, and generate hallucination-free financial reports. 

Instead of relying on a single zero-shot LLM prompt, this system utilizes a **stateful LangGraph architecture**. It routes data through specialized AI agents (Data Preparation, Financial Analysis, Qualitative Analysis) and forces the output through a deterministic Reviewer Agent to catch missing metrics before final generation.

## 🧠 System Architecture

![LangGraph Workflow Diagram](workflow.png).

The pipeline is built on a directed acyclic graph (DAG) with the following nodes:
1. **Data Prep Agent:** Extracts and formats precise ticker symbols (including international exchange suffixes) to bypass scraping blocks.
2. **Financial Stats Agent:** Hooks into `yfinance` with custom session headers to pull live P/E ratios, market caps, and cash flow data.
3. **Qualitative Agent:** Utilizes SerpAPI to scrape recent news, sentiment, and macroeconomic headwinds.
4. **Reviewer Agent:** A strict quality-control node that evaluates the combined data against a Pydantic schema. If metrics are missing (e.g., `None` values), it triggers a self-correcting loop to fetch the data again.
5. **Report Generator:** Compiles the verified state into a structured Markdown equity report.

## 🛠 Tech Stack
* **Orchestration:** LangGraph, LangChain
* **LLM:** Google Gemini (via `langchain-google-genai`)
* **Data Retrieval:** `yfinance`, SerpAPI
* **Frontend:** Streamlit Community Cloud
* **Validation:** Pydantic

## 💻 Local Setup

If you want to run this multi-agent system locally:

1. **Clone the repository:**
```bash
   git clone [https://github.com/Ruddrayadav/autonomous-equity-analyst.git](https://github.com/Ruddrayadav/autonomous-equity-analyst.git)
   cd autonomous-equity-analyst
