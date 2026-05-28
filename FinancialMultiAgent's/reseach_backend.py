
from langgraph.graph import StateGraph, START, END 
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage , SystemMessage , BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import serpapi
import os
from pydantic import BaseModel , Field 


from langgraph.prebuilt import ToolNode, tools_condition 
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests
import yfinance as yf
from typing import Dict, Any , Optional , TypedDict , List , Literal


load_dotenv()
API_KEY = os.getenv("WEATHER_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


def serpapi_search(query: str, limit: int =40):
    """
    Use this tool to search latest news on Google News.
    Returns only important news titles and sources in clean text format.
    """

    client = serpapi.Client(api_key=os.getenv("SERPAI_API_KEY"))

    results = client.search({
        "documentation_path": "/google-news-api",
        "engine": "google_news",
        "q": query,
        "hl": "en",
        "gl": "us"
    })

    news_results = results.get("news_results", [])

    formatted_news = []

    for item in news_results[:limit]:

        if "stories" in item:

            for story in item["stories"]:

                title = story.get("title", "No title")
                source = story.get("source", {}).get("name", "Unknown")

                formatted_news.append(
                    f"- {title} ({source})"
                )
        else:

            title = item.get("title", "No title")
            source = item.get("source", {}).get("name", "Unknown")

            formatted_news.append(
                f"- {title} ({source})"
            )

    return "\n".join(formatted_news)


def get_stock_data(symbol: str) -> Dict[str, Any]:
    """
    Comprehensive stock research tool using yfinance
    """

    stock = yf.Ticker(symbol)

    try:
        info = stock.info

        # Financial statements
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow

        # Latest rows
        latest_financial_col = financials.columns[0]
        latest_balance_col = balance_sheet.columns[0]
        latest_cashflow_col = cashflow.columns[0]

        # Revenue
        revenue = financials.loc["Total Revenue", latest_financial_col] \
            if "Total Revenue" in financials.index else None

        # Net income
        net_income = financials.loc["Net Income", latest_financial_col] \
            if "Net Income" in financials.index else None

        # Debt
        total_debt = balance_sheet.loc["Total Debt", latest_balance_col] \
            if "Total Debt" in balance_sheet.index else None

        # Cash Flow
        operating_cashflow = cashflow.loc["Operating Cash Flow", latest_cashflow_col] \
            if "Operating Cash Flow" in cashflow.index else None

        # News Headlines
        news = []

        for article in stock.news[:5]:
            news.append({
                "title": article.get("title"),
                "publisher": article.get("publisher")
            })

        data = {
            "symbol": symbol,

            "company_name": info.get("longName"),

            "current_price": info.get("currentPrice"),

            "market_cap": info.get("marketCap"),

            "pe_ratio": info.get("trailingPE"),

            "eps": info.get("trailingEps"),

            "revenue_growth": info.get("revenueGrowth"),

            "sector": info.get("sector"),

            "industry": info.get("industry"),

            "total_revenue": revenue,

            "net_income": net_income,

            "debt": total_debt,

            "cash_flow": operating_cashflow,

            "recommendation": info.get("recommendationKey"),

            "business_summary": info.get("longBusinessSummary"),
            "news": news,
            "profit_margins": info.get("profitMargins"),
            "return_on_equity": info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
        }

        return data

    except Exception as e:
        return {
            "error": str(e)
        }

# %%


# %%

# -----------------------------------
# Graph State 
# -----------------------------------


class MarketResearch(TypedDict):
    latest_news: List[str]
    earnings: str
    important_announcements: List[str]


class FinancialMetrics(TypedDict):
    pe_ratio: float
    revenue_growth: str
    eps: float
    debt: str
    cash_flow: str
    explanation: str


class RiskAnalysis(TypedDict):
    risks: List[str]
    severity: str

class ReviewDecision(TypedDict):
    score: int
    approved: bool
    feedback: str
    retry_agent: str


class ResearchState(TypedDict):
    user_query: str
    company: str

    ticker_symbol : str
    optimized_search_query : str


    market_research: MarketResearch
    
    financial_metrics: FinancialMetrics

    risk_analysis: RiskAnalysis

    review_decision : ReviewDecision

    final_report: str

    retry_agent : int



# -----------------------------------
# Structured Output for LLM
# -----------------------------------

class DataExtractionOutput(BaseModel):
    ticker_symbol: str = Field(
        description="The official Yahoo Finance stock ticker symbol (e.g., TSLA for Tesla, AAPL for Apple). You must convert company names to exact tickers."
    )
    optimized_search_query: str = Field(
        description="A highly optimized Google News search query based on the user's prompt (e.g., 'Tesla Q3 earnings announcements and market news')."
    )


class MarketResearchOutput(BaseModel):


    latest_news: List[str] = Field(
        default=None,
        description="Latest important news headlines about the stock"
    )

    earnings: Optional[str] = Field(
        default=None,
        description="Recent earnings report summary"
    )

    important_announcements: List[str] = Field(
        default=None,
        description="Important company announcements, partnerships, acquisitions, or launches"
    )


# -----------------------------------
# Structured Output for LLM
# -----------------------------------

class FinancialMetricsOutput(BaseModel):

    pe_ratio: Optional[float] = Field(
        default=None,
        description="Price to Earnings ratio of the company"
    )

    revenue_growth: Optional[str] = Field(
        default=None,
        description="Year-over-year revenue growth of the company"
    )

    eps: Optional[float] = Field(
        default=None,
        description="Earnings per share of the company"
    )

    debt: Optional[str] = Field(
        default=None,
        description="Total debt or liabilities of the company"
    )

    cash_flow: Optional[str] = Field(
        default=None,
        description="Operating cash flow of the company"
    )

    explanation: str = Field(
        description="Short explanation of the company's financial condition"
    )



class RiskAnalysisOutput(BaseModel):

    risks: List[str] = Field(
        description="List of major financial or business risks"
    )

    severity: str = Field(
        description="Overall risk severity: Low, Medium, or High"
    )

    explanation : str = Field(
        description="Explain Everything to the user in simple term so that user can understand easily"
    )


class ReviewAgentOutput(BaseModel):
    reasoning: str = Field(
        description="Think out loud here first. Evaluate the provided data against the strict rules. Explain why it passes or fails BEFORE giving the final score."
    )

    score: int
    approved: bool
    feedback: str
    retry_agent: Literal[
        "market_research",
        "financial_metrics",
        "risk_analysis",
        "none"
    ]

extractor_llm = llm.with_structured_output(DataExtractionOutput)

market_researcher_llm = llm.with_structured_output(MarketResearchOutput)


financial_market_llm = llm.with_structured_output(FinancialMetricsOutput)

# %%
risk_analyzer_llm = llm.with_structured_output(RiskAnalysisOutput)

# %%
review_agent_llm = llm.with_structured_output(ReviewAgentOutput)


def dataPrepAgent(state: ResearchState):
    messages = [
        SystemMessage(
            content="""
            You are a data preparation agent. 
            The user wants to research a company. 
            Your job is to identify the exact stock ticker symbol and write a highly effective news search query for serpai search.
            """
        ),
        HumanMessage(content=f"User Query: {state['user_query']} \n Company: {state['company']}")
    ]
    
    response = extractor_llm.invoke(messages)
    
    return {
        "ticker_symbol": response.ticker_symbol,
        "optimized_search_query": response.optimized_search_query
    }

def marketReseachAgent(state : ResearchState):
        
        raw_news_headlines = serpapi_search(state["optimized_search_query"])
        messages = [

        SystemMessage(
            content="""
            You are a professional stock market research analyst.

            Your task is to:
            - analyze latest stock market news
            - identify recent earnings information
            - identify important company announcements
            - summarize only important and high-signal information

            Ignore:
            - advertisements
            - spam
            - unrelated news
            - unnecessary links

            Always return concise and factual information.
            """
        ),

        HumanMessage(
            content=f"""
            Research the following stock/company:

            User Query:
            {state['user_query']}

            Gather:
            1. Latest important news
            2. Recent earnings information
            3. Important announcements

            LATEST NEWS :
            {raw_news_headlines}

            """
        )
    ]
        
        response = market_researcher_llm.invoke(messages)

        return {
    "market_research": response.model_dump()
         }

    

def financialMarketingAgent(state : ResearchState):

    raw_financial_data = get_stock_data(state["ticker_symbol"])


    messages = [

        SystemMessage(
            content="""
            You are an expert financial analyst.

            Your job is to analyze the company's financial health u
            Make sure the analysis will be professinal like a senior is doing.

            Give:
            - accurate metrics
            - concise explanation
            - simple financial interpretation

            Ignore unnecessary details and focus only on important financial indicators.
            """
        ),

        HumanMessage(
            content=f"""
            Analyze the financial metrics of this stock/company:

            {state['company']} ({state['ticker_symbol']})

            Here the raw data of company if u see any error pls return error occured in data

            RAW_DATA :
            {raw_financial_data}
        
            """

            
        )
         ]
    
    response = financial_market_llm.invoke(messages)

    return {
    "financial_metrics": response.model_dump()
}
    

# %%

def riskAgent(state: ResearchState):
    messages = [
        
    SystemMessage(
        content="""
        You are a professional Risk Assessment Agent.

        Your job is to analyze:
        1. Market research data
        2. Financial metrics
        3. News and announcements

        Then identify:
        - Major business risks
        - Financial risks
        - Market risks
        - Operational risks

        Return:
        - risks (list of risks)
        - severity (LOW, MEDIUM, HIGH)
        - explanation

        Be concise but insightful.
        """
        ),

        HumanMessage(
            content=f"""
        Company: {state['company']}

        Latest News:
        {state['market_research']['latest_news']}

        Announcements:
        {state['market_research']['important_announcements']}

        Financial Metrics:
        - PE Ratio: {state['financial_metrics']['pe_ratio']}
        - Revenue Growth: {state['financial_metrics']['revenue_growth']}
        - EPS: {state['financial_metrics']['eps']}
        - Debt to Equity: {state['financial_metrics']['debt']}
        - Cash Flow: {state['financial_metrics']['cash_flow']}

        Analyze the risks for this company.
        """
            )
    ]
    response = risk_analyzer_llm.invoke(messages)

    return {
    "risk_analysis": response.model_dump()
    }

# %%
def reviewAgent(state: ResearchState):
    messages = [
        SystemMessage(
            content="""
            You are a senior investment research reviewer.

            Your job is to review the provided research. 
            
            CRITICAL RULES:
            1. ONLY evaluate based on the data provided. DO NOT penalize the report for missing "industry benchmarks," "competitor analysis," or "historical trends" if that data was not explicitly provided to you.
            2. It is strictly ACCEPTABLE for 'Important Announcements' to be empty if there is no recent news. Do not fail the review for this.
            3. Focus on logical consistency. If there are discrepancies in the data, instruct the relevant agent to explain the discrepancy, not magically fix it.
            4. THINK OUT LOUD FIRST in your 'reasoning' field before deciding if it is approved.
            
            Give:
            - reasoning (Think out loud first)
            - score (0-100)
            - approved (true/false)
            - feedback
            - retry_agent (market_research, financial_metrics, risk_analysis, or none)
            """
        ),
        HumanMessage(
            content=f"""
            Company: {state['company']}

            ========================
            MARKET RESEARCH
            ========================
            Latest News/Updates:
            {state['market_research'].get('latest_news', 'None')}

            Important Announcements:
            {state['market_research'].get('important_announcements', 'None')}

            ========================
            FINANCIAL METRICS
            ========================
            PE Ratio: {state['financial_metrics']['pe_ratio']}
            Revenue Growth: {state['financial_metrics']['revenue_growth']}
            EPS: {state['financial_metrics']['eps']}
            Debt: {state['financial_metrics']['debt']}
            Cash Flow: {state['financial_metrics']['cash_flow']}
            
            Additional Metrics: 
            {state['financial_metrics'].get('additional_metrics', 'None')}
            
            Corrections Made in this Attempt: 
            {state['financial_metrics'].get('corrections_made', 'None')}

            ========================
            RISK ANALYSIS
            ========================
            Risks: {state['risk_analysis']['risks']}
            Severity: {state['risk_analysis']['severity']}

            Now review the overall research quality using the strict rules.
            """
        )
    ]

    # Call the LLM (which is already bound to your new ReviewAgentOutput Pydantic schema)
    response = review_agent_llm.invoke(messages)

    # Grab the current loop count (defaulting to 0 on the first pass)
    current_count = state.get("loop_count", 0)

    # Return the decision AND the incremented counter for the circuit breaker
    return {
        "review_decision": response.model_dump(),
        "loop_count": current_count + 1
    }

# %%
def conditional_routing(state: ResearchState):
    current_loops = state.get("loop_count",0)
    
    # 2. If approved, go to report
    if state['review_decision']["approved"]:
        return "report_generator"
    
    # 3. CIRCUIT BREAKER: If we've looped twice, force it to generate the report anyway
    if current_loops >= 2:
        print("\n⚠️ Max retries reached. Forcing report generation with current data.")
        return "report_generator"
    
    # 4. Otherwise, route to the retry agent
    retry = state['review_decision']['retry_agent']
    if retry == "market_research":
        return "marketing_agent"
    elif retry == "financial_metrics":
        return "financial_stats_agent"
    elif retry == "risk_analysis":
        return "risk_analyzer_agent"

    return END




def reportGeneratorAgent(state: ResearchState):

    messages = [

        SystemMessage(
            content="""
        You are a professional financial report generation agent.

        Your task is to create a clean, professional, investor-style report
        based on:
        - Market Research
        - Financial Metrics
        - Risk Analysis
        - Review Feedback

        The report should contain:

        1. Executive Summary
        2. Market Research Overview
        3. Financial Analysis
        4. Risk Assessment
        5. Final Recommendation

        Formatting Rules:
        - Use proper headings
        - Keep explanations concise but insightful
        - Make the report look professional
        - Use bullet points where appropriate
        - Mention overall review score

        Do not invent data.
        Use only the provided information.
        """
                ),

                HumanMessage(
                    content=f"""
        Company: {state['company']}

        ====================================
        MARKET RESEARCH
        ====================================

        Latest News:
        {state['market_research']['latest_news']}

        Important Announcements:
        {state['market_research']['important_announcements']}


        ====================================
        FINANCIAL METRICS
        ====================================

        PE Ratio:
        {state['financial_metrics']['pe_ratio']}

        Revenue Growth:
        {state['financial_metrics']['revenue_growth']}

        EPS:
        {state['financial_metrics']['eps']}

        Debt:
        {state['financial_metrics']['debt']}

        Cash Flow:
        {state['financial_metrics']['cash_flow']}


        ====================================
        RISK ANALYSIS
        ====================================

        Risks:
        {state['risk_analysis']['risks']}

        Severity:
        {state['risk_analysis']['severity']}


        ====================================
        REVIEW AGENT
        ====================================

        Score:
        {state['review_decision']['score']}

        Feedback:
        {state['review_decision']['feedback']}

        Generate the final professional report.
        """
                )
    ]

    response = llm.invoke(messages)

    return {
        "final_report": response.content
    }



graph = StateGraph(ResearchState)
graph.add_node('data_prep_agent', dataPrepAgent)
graph.add_node('marketing_agent',marketReseachAgent)
graph.add_node('financial_stats_agent',financialMarketingAgent)
graph.add_node('risk_analyzer_agent',riskAgent)
graph.add_node('review_agent',reviewAgent)
graph.add_node('report_generator',reportGeneratorAgent)


graph.add_edge(START, "data_prep_agent")
graph.add_edge("data_prep_agent", "marketing_agent")
graph.add_edge("marketing_agent","financial_stats_agent")
graph.add_edge("financial_stats_agent","risk_analyzer_agent")
graph.add_edge("risk_analyzer_agent","review_agent")
graph.add_conditional_edges(
    "review_agent",
    conditional_routing,
    {
        "report_generator": "report_generator",
        "marketing_agent": "marketing_agent",
        "financial_stats_agent": "financial_stats_agent",
        "risk_analyzer_agent": "risk_analyzer_agent",
        END: END
    }
)

graph.add_edge("report_generator",END)


cheackPointer = InMemorySaver()

workflow = graph.compile(checkpointer=cheackPointer)


def run_research(user_query: str, company_name: str):
    """
    Executes the LangGraph research pipeline and prints updates in real-time.
    """
    # 1. Initialize the starting state
    initial_state = {
        "user_query": user_query,
        "company": company_name,
        # The rest of the state variables will be populated by the agents
    }

    # 2. We use a thread_id so the checkpointer can track this specific run
    config = {"configurable": {"thread_id": "research_run_001"}}

    print(f"\n🚀 Starting AI Research Pipeline for: {company_name}...\n")
    print("-" * 50)

    # 3. Stream the graph execution
    # .stream() yields the output of each node as it finishes
    for output in workflow.stream(initial_state, config=config):
        
        # 'output' is a dictionary where the key is the name of the node that just finished
        for node_name, state_update in output.items():
            print(f"✅ Finished: {node_name}")

            if node_name == "review_agent":
                decision = state_update['review_decision']
                
                print(f"\n🧠 Reviewer's Internal Monologue:\n{decision['reasoning']}\n")
                
                if not decision['approved']:
                    print(f"⚠️ Review Failed (Score: {decision['score']})! Reason: {decision['feedback']}")
                    print(f"🔄 Rerouting back to: {decision['retry_agent']}\n")
            

    print("-" * 50)
    
    # 4. Fetch the final state to get the generated report
    final_state = workflow.get_state(config).values
    
    print("\n📊 FINAL REPORT GENERATED:\n")
    print(final_state.get("final_report", "Error: No report generated."))


# ---------------------------------------------------------
# Terminal Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Welcome to the Multi-Agent Financial Research System")
    
    query = input("Enter what you want to know (e.g., 'How is the company doing?'): ")
    company = input("Enter the company name (e.g., 'Tesla' or 'Apple'): ")
    
    run_research(query, company)





