import streamlit as st
import time

# IMPORT YOUR COMPILED GRAPH HERE
# Change 'research_backend' to whatever you named your original python file
from reseach_backend import workflow

# ---------------------------------------------------------
# 1. Page Configuration & Custom CSS (The "Wow" Factor)
# ---------------------------------------------------------
st.set_page_config(page_title="AI Equity Analyst", page_icon="📈", layout="wide")

st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Neon Blue Header */
    h1 {
        color: #00d2ff;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 40px;
    }

    /* Custom Run Button */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 210, 255, 0.4);
    }

    /* Final Report Box */
    .report-box {
        background-color: #1E2530;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        border-top: 4px solid #00d2ff;
        color: #E0E0E0;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Application Header
# ---------------------------------------------------------
st.markdown("<h1>🤖 Autonomous AI Equity Analyst</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Powered by LangGraph Multi-Agent Orchestration & Real-Time Market Data</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. User Input Layout
# ---------------------------------------------------------
# Creating a clean 2-column layout for inputs
col1, col2 = st.columns([1, 2])

with col1:
    company = st.text_input("🏢 Company Name", placeholder="e.g., Tata Steel")
    
with col2:
    query = st.text_input("🎯 Research Objective", placeholder="e.g., Is this a safe long-term investment? Include ROE and Margins.")

# ---------------------------------------------------------
# 4. Execution Engine & Real-Time UI Updates
# ---------------------------------------------------------
if st.button("🚀 Execute Autonomous Research"):
    
    if not company or not query:
        st.warning("⚠️ Please provide both a company name and a research objective to begin.")
    else:
        # Initialize state for LangGraph
        initial_state = {
            "user_query": query,
            "company": company,
        }
        
        # Use a unique thread ID for the session
        config = {"configurable": {"thread_id": f"streamlit_session_{int(time.time())}"}}

        # The interactive status box (collapses when done)
        with st.status("Initializing AI Agent Pipeline...", expanded=True) as status:
            
            # Stream the LangGraph workflow directly to the UI
            for output in workflow.stream(initial_state, config=config):
                
                for node_name, state_update in output.items():
                    # Format node names to look professional (e.g., "financial_stats_agent" -> "Financial Stats Agent")
                    clean_name = node_name.replace("_", " ").title()
                    
                    if node_name == "review_agent":
                        decision = state_update['review_decision']
                        
                        # Show the Reviewer's Internal Monologue in an expander
                        with st.expander(f"🧠 Reviewer's Internal Monologue (Score: {decision['score']}/100)"):
                            st.write(decision['reasoning'])
                            
                        if not decision['approved']:
                            st.error(f"⚠️ **Review Failed!** Reason: {decision['feedback']}")
                            st.warning(f"🔄 Rerouting pipeline back to: **{decision['retry_agent'].replace('_', ' ').title()}**")
                        else:
                            st.success(f"✅ **{clean_name}**: Report Approved for Generation!")
                    
                    else:
                        st.write(f"✅ **{clean_name}** completed successfully.")

            # Close the status box
            status.update(label="Research Pipeline Complete!", state="complete", expanded=False)

        # ---------------------------------------------------------
        # 5. Render Final Report
        # ---------------------------------------------------------
        final_state = workflow.get_state(config).values
        final_report = final_state.get("final_report", None)

        if final_report:
            st.markdown("### 📊 Executive Investment Brief")
            # Wrap the markdown in our custom CSS div
            st.markdown(f"<div class='report-box'>{final_report}</div>", unsafe_allow_html=True)
        else:
            st.error("Failed to generate the final report. Please check the terminal for backend errors.")