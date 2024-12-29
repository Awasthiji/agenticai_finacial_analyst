import streamlit as st
import openai
import phi.api
from phi.model.openai import OpenAIChat
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Financial Analyst Engine",
    page_icon="🤖",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Initialize API keys directly from environment variables
phi_api_key = os.getenv("PHI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Validate API keys
if not all([phi_api_key, openai_api_key, groq_api_key]):
    st.error("Missing required API keys. Please ensure the environment variables are set.")
    st.stop()

# Set API keys
phi.api = phi_api_key
openai.api_key = openai_api_key

# Initialize Groq model
@st.cache_resource
def init_groq_model():
    return Groq(
        api_key=groq_api_key,
        id="llama3-70b-8192"
    )

# Initialize agents
@st.cache_resource
def init_agents():
    groq_model = init_groq_model()
    
    web_search_agent = Agent(
        name="Web Search Agent",
        role="Search the web for information",
        model=groq_model,
        tools=[DuckDuckGo()],
        instructions=["Always include the sources"],
        show_tools_calls=True,
        markdown=True,
    )

    financial_agent = Agent(
        name="Financial Agent",
        role="Analyze financial data and provide insights",
        model=groq_model,
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, 
                           stock_fundamentals=True, company_news=True)],
        instructions=["Use tables to display the Data"],
        show_tools_calls=True,
        markdown=True,
    )

    multi_agent = Agent(
        team=[web_search_agent, financial_agent],
        model=groq_model,
        instructions=["Always include the sources", "Use tables to display the Data"],
        show_tools_calls=True,
        markdown=True,
    )
    
    return web_search_agent, financial_agent, multi_agent

def extract_content(response):
    """Extract clean content from agent response"""
    if hasattr(response, 'content'):
        return response.content
    elif isinstance(response, dict) and 'content' in response:
        return response['content']
    elif isinstance(response, str):
        return response
    return str(response)

# Main app
def main():
    st.title("Multi-Agent Financial Analyst Engine 🤖")
    
    # Initialize agents
    web_search_agent, financial_agent, multi_agent = init_agents()
    
    # Agent selection
    agent_type = st.radio(
        "Select Agent Type:",
        ["Web Search Agent", "Financial Agent", "Multi Agent"],
        horizontal=True
    )
    
    # Query input
    query = st.text_area("Enter your query:", height=100)
    
    # Create containers for displaying results
    response_container = st.container()
    
    if st.button("Submit", type="primary"):
        if not query:
            st.warning("Please enter a query.")
            return
            
        with st.spinner("Processing your request..."):
            try:
                # Select appropriate agent based on user choice
                if agent_type == "Web Search Agent":
                    response = web_search_agent.run(query)
                elif agent_type == "Financial Agent":
                    response = financial_agent.run(query)
                else:  # Multi Agent
                    response = multi_agent.run(query)
                
                # Extract and display clean content
                clean_response = extract_content(response)
                
                with response_container:
                    st.markdown("### Response:")
                    st.markdown(clean_response)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Add instructions/examples in the sidebar
    with st.sidebar:
        st.markdown("### Example Queries")
        if agent_type == "Web Search Agent":
            st.markdown("""
            - "What are the latest developments in AI?"
            - "What is the current situation in global markets?"
            """)
        elif agent_type == "Financial Agent":
            st.markdown("""
            - "Show me AAPL stock performance and analyst recommendations"
            - "Compare the fundamentals of MSFT and GOOGL"
            """)
        else:
            st.markdown("""
            - "What is the current stock price of NVIDIA and any recent news about AI chips?"
            - "How are recent tech layoffs affecting stock prices of major tech companies?"
            """)

if __name__ == "__main__":
    main()
