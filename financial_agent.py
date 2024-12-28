from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Define the Groq model configuration once
groq_model = Groq(
    api_key=groq_api_key,
    id="llama3-70b-8192"  # Updated to correct model ID
)

# Web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=groq_model,
    tools=[DuckDuckGo()],
    instructions=["Always include the sources"],
    show_tools_calls=True,
    markdown=True,
)

# Financial agent
financial_agent = Agent(
    name="Financial Agent",
    role="Analyze financial data and provide insights",
    model=groq_model,
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["Use tables to display the Data"],
    show_tools_calls=True,
    markdown=True,
)

# Multi agent
multi_agent = Agent(
    team=[web_search_agent, financial_agent],
    model=groq_model,
    instructions=["Always include the sources", "Use tables to display the Data"],
    show_tools_calls=True,
    markdown=True,
)

# Execute the query
multi_agent.print_response("Summarise the analyst recommendations and share the latest news about Apple", stream=True)