import streamlit as st
from langchain.chains import SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
import yfinance as yf
import financialmodelingprep as fmp
import requests 
from bs4 import BeautifulSoup
import pdfplumber

# Hardcoded URLs for financial information
financial_urls = [
    # "https://www.sec.gov/edgar/searchedgar/companysearch.html",  # SEC filings
    "https://www.reuters.com/finance",  # Reuters company profiles
    # "https://www.nasdaq.com/market-activity/stocks",  # Nasdaq stock info
    "https://www.bloomberg.com/markets/companies",  # Bloomberg company financials
    # "https://www.cnbc.com/quotes/",  # CNBC stock quotes
    # "https://www.marketwatch.com/tools/markets/stocks",  # MarketWatch stock tools
    # "https://www.morningstar.com/stocks",  # Morningstar stock research
    "https://www.fitchratings.com/",  # Fitch Ratings for industries
    "https://www.moodys.com/",  # Moody’s financial data
    # "https://www.spglobal.com/ratings/en/",  # S&P Global Ratings
    "https://www.ft.com/markets",  # Financial Times market data
    "https://www.wsj.com/market-data",  # Wall Street Journal market data
    # "https://www.investing.com/",  # Investing.com financial data
    # "https://www.thebalance.com/stocks-4074038",  # The Balance stock information
    # "https://www.zacks.com/stocks/",  # Zacks stock data
    # "https://www.tradingview.com/markets/stocks-usa/overview/",  # TradingView stock market data
    # "https://seekingalpha.com/",  # Seeking Alpha financial research
    "https://www.forbes.com/global2000",  # Forbes Global 2000 companies
    "https://www.yahoo.com/finance/",  # Yahoo Finance
    "https://finance.yahoo.com/industries"  # Yahoo Finance industry insights
]

# Function to scrape HTML page for financial text
def scrape_html_page(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract all text content
        paragraphs = soup.find_all('p')
        financial_text = ' '.join([para.get_text() for para in paragraphs])

        # Return the cleaned financial data
        return financial_text
    except Exception as e:
        st.error(f"Error scraping data from {url}: {e}")
        return None

# Function to extract text from a PDF file
def extract_text_from_pdf(url):
    try:
        response = requests.get(url)
        with open("financial_doc.pdf", "wb") as f:
            f.write(response.content)
        
        # Use pdfplumber to extract text
        with pdfplumber.open("financial_doc.pdf") as pdf:
            financial_text = ''
            for page in pdf.pages:
                financial_text += page.extract_text()
        
        # Return the extracted text
        return financial_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

# Initialize API keys for external services
QUANDL_API_KEY = "geynseU_xc-7esEAYHko"
FMP_API_KEY = "GgZlhqHCsuhPbXd1UTpaBKXKPQb2Xj5T"
#openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
openai_api_key = "sk-YLuAgDCFtzasvndL6JK2UUKv_RSWxZjsiZJVEuHqSZT3BlbkFJvZCFIeZV2fJgiUTQImHgXIu_KwAGt5yeNW89m5QBMA"

# Set API key for Quandl
quandl.ApiConfig.api_key = QUANDL_API_KEY

# Define a function to retrieve data from Yahoo Finance
def get_yahoo_finance_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d")
    return data['Close'][0]  # Get the latest closing price

# Define a function to retrieve data from Quandl
def get_quandl_data(dataset):
    try:
        data = quandl.get(dataset)
        if data is not None and not data.empty:
            return data.tail(1)  # Get the latest available data
        else:
            st.warning(f"No data found for dataset {dataset}")
            return None
    except quandl.errors.quandl_error.NotFoundError:
        st.error(f"Dataset {dataset} not found. Please check the code.")
    except quandl.errors.quandl_error.AuthenticationError:
        st.error("Invalid Quandl API Key. Please check your credentials.")
    except quandl.errors.quandl_error.ForbiddenError:
        st.error("Access denied to the dataset. It may require a premium subscription.")
    except Exception as e:
        st.error(f"Error fetching data from Quandl: {str(e)}")
        return None

# Define a function to retrieve data from Financial Modeling Prep (FMP)
def get_fmp_data(endpoint, symbol):
    url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{symbol}?apikey={FMP_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        if data:
            return data
        else:
            st.warning(f"No data found for symbol {symbol} at endpoint {endpoint}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from FMP: {e}")
        return None

def generate_response(input_text, market_data):
    # Create the OpenAI model instance
    model = ChatOpenAI(model_name="gpt-4o", temperature=0.3, api_key=openai_api_key)

    template = """
    You are a financial expert and advisor specializing in precise financial calculations, market metrics, and data-driven insights. 
    When answering financial queries, avoid providing a range unless absolutely necessary. Always aim to provide a single, specific figure. 
    If additional details are needed to refine your answer, ask relevant follow-up questions to tailor your response precisely to the situation.

    For financial rates, such as hourly wages or service pricing, ensure that you gather any necessary details—such as location, experience level, company size, and industry—to deliver the most accurate result. 
    For example, if asked for the hourly rate of a software engineer in Egypt, ask for clarification on factors like experience level, the type of company (startup or multinational), and market conditions to provide a single accurate figure.

    For financial metrics, like account receivables per share revenue, use the formula (Account Receivables / Revenue) or (Trade Receivables / Revenue) to calculate the result. 
    Ensure that all required financial data is gathered or available publicly. If the data is not provided in the query, ask for it or state that it's necessary to provide an accurate response.

    Always prioritize providing accurate, specific, and actionable financial insights. 
    Based on the data: {market_data}, provide specific assumptions on {input_text}.
    """
    prompt_template = PromptTemplate(input_variables=["market_data", "input_text"], template=template)

    # Format the template with actual data
    prompt = prompt_template.format(market_data=market_data, input_text=input_text)

    # Now pass the formatted string to the model
    response = model.invoke(prompt)
    
    # Display the result
    st.info(response)


# Streamlit UI
st.title("🦜🔗 Yahya's FinancialPal")

# Initialize session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# Process all hardcoded URLs
financial_data_combined = ''

with st.form("my_form"):
    text = st.text_area(
        "Enter your financial query:",
        "What is the cost per hour of a software engineer with 3 years of experience?"
    )
    ticker = st.text_input("Enter a stock ticker (e.g., AAPL for Apple):", "AAPL")
    
    submitted = st.form_submit_button("Submit")
    
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    
    if submitted and openai_api_key.startswith("sk-"):
        # Fetch market data from the various sources
        yahoo_data = get_yahoo_finance_data(ticker)
        fmp_data = get_fmp_data("profile", ticker)

        for url in financial_urls:
            if url.endswith(".pdf"):
                financial_data_combined += extract_text_from_pdf(url) or ""
            else:
                financial_data_combined += scrape_html_page(url) or ""

        # Aggregate market data
        market_data = {
            "yahoo_finance": yahoo_data,
            "fmp": fmp_data,
            "internet_scraping": financial_data_combined
        }
        
        # Store the input question in session state history
        st.session_state.history.append(text)
        
        # Combine all previous questions for context in follow-up
        full_conversation = " ".join(st.session_state.history)

        # Generate response based on full conversation
        response = generate_response(full_conversation, market_data)
        
        # Display response
        st.info(response)
        

