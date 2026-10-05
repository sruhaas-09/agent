from dotenv import load_dotenv
from typing import Literal
import os
import requests
from tavily import TavilyClient
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage
from langchain.agents import create_agent
from query import retriver

load_dotenv()

@tool
def calculator(a: float, b: float,operation: Literal["add", "subtract", "multiply", "divide"]
) -> float:
    """
    Perform a basic arithmetic operation on two numbers.
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError("Unsupported operation. Use add, subtract, multiply, or divide.")



OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

@tool
def get_weather(city: str, country: str = "") -> str:
    """
    Get current weather information for a city, including
    temperature, feels-like temperature, humidity, and weather
    conditions. Use this tool whenever the user asks about
    weather, temperature, humidity, or current conditions
    for a specific city.
     
    Args:
        city: Name of the city.
        country: Optional country code, such as IN, US, or GB.
    """

    location = f"{city},{country}" if country else city

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": location,
        "appid": OPENWEATHER_API_KEY,
        "units":"metrics"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return f"Unable to get weather data: {response.text}"

    data = response.json()

    temperature = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    description = data["weather"][0]["description"]

    return (
        f"Weather in {location}: "
        f"{description}, "
        f"temperature {temperature}°C, "
        f"feels like {feels_like}°C, "
        f"humidity {humidity}%."
    )

@tool
def currency_converter(
    amount: float,
    from_currency: str,
    to_currency: str
) -> str:
    """
    Convert an amount from one currency to another using the latest
    available exchange rate.

    Args:
        amount: Amount of money to convert.
        from_currency: Three-letter source currency code, e.g. USD.
        to_currency: Three-letter target currency code, e.g. INR.
    """

    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    url = f"https://api.frankfurter.app/latest"

    params = {
        "amount": amount,
        "from": from_currency,
        "to": to_currency
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return f"Unable to convert currency: {response.text}"

    data = response.json()

    converted_amount = data["rates"][to_currency]

    return (
        f"{amount} {from_currency} = "
        f"{converted_amount} {to_currency}"
    )



tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

@tool
def web_search(query: str) -> str:
    """
    Search the internet for current, recent, or up-to-date information.

    Use this tool for:
    - latest news
    - events that happened recently
    - information from the last few hours or days
    - current affairs
    - recent developments
    - current facts that may have changed
    - news about a country, person, company, technology, etc.

    Do NOT use this tool for mathematical calculations.
    Do NOT use this tool for weather when a dedicated weather tool is available.

    Args:
        query: The search query to search on the web.
    """

    response = tavily_client.search(
        query=query,
        max_results=3
    )

    results = response.get("results", [])
    # "Look inside response for the key results. If it exists, give me its value. If it doesn't exist, give me an empty list."

    if not results:
        return "No search results found."

    output = []

    for result in results:
        output.append(
            f"Title: {result.get('title')}\n"
            f"Content: {result.get('content')}\n"
            f"URL: {result.get('url')}"
        )

    return "\n\n".join(output)


@tool
def retrieve_from_rag(query: str) -> str:

    """
    Answer questions related to Retrieval-Augmented Generation (RAG)
    using the RAG research paper stored in the internal knowledge base.

    Use this tool when the user asks about concepts, architecture,
    methods, techniques, components, or other information specifically
    discussed in the RAG research paper.

    This tool can be used for questions about topics such as:
    - What is RAG?
    - RAG architecture and components
    - Retrieval methods
    - Document retrieval and knowledge augmentation
    - RAG generation process
    - Different RAG approaches or techniques
    - Advantages and limitations of RAG
    - Comparisons or findings discussed in the paper
    - Any other question whose answer should be derived from the
      contents of the RAG research paper.

    Do not use this tool for general questions unrelated to the
    RAG research paper.

    Args:
        query: The user's question about RAG or information discussed
               in the RAG research paper.
    """
    
    docs = retriver(query)
    if not docs:
        return "No relevant information was found in the knowledge base."

    return docs

tools=[calculator,get_weather,currency_converter,web_search,retrieve_from_rag]


#model intialization
llm=ChatOllama(
    model="llama3.2",
    temperature=0
)

#agent creation
agent=create_agent(model=llm,
                   tools=tools)


result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What problem does Retrieval-Augmented Generation aim to address?"
        }
    ]
})

print(result)





