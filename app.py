import os
from dotenv import load_dotenv
import requests
from smolagents import CodeAgent, LiteLLMModel, Tool
from typing import Dict, Any

# Load environment variables
load_dotenv()

#Constants
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
MODEL_ID = "gemini/gemini-2.0-flash-lite"

# --- TOOL FUNCTIONS ---
def get_current_weather(city: str) -> str:
    """
    Gets the real-time current weather for a given city using the OpenWeatherMap API.
    """
    # --- NEW: Live API Call ---

    if not OPENWEATHER_API_KEY:
        return "Error: OpenWeatherMap API key is not set."
    elif not city:
        return "Error: City name is required."
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"  # Use 'imperial' for Fahrenheit
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        
        data = response.json()
        
        # Check if the API returned valid data
        if data.get("cod") != 200:
            return f"Error from weather API: {data.get('message', 'Unknown error')}"

        weather_description = data['weather'][0]['description']
        temperature = data['main']['temp']
        
        return f"The weather in {city} is currently {weather_description} with a temperature of {temperature}°C."

    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            return f"Sorry, I couldn't find weather data for {city}. Please check the city name."
        return f"An HTTP error occurred: {http_err}"
    except Exception as e:
        return f"An error occurred while fetching weather data: {e}"


def find_location(location_query: str) -> str:
    """Mock function to find a location based on a description."""
    locations = {
        "the capital of spain": "Madrid",
        "the capital of france": "Paris",
        "the city with the eiffel tower": "Paris"
    }
    return locations.get(location_query.lower(), f"Sorry, I couldn't find a city for '{location_query}'.")

def calculate(expression: str) -> str:
    """Mock function to evaluate a mathematical expression."""
    try:
        # A safer way to evaluate mathematical expressions
        result = eval(expression, {"__builtins__": None}, {})
        return f"The result of '{expression}' is {result}."
    except Exception as e:
        return f"Could not evaluate the expression. Error: {e}"

# --- TOOL CLASSES (WeatherTool is now powered by a live API) ---
class WeatherTool(Tool):
    """A tool to get the current weather for a specific city."""
    name: str = "get_current_weather"
    description: str = "Get the real-time current weather for a given city. Should be used after you know the specific city name."
    
    inputs: Dict[str, Dict[str, str]] = {
        "city": {
            "description": "The specific city name to get the weather for, e.g., 'Madrid'",
            "type": "string"
        }
    }
    output_type = "string"

    def forward(self, city: str) -> str:
        """Execute the weather check for the given city"""
        print(f"--- WeatherTool running: Getting LIVE weather for {city} ---")
        return get_current_weather(city)

class LocationTool(Tool):
    """A tool to find a specific city name from a general query."""
    name: str = "find_location"
    description: str = "Finds a specific city name based on a descriptive query (e.g., 'capital of France'). Use this first if the user doesn't provide a specific city name."
    inputs: Dict[str, Dict[str, str]] = {
        "location_query": {
            "description": "The general query for a location, e.g., 'the capital of Spain'",
            "type": "string"
        }
    }
    output_type = "string"

    def forward(self, location_query: str) -> str:
        print(f"--- LocationTool running: Finding location for '{location_query}' ---")
        return find_location(location_query)
        
class CalculatorTool(Tool):
    """A tool to perform basic arithmetic."""
    name: str = "calculate"
    description: str = "Evaluates a simple mathematical expression like '5 * 10' or '100 + 5'."
    inputs: Dict[str, Dict[str, str]] = {
        "expression": {
            "description": "The mathematical expression to evaluate, e.g., '5 * 123'",
            "type": "string"
        }
    }
    output_type = "string"

    def forward(self, expression: str) -> str:
        print(f"--- CalculatorTool running: Evaluating '{expression}' ---")
        return calculate(expression)

# --- AGENT CLASS ---
class MyAgent:
    """A simple agent that can use tools to answer questions."""
    def __init__(self):
        """Initializes the Agent with a model and multiple tools."""
        print("Initializing agent...")
        self.model = LiteLLMModel(
            model_id=MODEL_ID,
            api_key=GOOGLE_API_KEY
        )
        self.agent = CodeAgent(
            model=self.model,
            tools=[WeatherTool(), LocationTool(), CalculatorTool()]
        )
        print("Agent initialized successfully with 3 tools (including live weather).")
    
    def run(self, query: str):
        """Runs the agent with a given query."""
        print(f"\n--- Running agent with query: '{query}' ---")
        response = self.agent.run(query)
        return str(response)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Check for required API keys
    if not GOOGLE_API_KEY or not OPENWEATHER_API_KEY:
        print("Error: GOOGLE_API_KEY or OPENWEATHER_API_KEY not found.")
        print("Please make sure both are set in your .env file.")
    else:
        # Get the city from the user first
        city_query = input("Please enter a city name (e.g., London, Tokyo): ")

        if city_query:
            # Initialize the agent
            agent = MyAgent()
            
            # Construct the question for the agent using the user's input
            full_query = f"What's the weather in {city_query}?"
            
            # Run the agent with the constructed question
            response = agent.run(full_query)
            
            # Print the final response
            print("\n--- Final Response ---")
            print(response)
        else:
            print("No city entered. Exiting.")
        
        # Example 1: Query that now uses a live API
        #response = agent.run("What's the weather in London?")
        #print("\n--- Final Response ---")
        #print(response)
        
        # Example 2: Query that chains tools with a live API call at the end
        #response = agent.run("What's the weather in the city with the eiffel tower?")
        #print("\n--- Final Response ---")
        #print(response)