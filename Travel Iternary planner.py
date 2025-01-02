import os
from typing import List, TypedDict, Annotated
from langgraph.graph import StateGraph  # For state graph representation
from langchain_core.messages import HumanMessage, AIMessage  # To handle conversation messages
from langchain_core.prompts import ChatPromptTemplate  # To format prompts for LLMs
from langchain_core.runnables.graph import MermaidDrawMethod  # Graph drawing method
from langchain_groq import ChatGroq  # To interface with the ChatGroq LLM
from IPython.display import display  # To display outputs in Jupyter/IPython environments
import gradio as gr  # For creating a user interface

# Define the state of the planner
class PlannerState(TypedDict):
    """
    PlannerState holds the state of the travel planning process, including:
    - messages: Conversation messages exchanged with the LLM.
    - city: The city the user wants to visit.
    - interests: A list of user's interests.
    - itinerary: Generated itinerary string.
    """
    messages: Annotated[List[HumanMessage | AIMessage], "Messages in the conversation"]
    city: str
    interests: List[str]
    itinerary: str

# Initialize the LLM with the Groq API
llm = ChatGroq(
    temperature=0,
    groq_api_key="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",  # Replace with your valid API key
    model_name="llama-3.3-70b-versatile"
)

# Define the prompt template for generating the itinerary
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Create a day trip itinerary for {city} based on the user's interests: {interests}. Provide a brief, bulleted itinerary."),
    ("human", "Create an itinerary for my day trip."),
])

# Function to input city into the state
def input_city(city: str, state: PlannerState) -> PlannerState:
    """
    Updates the state with the user's chosen city.
    
    Args:
        city (str): The city for the day trip.
        state (PlannerState): The current planner state.
    
    Returns:
        PlannerState: Updated state.
    """
    return {
        **state,
        "city": city,
        "messages": state['messages'] + [HumanMessage(content=f"I want to plan a trip to {city}.")],
    }

# Function to input interests into the state
def input_interests(interests: str, state: PlannerState) -> PlannerState:
    """
    Updates the state with the user's interests.
    
    Args:
        interests (str): Comma-separated string of interests.
        state (PlannerState): The current planner state.
    
    Returns:
        PlannerState: Updated state.
    """
    return {
        **state,
        "interests": interests.split(", "),  # Split interests into a list
        "messages": state['messages'] + [HumanMessage(content=f"My interests are: {interests}.")],
    }

# Function to generate an itinerary using the LLM
def create_itinerary(state: PlannerState) -> str:
    """
    Generates an itinerary based on the user's city and interests using the LLM.
    
    Args:
        state (PlannerState): The current planner state.
    
    Returns:
        str: Generated itinerary.
    """
    response = llm.invoke(itinerary_prompt.format_messages(
        city=state['city'],
        interests=", ".join(state['interests'])
    ))
    state["itinerary"] = response.content
    state["messages"] += [AIMessage(content=response.content)]
    return response.content

# Wrapper function to integrate the planning steps
def travel_planner(city: str, interests: str) -> str:
    """
    Main function to plan a day trip.
    
    Args:
        city (str): City name for the trip.
        interests (str): Comma-separated string of interests.
    
    Returns:
        str: Generated itinerary.
    """
    state: PlannerState = {
        "messages": [],
        "city": "",
        "interests": [],
        "itinerary": "",
    }

    # Update state with user inputs
    state = input_city(city, state)
    state = input_interests(interests, state)

    # Generate the itinerary
    itinerary = create_itinerary(state)

    return itinerary

# Gradio Interface
interface = gr.Interface(
    fn=travel_planner,  # Main function
    inputs=[
        gr.Textbox(label="Enter the city for your day trip"),  # Input for city
        gr.Textbox(label="Enter your interests (comma-separated)"),  # Input for interests
    ],
    outputs=gr.Textbox(label="Generated Itinerary"),  # Output for the generated itinerary
    title="Travel Itinerary Planner",  # Title of the app
    description="Enter a city and your interests to generate a personalized day trip itinerary."
)

# Launch the Gradio interface
interface.launch(share=False)  # Set share=True to get a public URL

