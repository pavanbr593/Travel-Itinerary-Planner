import os
from typing import List, TypedDict, Annotated
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_groq import ChatGroq
from IPython.display import display
import gradio as gr

class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "Messages in the conversation"]
    city: str
    interests: List[str]
    itinerary: str


llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile"
)


itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Create a day trip itinerary for {city} based on the user's interests: {interests}. Provide a brief, bulleted itinerary."),
    ("human", "Create an itinerary for my day trip."),
])



def input_city(city: str, state: PlannerState) -> PlannerState:
    return {
        **state,
        "city": city,
        "messages": state['messages'] + [HumanMessage(content=f"I want to plan a trip to {city}.")],
    }



def input_interests(interests: str, state: PlannerState) -> PlannerState:
    return {
        **state,
        "interests": interests.split(", "),
        "messages": state['messages'] + [HumanMessage(content=f"My interests are: {interests}.")],
    }



def create_itinerary(state: PlannerState) -> str:
    response = llm.invoke(itinerary_prompt.format_messages(
        city=state['city'],
        interests=", ".join(state['interests'])
    ))
    state["itinerary"] = response.content
    state["messages"] += [AIMessage(content=response.content)]
    return response.content



def travel_planner(city: str, interests: str) -> str:
    state: PlannerState = {
        "messages": [],
        "city": "",
        "interests": [],
        "itinerary": "",
    }

   
    state = input_city(city, state)
    state = input_interests(interests, state)

    
    itinerary = create_itinerary(state)

    return itinerary

interface = gr.Interface(
    fn=travel_planner,
    inputs=[
        gr.Textbox(label="Enter the city for your day trip"),
        gr.Textbox(label="Enter your interests (comma-separated)"),
    ],
    outputs=gr.Textbox(label="Generated Itinerary"),
    title="Travel Itinerary Planner",
    description="Enter a city and your interests to generate a personalized day trip itinerary."
)


interface.launch(share=False)
