import os
import json
from time import sleep
from typing import List, TypedDict, Annotated
from langgraph.graph import StateGraph  # For state graph representation
from langchain_core.messages import HumanMessage, AIMessage  # To handle conversation messages
from langchain_core.prompts import ChatPromptTemplate  # To format prompts for LLMs
from langchain_core.runnables.graph import MermaidDrawMethod  # Graph drawing method
from langchain_groq import ChatGroq  # To interface with the ChatGroq LLM
from IPython.display import display  # To display outputs in Jupyter/IPython environments
import gradio as gr  # For creating a user interface
import gradio.themes as grthemes
from gradio.themes.utils import colors, sizes, fonts

# Define the state of the planner
class PlannerState(TypedDict):
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

# Offline storage file
HISTORY_FILE = "itinerary_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# Function to input city into the state
def input_city(city: str, state: PlannerState) -> PlannerState:
    sleep(2)  # Adding a 2-second delay
    return {
        **state,
        "city": city,
        "messages": state['messages'] + [HumanMessage(content=f"I want to plan a trip to {city}.")],
    }

# Function to input interests into the state
def input_interests(interests: str, state: PlannerState) -> PlannerState:
    sleep(2)  # Adding a 2-second delay
    return {
        **state,
        "interests": interests.split(", "),
        "messages": state['messages'] + [HumanMessage(content=f"My interests are: {interests}.")],
    }

# Function to generate an itinerary using the LLM
def create_itinerary(state: PlannerState) -> str:
    response = llm.invoke(itinerary_prompt.format_messages(
        city=state['city'],
        interests=", ".join(state['interests'])
    ))
    state["itinerary"] = response.content
    state["messages"] += [AIMessage(content=response.content)]
    return response.content

# Wrapper function to integrate the planning steps
def travel_planner(city: str, interests: str):
    state: PlannerState = {
        "messages": [],
        "city": "",
        "interests": [],
        "itinerary": "",
    }

    state = input_city(city, state)
    state = input_interests(interests, state)

    itinerary = create_itinerary(state)

    # Save history
    history = load_history()
    history.append({"city": city, "interests": interests, "itinerary": itinerary})
    save_history(history)

    return itinerary

# Gradio Interface
def interface_fn(city: str, interests: str):
    itinerary = travel_planner(city, interests)
    return itinerary

def load_past_itineraries():
    history = load_history()
    if not history:
        return "No past itineraries found."
    return "\n\n".join([
        f"City: {entry['city']}\nInterests: {entry['interests']}\nItinerary:\n{entry['itinerary']}"
        for entry in history
    ])

def download_itinerary(itinerary):
    file_path = "itinerary.txt"
    with open(file_path, "w") as f:
        f.write(itinerary)
    return file_path

# Custom theme for the interface
custom_theme = grthemes.Base(
    primary_hue=colors.blue,
    secondary_hue=colors.cyan,
    neutral_hue=colors.gray,
    font=fonts.GoogleFont("Poppins"),
    font_mono=fonts.GoogleFont("Roboto Mono"),
    spacing_size=sizes.spacing_md,
    radius_size=sizes.radius_md,
    text_size=sizes.text_md,
)

# Gradio Interface with improved design
with gr.Blocks(theme=custom_theme, title="Travel Itinerary Planner") as main_interface:
    gr.Markdown("""
    # 🌍 Travel Itinerary Planner
    Create personalized day trip itineraries based on your interests and destination.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Trip Details")
            city_input = gr.Textbox(
                label="Enter your destination city",
                placeholder="e.g., Paris, Tokyo, New York",
                info="Where would you like to visit?"
            )
            interests_input = gr.Textbox(
                label="Your interests (comma-separated)",
                placeholder="e.g., museums, food, nature, shopping",
                info="What activities interest you?"
            )
            generate_button = gr.Button(
                "✨ Generate Itinerary",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Generated Itinerary")
            generated_itinerary = gr.Textbox(
                label="Your personalized itinerary",
                lines=10,
                interactive=False,
                show_copy_button=True
            )
            
            with gr.Row():
                copy_button = gr.Button(
                    "📋 Copy to Clipboard",
                    variant="secondary"
                )
                download_button = gr.Button(
                    "⬇️ Download Itinerary",
                    variant="secondary"
                )
    
    with gr.Accordion("📚 Past Itineraries", open=False):
        past_itineraries = gr.Textbox(
            label="Your travel history",
            lines=8,
            interactive=False,
            value=load_past_itineraries()
        )

    def handle_generate(city, interests):
        if not city or not interests:
            return "Please enter both a city and your interests."
        itinerary = travel_planner(city, interests)
        return itinerary

    def handle_copy(itinerary):
        import pyperclip
        pyperclip.copy(itinerary)
        return "✅ Itinerary copied to clipboard!"

    def handle_download(itinerary):
        file_path = "itinerary.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(itinerary)
        return f"✅ Itinerary saved to {file_path}"

    generate_button.click(
        fn=handle_generate,
        inputs=[city_input, interests_input],
        outputs=generated_itinerary
    )

    copy_button.click(
        fn=handle_copy,
        inputs=generated_itinerary,
        outputs=gr.Textbox(label="Copy Status", visible=False)
    )

    download_button.click(
        fn=handle_download,
        inputs=generated_itinerary,
        outputs=gr.Textbox(label="Download Status", visible=False)
    )

    # Add some CSS for better styling
    main_interface.css = """
    .gradio-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    .gradio-container h1 {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 30px;
    }
    .gradio-container h3 {
        color: #3498db;
        margin-bottom: 15px;
    }
    .gradio-container .textbox {
        border-radius: 8px;
        padding: 15px;
    }
    .gradio-container button {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }
    .gradio-container .accordion {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
    }
    """

main_interface.launch(share=False)
