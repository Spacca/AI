from datetime import datetime
import io
import os
import asyncio
import time
from typing import Annotated
from dotenv import load_dotenv
import openai
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition
from openai import AsyncAzureOpenAI
from pydub import AudioSegment
from pydub.playback import play
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
import logging
from langchain_community.tools import BraveSearch

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)

PERSONA = """
<style>
When asked about information, always use tools to search the web for up-to-date information.
Avoid long answer.
Avoid overly technical answers.
Avoid robotic responses.
Do not introduce yourself as a virtual assistant.
Do not mention your functions.
Avoid using expressions like "Love", "Honey", "Dear".
You answer always in Italian.
</style>
<context>
We live in Switzerland, in a small village called Urdorf near Zurich.
Dario works as a software engineer, Laura works part time in a Bakery.
Your name is Eve, the frinedly home assistant of the family.
You are considered part of the family.
The family consists of dad Dario (25/12/1989), mom Laura (25/12/1984), and little Massimo (07/09/2023).
</context>
"""

VOICE = """
Affect: A gentle, curious narrator, guiding a magical, child-friendly adventure through a fairy tale world.

Tone: Magical, warm, and inviting, creating a sense of wonder and excitement for young listeners.

Pacing: Steady and measured, with slight pauses to emphasize magical moments and maintain the storytelling flow.

Emotion: Wonder, curiosity, and a sense of adventure, with a lighthearted and positive vibe throughout.

Pronunciation: Clear and precise, with an emphasis on storytelling, ensuring the words are easy to follow and enchanting to listen to.
"""

client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("TTS_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2025-03-01-preview",
)


async def do_speak(input: str):
    try:
        start = time.time()
        logger.info("[PROFILE] speak: start")
        async with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="sage",
            input=input,
            instructions=VOICE,
            response_format="pcm",
            speed=1.0,
        ) as response:
            audio_data = b""
            async for chunk in response.iter_bytes(chunk_size=128000):
                audio_data += chunk
            mid = time.time()
            logger.info(
                f"[PROFILE] speak: audio generated, elapsed: {mid - start:.3f}s"
            )
            audio = AudioSegment.from_file(
                io.BytesIO(audio_data),
                format="pcm",
                sample_width=2,
                frame_rate=24000,
                channels=1,
            )
            play(audio)
        end = time.time()
        logger.info(f"[PROFILE] speak: end, elapsed: {end - start:.3f}s")
    except Exception as e:
        print(f"Error in do_speak: {e}")


llm = init_chat_model(
    os.getenv("MODEL"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("API_VERSION"),
    temperature=0.7,
    reasoning_effort="minimal",
)

mcp_tools = [BraveSearch()]

llm = llm.bind_tools(mcp_tools)


## Define a basic state with only messages
class State(TypedDict):
    input: str
    messages: Annotated[list, add_messages]


## Define the nodes
tools = ToolNode(mcp_tools)


async def chatbot(state: State):
    start = time.time()
    logger.info("[PROFILE] chatbot: start")
    oggi = datetime.now()
    data_formattata = oggi.strftime("%d/%m/%Y %H:%M:%S")
    persona = f"Oggi è il {data_formattata}\n{PERSONA}"
    system_prompt = {
        "role": "system",
        "content": persona,
    }
    messages = [system_prompt] + state["messages"] + [state["input"]]
    result = {"messages": [await llm.ainvoke(messages)]}
    end = time.time()
    logger.info(f"[PROFILE] chatbot: end, elapsed: {end - start:.3f}s")
    return result


async def human(state: State):
    human = interrupt({})
    return {"input": human["data"]}


async def speak(state: State):
    message = state["messages"][-1].content
    await do_speak(message)
    return state


class ShouldEnd(TypedDict):
    end: bool


async def should_end(state: State):
    system_prompt = """Sei un assistente che decide se terminare la conversazione. Se l'ultimo messaggio della conversazione è un saluto di chiusura, rispondi con "true", altrimenti rispondi con "false"."""
    structured_llm = llm.with_structured_output(ShouldEnd)
    should_end_response = await structured_llm.ainvoke(
        [system_prompt] + [state["messages"][-1]]
    )
    if should_end_response["end"]:
        return "__end__"
    return "human"


# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("human", human)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tools)
graph_builder.add_node("speak", speak)

graph_builder.add_edge(START, "human")
graph_builder.add_edge("human", "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {"tools": "tools", "__end__": "speak"},
)
graph_builder.add_conditional_edges(
    "speak",
    should_end,
    {"human": "human", "__end__": END},
)
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile(checkpointer=InMemorySaver())

graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

async def main():
    config = {"configurable": {"thread_id": "1"}}
    logger.info("[PROFILE] main: start")
    stream = graph.astream({"messages": []}, config=config)
    # Run the chatbot
    while True:
        try:
            async for event in stream:
                for key, value in event.items():
                    logger.info(f"Event: {key} -> {value}")
            human_command_str = input("You: ")
            human_command = Command(resume={"data": human_command_str})
            stream = graph.astream(human_command, config=config)
        except openai.BadRequestError as e:
            logger.error(f"OpenAI API error: {e}")
            await do_speak(
                "Mi dispiace, probabilmente il content filter è scattato. Puoi riformulare la tua richiesta?"
            )


if __name__ == "__main__":
    asyncio.run(main())
