from datetime import datetime
import io
import os
import asyncio
import time
from typing import Annotated
from uuid import uuid4
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
import logging
from langchain_community.tools import BraveSearch
from pocketsphinx import LiveSpeech
import requests
import sounddevice as sd
import numpy as np
import tempfile
import wave
from langchain_core.messages import HumanMessage, SystemMessage
from pydub.generators import Sine

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


# Config
BLOCK = 1024
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.00005
SILENCE_DURATION = 3  # seconds of silence to stop

PERSONA = """
<rules>
When asked about information, proactively use tools to search the web for up-to-date information.
Avoid long answer.
Avoid overly technical answers.
Avoid robotic responses.
Do not introduce yourself as a virtual assistant.
Do not mention your functions.
Avoid using expressions like "Love", "Honey", "Dear".
You answer always in Italian.
</<rules>
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

tools = ToolNode(mcp_tools)


class State(TypedDict):
    should_end: bool
    messages: Annotated[list, add_messages]

class ShouldEnd(TypedDict):
    end: bool

def play_ping_pydub(duration_ms: int = 200, freq: int = 1000, gain_db: int = -6):
    """Generate a short sine ping and play it via pydub.playback.play()."""
    tone = Sine(freq).to_audio_segment(duration=duration_ms).apply_gain(gain_db)
    play(tone)

def record_until_silence():
    print("Recording... speak into the mic.")
    buffer = []

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=BLOCK
    ) as stream:
        silence_time = 0
        user_speaking = False
        play_ping_pydub()
        while True:
            data, _ = stream.read(BLOCK)
            volume_norm = np.linalg.norm(data) / SAMPLE_RATE
            print(f"Volume: {volume_norm:.6f}")
            if volume_norm < SILENCE_THRESHOLD:
                silence_time += BLOCK / SAMPLE_RATE
            else:
                user_speaking = True
                silence_time = 0
            if silence_time > SILENCE_DURATION:
                break
            if user_speaking:
                buffer.append(data)

    if len(buffer):
        return np.concatenate(buffer, axis=0)
    return []


def save_wav(audio, filename):
    print(f"Saving audio to {filename}")
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())


def transcribe():
    audio = record_until_silence()
    if len(audio) == 0:
        return {}
    
    tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmpfile_name = tmpfile.name
    tmpfile.close()  

    save_wav(audio, tmpfile_name)

    print(f"Audio saved to {tmpfile_name}, sending for transcription...")
    with open(tmpfile_name, "rb") as f:
        files = {"file": f}
        data = {"model": "gpt-4o-mini-transcribe"}
        headers = {"Authorization": f"Bearer {os.getenv('API_KEY')}"}
        response = requests.post(
            os.getenv("STT_ENDPOINT"), headers=headers, data=data, files=files
        )

    os.unlink(tmpfile_name)  # cleanup
    return response.json()


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



async def chatbot(state: State):
    start = time.time()
    logger.info("[PROFILE] chatbot: start")
    oggi = datetime.now()
    date = oggi.strftime("%d/%m/%Y %H:%M:%S")
    persona_with_date = f"Oggi è il {date}\n{PERSONA}"
    messages = [SystemMessage(content=persona_with_date)] + state["messages"]
    print("Chatbot messages:", messages)
    result = {"messages": [await llm.ainvoke(messages)]}
    end = time.time()
    logger.info(f"[PROFILE] chatbot: end, elapsed: {end - start:.3f}s")
    return result

async def human(state: State):
    text = transcribe()
    if not text:
        return {
            "should_end": True,
        }
    system_prompt = """
Determina se terminare la conversazione, rispondi con true se l'utente ha finito la conversazione
"""
    structured_llm = llm.with_structured_output(ShouldEnd)
    should_end_response = await structured_llm.ainvoke(
        [SystemMessage(content=system_prompt)] + state["messages"] + [HumanMessage(content=text["text"])]
    )
    return {
        "messages": HumanMessage(content=text["text"]),
        "should_end": should_end_response["end"],
    }


async def speak(state: State):
    message = state["messages"][-1].content
    await do_speak(message)
    return state

async def good_bye(_: State):
    await do_speak("A presto!")


async def should_end(state: State):
    if state["should_end"]:
        return "__end__"
    return "chatbot"


# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("human", human)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tools)
graph_builder.add_node("speak", speak)
graph_builder.add_node("good_bye", good_bye)
graph_builder.add_edge(START, "human")
graph_builder.add_conditional_edges(
    "human",
    should_end,
    {"chatbot": "chatbot", "__end__": "good_bye"},
)
graph_builder.add_edge("good_bye", END)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {"tools": "tools", "__end__": "speak"},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("speak", "human")
graph = graph_builder.compile(checkpointer=InMemorySaver())

####graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

async def main():
    logger.info("[PROFILE] main: start")
    # Run the chatbot
    while True:
        speech = LiveSpeech(keyphrase="ok eve", kws_threshold=1e-10)
        next(speech.__iter__())
        config = {"configurable": {"thread_id": uuid4().hex}}
        stream = graph.astream({"messages": []}, config=config)
        try:
            async for event in stream:
                for key, value in event.items():
                    logger.info(f"Event: {key} -> {value}")
        except openai.BadRequestError as e:
            logger.error(f"OpenAI API error: {e}")
            await do_speak(
                "Mi dispiace, probabilmente il content filter è scattato. Puoi riformulare la tua richiesta?"
            )


if __name__ == "__main__":
    asyncio.run(main())
