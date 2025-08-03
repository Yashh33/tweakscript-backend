from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv
import tiktoken

# Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TransformRequest(BaseModel):
    prompt: str
    notes: str

class TagTransformRequest(BaseModel):
    category: str
    selected_text: str
    timestamp: str  # e.g. [03:43]

def estimate_tokens(text: str) -> int:
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(enc.encode(text))

def chunk_text(text: str, chunk_size: int = 6000) -> list:
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

@app.post("/transform")
async def transform_notes(req: TransformRequest):
    full_text = f"{req.prompt.strip()}\n\n{req.notes.strip()}"
    token_count = estimate_tokens(full_text)
    transformed_chunks = []

    if token_count > 30000:
        note_chunks = chunk_text(req.notes)
        for idx, chunk in enumerate(note_chunks):
            prompt_with_chunk = f"{req.prompt.strip()}\n\n{chunk}"
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt_with_chunk}],
                temperature=0.2,
                max_tokens=32768,
                top_p=1,
                stream=False,
            )
            transformed_chunks.append(f"Chunk {idx+1}:\n{completion.choices[0].message.content.strip()}")
    else:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": full_text}],
            temperature=0.2,
            max_tokens=32768,
            top_p=1,
            stream=False,
        )
        transformed_chunks.append(completion.choices[0].message.content.strip())

    return {"transformed_notes": "\n\n---\n\n".join(transformed_chunks)}

@app.post("/tag-transform")
async def tag_and_transform(req: TagTransformRequest):
    dynamic_prompt = (
    f"You are assisting with documenting key insights from a requirement gathering call. "
    f"The user has selected the following excerpt from the transcript and tagged it as <{req.category}>. "
    f"Your task is to extract the key point(s) conveyed in this selection and rewrite them in a clear, professional, third-person format — as if they are bullet points in structured meeting notes. "
    f"Do not use dialogue or speaker names. Summarize only the essential information, and attach the starting and ending timestamps the timestamp {req.timestamp}. "
    f"Keep the output concise and relevant to the <{req.category}> tag.\n\n"
    f"Transcript Excerpt:\n{req.selected_text}"
    )


    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": dynamic_prompt}],
        temperature=0.2,
        max_tokens=32768,
        top_p=1,
        stream=False,
    )

    return {"transformed_text": completion.choices[0].message.content.strip()}
