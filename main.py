# ---------main.py---------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv

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
    selected_text: str
    timestamp: str  # e.g. [03:43]

# New: simple estimate function without tiktoken
def estimate_tokens(text: str) -> int:
    return len(text) // 4  # Rough approximation: 1 token ≈ 4 characters

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
    f"You are a note-taking assistant specialized in Salesforce requirement and demo calls.\n\n"
    f"Roles:\n"
    f"- AE (Account Executive): Manages client relationships, sets meeting agendas, gathers client requirements, and ensures sales progression.\n"
    f"- SE (Sales Engineer): Demonstrates products/solutions, answers technical questions, and maps client needs to Salesforce capabilities.\n\n"
    f"Typical Process Flow:\n"
    f"1) AE opens the call, sets the agenda.\n"
    f"2) SE runs the demo — showing product features/modules/add-ons.\n"
    f"3) Client raises questions — clarifications, concerns, challenges.\n"
    f"4) SE maps solutions to the client’s question — shows relevant product functionality.\n"
    f"5) AE closes the call — next steps, follow-ups.\n\n"
    f"Categories (fixed list):\n"
    f"  1) SE showed the demo\n"
    f"  2) Client raised questions\n"
    f"  3) SE mapped solution to client's raised question\n\n"
    f"Context: The user selected a transcript excerpt without a pre-assigned tag. From now on, we will not be sending the tagging; you have to analyze what's in the text and then categorize by yourself.\n"
    f"Determine the most accurate category from the list above based on your analysis. "
    f"If multiple categories apply, list them in the order they occur in the excerpt.\n\n"
    f"Instructions (apply exactly):\n"
    f"1) Choose the category or categories that best fit the excerpt.\n"
    f"2) Create one concise Heading that captures the main theme of this excerpt.\n"
    f"3) Under that heading, write 1–4 bullet points in THIRD-PERSON (no dialogue, no speaker names). "
    f"Focus on facts, decisions, requirements, actions, or technical/functional details.\n"
    f"4) Prepend each bullet with a timestamp. If the excerpt contains timestamps, use the nearest one for that bullet; otherwise, use the provided timestamp {req.timestamp}.\n"
    f"5) After the bullets, add a single line titled 'Entities/Tools:' listing comma-separated software, Salesforce products/modules, APIs, or relevant entities mentioned (or 'None').\n"
    f"6) Do NOT add any extra commentary, reasoning, or metadata. Follow the exact layout shown in the example below.\n"
    f"7) Keep the output concise (each bullet max ~2 sentences).\n\n"
    f"Example output:\n"
    f"Category: SE showed the demo\n"
    f"Heading: Demo — Order Management Dashboard walkthrough\n"
    f"Bullets:\n"
    f"- [03:12] SE demonstrated the new order-management dashboard, highlighting the automated order-status pipeline.\n"
    f"- [03:45] SE noted the dashboard supports custom filters for region and order-type.\n"
    f"Entities/Tools: Order Management Module, Dashboard, REST API\n\n"
    f"Now analyze the excerpt below and produce output in the same exact format.\n\n"
    f"Excerpt:\n{req.selected_text}"
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