import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import tiktoken

from model import GPT, GPTConfig

app = FastAPI()

# allow browser frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = tiktoken.get_encoding("gpt2")

# MUST match your training notebook
config = GPTConfig(
    block_size=128,
    vocab_size=50257,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True,
)

model = GPT(config).to(device)
state = torch.load("best_model_params.pt", map_location=device)
model.load_state_dict(state)
model.eval()


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 150
    temperature: float = 0.8
    top_k: int | None = 200


def sample_next_token(logits, temperature=1.0, top_k=None):
    logits = logits / max(temperature, 1e-6)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/generate_stream")
def generate_stream(req: GenerateRequest):
    prompt_ids = enc.encode(req.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    async def event_gen():
        nonlocal idx
        with torch.no_grad():
            for _ in range(req.max_new_tokens):
                idx_cond = idx[:, -config.block_size :]
                logits, _ = model(idx_cond)
                next_id = sample_next_token(
                    logits[:, -1, :],
                    temperature=req.temperature,
                    top_k=req.top_k,
                )
                idx = torch.cat([idx, next_id], dim=1)

                token_text = enc.decode([int(next_id.item())])
                yield {"event": "token", "data": token_text}

        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_gen())
