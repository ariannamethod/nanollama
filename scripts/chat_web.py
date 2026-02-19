"""
Web UI server for nanollama chat.

Usage:
    python -m scripts.chat_web --model-tag=chat
"""

import os
import json
import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import torch

from nanollama.common import compute_init, autodetect_device_type, print0
from nanollama.checkpoint_manager import load_model
from nanollama.engine import Engine


app = FastAPI(title="nanollama Chat")

# Global state
model = None
tokenizer = None
engine = None
device = None


def parse_args():
    parser = argparse.ArgumentParser(description="nanollama chat web server")
    parser.add_argument("--model-tag", type=str, default="chat", help="Model to load")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    return parser.parse_args()


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve the chat UI."""
    ui_path = Path(__file__).parent.parent / "go" / "ui.html"
    with open(ui_path, 'r') as f:
        return f.read()


@app.post("/chat")
async def chat(request: dict):
    """Handle chat requests."""
    global model, tokenizer, engine, device
    
    try:
        messages = request.get("messages", [])
        temperature = request.get("temperature", 0.7)
        max_tokens = request.get("max_tokens", 256)
        top_k = request.get("top_k", 40)
        
        if not messages:
            return JSONResponse({"response": "Please send a message."})
        
        # Format conversation
        conversation = {"messages": messages}
        
        # Tokenize
        prompt_ids, _ = tokenizer.render_conversation(conversation)
        
        # Add assistant start token
        assistant_start = tokenizer.encode_special("<|assistant_start|>")
        if assistant_start:
            prompt_ids.append(assistant_start)
        
        # Generate
        generated_tokens = []
        for token_column, _ in engine.generate(
            prompt_ids,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        ):
            token = token_column[0]
            # Check for end tokens
            if token == tokenizer.encode_special("<|assistant_end|>"):
                break
            if token == tokenizer.encode_special("<|eot_id|>"):
                break
            generated_tokens.append(token)
        
        # Decode response
        response = tokenizer.decode(generated_tokens)
        
        return JSONResponse({"response": response})
    
    except Exception as e:
        print0(f"Error in chat: {e}")
        return JSONResponse({"response": f"Error: {str(e)}"}, status_code=500)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}


def main():
    global model, tokenizer, engine, device
    
    args = parse_args()
    
    # Initialize compute
    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)
    
    # Load model
    print0(f"Loading model: {args.model_tag}")
    try:
        model, tokenizer, meta = load_model(args.model_tag, device, phase="eval")
        engine = Engine(model, tokenizer)
        print0("Model loaded successfully!")
    except Exception as e:
        print0(f"Warning: Could not load model: {e}")
        print0("Server will start but chat will not work until model is loaded.")
    
    # Start server
    print0(f"\nStarting server at http://{args.host}:{args.port}")
    print0(f"Open in your browser to chat with nanollama!")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
