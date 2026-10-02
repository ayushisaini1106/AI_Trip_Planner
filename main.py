from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agent.agentic_workflow import GraphBuilder
from utils.save_to_document import save_document
from starlette.responses import JSONResponse
import os
import time
import datetime
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set specific origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str


def invoke_with_retry(react_app, messages: dict, max_retries: int = 4) -> dict:
    """Invoke the graph with automatic retry on rate-limit (429) errors."""
    import re
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return react_app.invoke(messages)
        except Exception as e:
            err_str = str(e)
            is_rate_limit = (
                "429" in err_str or
                "rate_limit_exceeded" in err_str or
                "Rate limit" in err_str or
                "Request too large" in err_str or
                "OTPM" in err_str or
                "TPM" in err_str or
                "output tokens per minute" in err_str or
                "tokens per minute" in err_str
            )
            if is_rate_limit:
                # OTPM errors need 60s wait (per-minute window reset)
                if "output tokens per minute" in err_str or "OTPM" in err_str or "Request too large" in err_str:
                    wait_seconds = 62.0
                else:
                    # Try to extract wait time from Groq error message
                    wait_seconds = 15.0 * attempt
                    match = re.search(r"try again in (\d+\.?\d*)s", err_str)
                    if match:
                        wait_seconds = float(match.group(1)) + 3
                print(f"[Rate limit] Attempt {attempt}/{max_retries}. Waiting {wait_seconds:.0f}s before retry...")
                time.sleep(wait_seconds)
                last_error = e
            else:
                raise e
    raise last_error


@app.post("/query")
async def query_travel_agent(query: QueryRequest):
    try:
        load_dotenv(override=True)
        print(query)
        graph = GraphBuilder(model_provider="groq")
        react_app = graph()

        try:
            png_graph = react_app.get_graph().draw_mermaid_png()
            with open("my_graph.png", "wb") as f:
                f.write(png_graph)
            print(f"Graph saved as 'my_graph.png' in {os.getcwd()}")
        except Exception as img_err:
            print(f"Could not render graph PNG: {img_err}")

        messages = {"messages": [query.question]}
        output = invoke_with_retry(react_app, messages, max_retries=3)

        # If result is dict with messages:
        if isinstance(output, dict) and "messages" in output:
            final_output = output["messages"][-1].content  # Last AI response
        else:
            final_output = str(output)

        return {"answer": final_output}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})