import os
import json
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver  # Needed for graph state
import uuid
# from google import drive

# --- Import your model classes ---
from intent_classify import IntentClassifier
from config import BERT_MODEL
from config import GEMMA_BASE_MODEL_ID, GEMMA_ADAPTER_PATH
from reply_generator import EmailGenerator

#
# --- (MOVED) Load models ONCE at startup for efficiency ---
#
print("--- LOADING MODELS (This happens once) ---")
classifier = IntentClassifier(model_path=BERT_MODEL)
print("--- INTENT CLASSIFIER LOADED ---")
generator = EmailGenerator(GEMMA_BASE_MODEL_ID, GEMMA_ADAPTER_PATH)
print("Gemma Email Generator loaded.")
print("--- MODELS LOADED ---")
#
# --- (END MOVED) ---
#


# --- 1. Define the State ---
class GraphState(TypedDict):
    sender_email: str
    original_subject: str
    email_content: str
    intent: str
    task_details: dict
    draft_email: str
    reply_subject: str  # New field for the reply subject


# --- 2. Define The Nodes ---

def classify_intent(state: GraphState) -> GraphState:
    print("---CLASSIFYING EMAIL INTENT---")
    email = state["email_content"]
    # Use the globally loaded classifier (This will work now)
    prediction = classifier.predict(email)
    intent = prediction['label'] # Get the label from the dictionary
    print(f"Intent found: {intent}")
    return {"intent": intent}


def handle_merger(state: GraphState) -> GraphState:
    # ... (no changes here) ...
    print("---HANDLING MERGER ANNOUNCEMENT---")
    api_status = "SUCCESS"
    ticket_id = "#T-MERGER-7391"
    details = {
        "status": api_status,
        "ticket_id": ticket_id,
        "message": "Verification complete. Ticket raised for Senior Manager approval."
    }
    return {"task_details": details}


def handle_sustainability(state: GraphState) -> GraphState:
    # ... (no changes here) ...
    print("---HANDLING SUSTAINABILITY INITIATIVE---")
    rag_response = "According to our Q3 report, carbon emissions were reduced by 15%, and our renewable energy portfolio grew to 45%."
    details = {"rag_summary": rag_response}
    return {"task_details": details}


def handle_fallback(state: GraphState) -> GraphState:
    # ... (no changes here) ...
    print("---HANDLING FALLBACK---")
    details = {"message": "Query has been forwarded to the appropriate department for handling."}
    return {"task_details": details}


def generate_response(state: GraphState) -> GraphState:
    # ... (no changes here) ...
    print("---GENERATING DRAFT EMAIL---")
    intent = state["intent"]
    details = state["task_details"]
    original_subject = state["original_subject"]

    # Use the globally loaded generator (This will work now)
    reply_body = generator.generate(intent=intent, details=json.dumps(details))
    
    reply_subject = f"Re: {original_subject}"
    
    print(f"Draft generated:\n{reply_body[:100]}...")
    return {"draft_email": reply_body, "reply_subject": reply_subject}


# --- 3. Build the Graph ---
# ... (no changes to this section) ...

def route_after_classification(state: GraphState) -> str:
    intent = state["intent"].strip() 
    if intent == "Merger Announcement":
        return "handle_merger"
    elif intent == "Sustainability Initiative":
        return "handle_sustainability"
    else:
        return "handle_fallback"

workflow = StateGraph(GraphState)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("handle_merger", handle_merger)
workflow.add_node("handle_sustainability", handle_sustainability)
workflow.add_node("handle_fallback", handle_fallback)
workflow.add_node("generate_response", generate_response)
workflow.set_entry_point("classify_intent")
workflow.add_conditional_edges(
    "classify_intent",
    route_after_classification,
    {
        "handle_merger": "handle_merger",
        "handle_sustainability": "handle_sustainability",
        "handle_fallback": "handle_fallback",
    },
)
workflow.add_edge("handle_merger", "generate_response")
workflow.add_edge("handle_sustainability", "generate_response")
workflow.add_edge("handle_fallback", "generate_response")
workflow.add_edge("generate_response", END)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

print("--- LANGGRAPH WORKFLOW COMPILED (FULLY AUTOMATED) ---")


# --- 4. Create a Helper Function ---
def run_workflow(email_content: str, sender_email: str, subject: str) -> dict:
    """
    Runs the full LangGraph workflow for a single email.
    Returns a dictionary with the final reply subject and body.
    (Models are already loaded globally)
    """

    #
    # --- (REMOVED) Model loading was moved to the top of the file ---
    #
    
    # Use a unique thread_id for each run to keep states separate
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "email_content": email_content,
        "sender_email": sender_email,
        "original_subject": subject,
    }
    
    # Run the graph from start to finish
    final_state = app.invoke(initial_state, config=config)
    
    # Return the generated reply
    return {
        "reply_subject": final_state.get("reply_subject", f"Re: {subject}"),
        "reply_body": final_state.get("draft_email", "Error: Could not generate a reply.")
    }