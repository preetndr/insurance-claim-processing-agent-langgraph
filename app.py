from typing import TypedDict
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

st.set_page_config(page_title="Q2: Insurance Claim Processing Agent", layout="wide")
st.title("🛡️ Insurance Claim Processing Agent (LangGraph)")

# ---------------------------------------------------------
# User Google API Key Section + Direct Link to Google AI Studio
# ---------------------------------------------------------
st.subheader("🔑 Google Gemini API Key")
col_key, col_link = st.columns([2, 1])
with col_key:
    user_api_key = st.text_input(
        "Enter your Google Gemini API Key to use the app:",
        type="password",
        placeholder="Paste your Gemini API key here..."
    )
with col_link:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "👉 **[Get a free Gemini API Key from Google AI Studio](https://aistudio.google.com/app/apikey)**"
    )

if not user_api_key:
    st.info("Please enter your Google Gemini API Key above to start. Click the link next to the box to get one from Google AI Studio.")
    st.stop()

# Initialize Gemini 3.5 Flash Lite with the user's API key
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=user_api_key, temperature=0)

def extract_clean_text(response) -> str:
    """Extract only the plain text from the Gemini response, stripping all metadata/extras."""
    content = response.content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return str(content).strip()

# ---------------------------------------------------------
# LangGraph State & 5 Agents/Nodes
# ---------------------------------------------------------
class ClaimState(TypedDict, total=False):
    claim_id: str
    policy_status: str
    claim_amount: float
    documents_complete: bool
    suspicious_activity: bool
    doc_check: str
    eligibility_check: str
    fraud_check: str
    claim_summary: str
    decision: str
    human_review: str

# Node 1: Document Verification Agent
def document_verification_agent(state: ClaimState):
    return {"doc_check": "Verified" if state["documents_complete"] else "Missing Documents"}

# Node 2: Eligibility Check Agent
def eligibility_check_agent(state: ClaimState):
    return {"eligibility_check": "Eligible" if state["policy_status"] == "Active" else "Expired Policy"}

# Node 3: Fraud Detection Agent
def fraud_detection_agent(state: ClaimState):
    if state["suspicious_activity"] or state["claim_amount"] > 25000:
        return {"fraud_check": "High Risk / High Value"}
    return {"fraud_check": "Low Risk"}

# Node 4: Claim Summary Agent
def claim_summary_agent(state: ClaimState):
    if state["doc_check"] != "Verified":
        decision = "Reject"
    elif state["eligibility_check"] != "Eligible":
        decision = "Reject"
    elif state["fraud_check"] != "Low Risk":
        decision = "Human Approval"
    else:
        decision = "Auto Approve"

    prompt = (
        f"Write a short 2-line insurance claim summary for Claim {state['claim_id']} "
        f"(Amount: ${state['claim_amount']}). "
        f"Docs: {state['doc_check']}, Eligibility: {state['eligibility_check']}, "
        f"Fraud Check: {state['fraud_check']}. Final Decision: {decision}."
    )
    response = llm.invoke(prompt)
    summary = extract_clean_text(response)
    return {"claim_summary": summary, "decision": decision}

# Node 5: Human Approval Agent
def human_approval_agent(state: ClaimState):
    return {"human_review": "Escalated to Human Adjuster for manual review and approval."}

# Conditional Edge Router
def route_claim(state: ClaimState):
    if state["decision"] == "Human Approval":
        return "human_approval_agent"
    return END

# Build LangGraph Workflow
builder = StateGraph(ClaimState)
builder.add_node("document_verification_agent", document_verification_agent)
builder.add_node("eligibility_check_agent", eligibility_check_agent)
builder.add_node("fraud_detection_agent", fraud_detection_agent)
builder.add_node("claim_summary_agent", claim_summary_agent)
builder.add_node("human_approval_agent", human_approval_agent)

# Parallel execution of verification, eligibility, and fraud checks
builder.add_edge(START, "document_verification_agent")
builder.add_edge(START, "eligibility_check_agent")
builder.add_edge(START, "fraud_detection_agent")

builder.add_edge("document_verification_agent", "claim_summary_agent")
builder.add_edge("eligibility_check_agent", "claim_summary_agent")
builder.add_edge("fraud_detection_agent", "claim_summary_agent")

builder.add_conditional_edges("claim_summary_agent", route_claim)
builder.add_edge("human_approval_agent", END)

claim_graph = builder.compile()

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
scenario = st.selectbox(
    "Select a Claim Scenario:",
    [
        "1. Complete claim with valid documents -> Auto Approve",
        "2. Missing required documents -> Reject",
        "3. Suspicious claim amount -> Human Approval",
        "4. Expired policy -> Reject",
        "5. High-value claim with valid documents -> Human Approval"
    ]
)

defaults = {
    "1. Complete claim with valid documents -> Auto Approve": ("CLM-101", "Active", 4000.0, True, False),
    "2. Missing required documents -> Reject": ("CLM-102", "Active", 5000.0, False, False),
    "3. Suspicious claim amount -> Human Approval": ("CLM-103", "Active", 15000.0, True, True),
    "4. Expired policy -> Reject": ("CLM-104", "Expired", 3000.0, True, False),
    "5. High-value claim with valid documents -> Human Approval": ("CLM-105", "Active", 35000.0, True, False),
}[scenario]

claim_id = st.text_input("Claim ID", value=defaults[0])
policy_status = st.selectbox("Policy Status", ["Active", "Expired"], index=0 if defaults[1] == "Active" else 1)
claim_amount = st.number_input("Claim Amount ($)", value=defaults[2])
documents_complete = st.checkbox("All Required Documents Verified?", value=defaults[3])
suspicious_activity = st.checkbox("Suspicious Fraud Indicators Detected?", value=defaults[4])

if st.button("Process Claim"):
    input_state = {
        "claim_id": claim_id,
        "policy_status": policy_status,
        "claim_amount": claim_amount,
        "documents_complete": documents_complete,
        "suspicious_activity": suspicious_activity
    }
    output = claim_graph.invoke(input_state)
    st.session_state["claim_output"] = output

if "claim_output" in st.session_state:
    out = st.session_state["claim_output"]
    st.subheader(f"Final Decision: {out['decision']}")
    st.write(f"**Document Verification:** {out['doc_check']}")
    st.write(f"**Policy Eligibility:** {out['eligibility_check']}")
    st.write(f"**Fraud Check:** {out['fraud_check']}")
    st.write(f"**Claim Summary:** {out['claim_summary']}")

    if out["decision"] == "Human Approval":
        st.warning(out.get("human_review", "Escalated for Human-in-the-Loop review."))
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ Approve Claim (Human Reviewer)"):
                st.success(f"Claim {out['claim_id']} manually APPROVED by Human Adjuster.")
        with col_b:
            if st.button("❌ Reject Claim (Human Reviewer)"):
                st.error(f"Claim {out['claim_id']} manually REJECTED by Human Adjuster.")