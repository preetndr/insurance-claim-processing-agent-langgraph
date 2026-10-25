# 🛡️ Insurance Claim Processing Agent (LangGraph Multi-Agent Workflow)

An automated, multi-agent insurance claim workflow system built using **LangGraph**, **Google Gemini (`gemini-3.5-flash-lite`)**, and **Streamlit**. It automates claim processing by executing document verification, policy eligibility checks, and fraud detection in parallel, followed by executive summarization, conditional routing, and Human-in-the-Loop escalation.

---

## 🚀 Live Demo & Deployment
Explore the live application deployed on Streamlit Community Cloud:  
👉 **[https://insurance-claim-processing-ai-agent-langgraph.streamlit.app/](https://insurance-claim-processing-ai-agent-langgraph.streamlit.app/)**

---

## 🏗️ Multi-Agent Architecture & Workflow
The graph compiles **5 specialized nodes** orchestrated via LangGraph:
1. **Document Verification Agent (`document_verification_agent`):** Cross-checks submitted documents against required policy documents.
2. **Eligibility Check Agent (`eligibility_check_agent`):** Verifies active policy status and confirms the claim amount is within the coverage limit.
3. **Fraud Detection Agent (`fraud_detection_agent`):** Evaluates prior claim frequency, suspicious keywords in incident descriptions, and high-value risk thresholds.
*(Note: Nodes 1, 2, and 3 execute **in parallel** from `START` to maximize processing efficiency).*
4. **Claim Summary Agent (`claim_summary_agent`):** Aggregates parallel check outputs, generates a clean executive summary using Gemini, and determines routing decisions (`Auto Approve`, `Reject`, or `Human Approval`).
5. **Human Approval Agent (`human_approval_agent`):** Handles edge cases, suspicious claims, or high-value claims by preparing an escalation brief for manual human reviewer sign-off.

---

## 📂 Project Structure
```text
insurance-claim-agent/
├── app.py              # Main interactive Streamlit multi-agent web application
├── project.ipynb       # Jupyter Notebook demonstrating all 5 claim test scenarios
├── requirements.txt    # Python package dependencies
├── .env                # Local environment variables configuration
└── README.md           # Project documentation
```

---

⚙️ System Prerequisites
- Python 3.10 or higher installed on your machine.
- A Google Gemini API Key. (You can obtain one for free in seconds from [Google AI Studio](https://aistudio.google.com/app/apikey)).

🛠️ Setup & Installation Instructions
- Choose either Method A (uv recommended for speed) or Method B (traditional pip).

Method A: Using uv (Extremely Fast)
1. Install uv if it is not already installed on your system:
```bash
pip install uv
```
2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
3. Sync and install all required dependencies:
```bash
uv pip install -r requirements.txt
```

Method B: Using Standard pip
1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
2. Upgrade pip and install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

▶️ Running the Project
Step 1: Configure Environment Variables
Create a file named `.env` in the root directory of your project folder and add your Gemini API key:
```env
GOOGLE_API_KEY="your_actual_gemini_api_key_here"
```
*(Note: The Streamlit app also provides a secure input box in the UI if you prefer entering your API key interactively at runtime).*

Step 2: Run the Jupyter Notebook
To run all 5 test scenarios (Auto Approve, Rejection for missing docs/expired policies, and Human-in-the-Loop review for fraud/high value):
```bash
jupyter notebook project.ipynb
```

Step 3: Launch the Streamlit Web Application
To run the interactive multi-agent dashboard locally:
```bash
streamlit run app.py
```
Open your browser and navigate to the local URL displayed in your terminal (usually http://localhost:8501).

💡 Troubleshooting
- Missing API Key Error: Ensure your .env file is placed in the exact root folder of the project, or paste your key directly into the Streamlit input field.
- LangGraph state issues: Ensure you are using langgraph and langchain-google-genai versions matching requirements.txt.
