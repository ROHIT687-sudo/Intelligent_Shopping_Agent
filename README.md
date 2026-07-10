# Intelligent Shopping Agent 🛍️🤖

An agentic AI-powered consumer companion engineered to eliminate marketplace data fragmentation, expose automated review manipulation, and forecast dynamic price volatility. Built natively in Python using the **IBM watsonx.ai ecosystem** and the **Meta Llama-3.3-70B-Instruct** foundation model.

---

## 🚀 Core Features

- **Multi-Merchant Data Fusion:** Aggregates and normalizes raw product catalogs, pricing sheets, and specification tables side-by-side from Amazon, eBay, and Walmart into a unified matrix view [11.1].
- **Heuristic Review Integrity Auditor:** Runs localized, low-latency Python regular expression (`regex`) token scans to isolate character word-frequency repetitions, instantly exposing automated fake bot review chains offline [11.3].
- **Statistical Market Volatility Predictor:** Computes real-time Standard Deviations over dynamic cross-platform retail price variations to generate automated buying velocity flags (**Buy Now** vs. **Wait**) [11.3].
- **Eco-Footprint Tracking Tool:** Dynamically calculates an objective product sustainability index scale (0-10) using fabric composition strings and component resource data.
- **Cognitive Multi-Agent Orchestration:** Establishes secure cloud connections to route structured prompt vector templates through the enterprise **Meta Llama-3.3-70B-Instruct model** via authenticated **watsonx.ai Studio** nodes to synthesize actionable 3-sentence buying summaries [11.2].

---

## 🛠️ Technology Stack & Infrastructure

- **Frontend Interface Framework:** Streamlit (UI Dashboard Canvas & Sidebar Control Inputs) [11.4]
- **Cognitive Core Reasoning Model:** `meta-llama/llama-3-3-70b-instruct` (via IBM watsonx Cloud SDK) [11.2]
- **Cloud Computing Gateway Backbone:** IBM Cloud (Dallas Server Node) with IAM Security Token Validation [11.2]
- **Data Structuring Environment:** Python 3.10+ Native Calculation Arrays (`re`, `math`) [11.3]
- **Credential Decoupling Storage:** `python-dotenv` (Encapsulated Project Environment Files)

---

## 📂 Project Workspace Repository Directory Structure

```text
├── .env                                # Protected credential variables (API keys, Project ID)
├── app.py                              # Main application production source script execution file
├── requirements.txt                    # System environment requirement lock dependencies
├── Problem statement 2026-2027 (1).docx # Evaluation grading criteria requirement document
└── Intelligent_Shopping_Agent_Problem_11.pptx # Technical presentation evaluation slide deck
```

---

## ⚙️ Installation & Workspace Setup

### 1. Clone the Project Repository Locally
```bash
gh repo clone ROHIT687-sudo/Intelligent_Shopping_Agent
cd Intelligent_Shopping_Agent
```

### 2. Install Required Production Libraries
```bash
pip install -r requirements.txt
```

### 3. Initialize Local Cryptographic Credentials File
Create a hidden `.env` file in the root directory layout workspace containing your authenticated IBM credentials parameters:
```env
WATSONX_APIKEY="your_secure_ibm_cloud_platform_api_key"
PROJECT_ID="de474e88-6096-4866-9c27-4a70e0be2555"
```

### 4. Run the Streamlit Dashboard System Application
```bash
streamlit run app.py
```
