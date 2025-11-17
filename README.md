# AI Customer Success Command Center

AI-driven customer success & support dashboard built with **Python**, **Streamlit**, and **LLM agents**.  
It simulates a SaaS environment and shows how an AI-driven CS/TAM can use data + automation to manage accounts.

---

## 🔍 What this project does

- **Ticket insights**
  - Total tickets, open vs. resolved
  - Average resolution time
  - Backlog overview

- **Account health & churn**
  - Synthetic customer health score
  - Churn-risk flag using a simple ML model (`churn_model.pkl`)
  - Explanation text for *why* an account is at risk

- **AI Success Agent**
  - LLM-powered assistant (via `agents.py`)
  - Explains churn drivers in plain language
  - Suggests next-best actions / success plan for the CSM

- **CS UI**
  - Streamlit dashboard styled as an “enterprise CS cockpit”
  - Timeline filters for analyzing tickets and health over time

---

## 🛠 Tech stack

- Python 3.x
- Streamlit
- Pandas, NumPy
- Scikit-learn (for churn model)
- Plotly / other plotting libs
- LLM / OpenAI client (for the agent)

---

## 🚀 How to run locally

```bash
# 1. Clone the repo
git clone https://github.com/tanuchawan11-eng/AI-Customer-Success-Command-Center.git
cd AI-Customer-Success-Command-Center

# 2. (Optional) Create and activate virtual env
python3 -m venv venv
source venv/bin/activate    # on macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt   # if you add this file later

# 4. Run Streamlit app
streamlit run app.py
# AI-Customer-Success-Command-Center
AI-driven customer success &amp; support dashboard built with AI agents, Python and Streamlit.
