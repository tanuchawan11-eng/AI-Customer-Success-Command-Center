import streamlit as st
import pandas as pd
import joblib
import numpy as np
from agents import run_agent
import plotly.express as px
from datetime import datetime, timedelta
import os
import pandas as pd
import streamlit as st
import urllib.parse
import json

# HubSpot ticket URL template (replace with your real portal ID later)
HUBSPOT_PORTAL_ID = "147272589"  # TODO: put your real HubSpot portal ID here
HUBSPOT_TICKET_URL_TEMPLATE = (
    f"https://app.hubspot.com/contacts/{HUBSPOT_PORTAL_ID}/ticket/{{ticket_id}}"
)



@st.cache_data
def load_tickets_data():
    csv_path = os.path.join("data", "hubspot_tickets.csv")
    df = pd.read_csv(csv_path)
    return df
def pick_column(df, candidates, fallback=None):
    """Pick the first column name from `candidates` that exists in df.columns."""
    for c in candidates:
        if c in df.columns:
            return c
    return fallback or df.columns[0]


# ------------ ENTERPRISE UI CSS ------------
def inject_css():
    st.markdown("""
        <style>
                /* SIDEBAR UPGRADE */
section[data-testid="stSidebar"] .stRadio label {
    font-size: 16px !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] .stSidebarContent {
    font-size: 16px !important;
}


        .stApp { background-color: #F3F4F6; }

        /* HEADER */
        .cs-main-title {
            font-size: 34px;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: -5px;
        }

        .cs-subtitle {
            font-size: 16px;
            color: #6B7280;
            margin-bottom: 25px;
        }

        /* PREMIUM KPI CARD */
        .kpi-card {
            background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
            border-radius: 16px;
            padding: 18px 20px 14px 20px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.10);
            border: 1px solid #E5E7EB;
        }

        .kpi-title {
            font-size: 13px;
            text-transform: uppercase;
            color: #6B7280;
            font-weight: 700;
            letter-spacing: .5px;
        }

        .kpi-value {
            font-size: 30px;
            font-weight: 800;
            margin-top: 4px;
            color: #111827;
        }

        .kpi-subtext {
            font-size: 12px;
            color: #6B7280;
            margin-top: -4px;
        }

        .kpi-icon {
            font-size: 22px;
            float: right;
            margin-top: -40px;
            opacity: .85;
        }

        /* TREND ICONS */
        .trend-up {
            color: #16A34A;  /* green */
            font-weight: 900;
        }

        .trend-down {
            color: #DC2626; /* red */
            font-weight: 900;
        }

        </style>
    """, unsafe_allow_html=True)

def kpi_card(title, value, sub="", icon="📊", trend=None):
    trend_symbol = ""
    if trend == "up":
        trend_symbol = "<span class='trend-up'>▲</span>"
    elif trend == "down":
        trend_symbol = "<span class='trend-down'>▼</span>"

    st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>{title}</div>
            <div class='kpi-value'>{value} {trend_symbol}</div>
            <div class='kpi-subtext'>{sub}</div>
            <div class='kpi-icon'>{icon}</div>
        </div>
    """, unsafe_allow_html=True)

# ================= LOAD MODEL & DATASETS =================
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()

df = pd.read_csv("customer_churn_sample.csv")

# --- ensure churn_percent exists and has variation ---
if "churn_percent" in df.columns:
    # if it's already 0–1, scale to %
    if df["churn_percent"].max() <= 1.0:
        df["churn_percent"] = df["churn_percent"] * 100

else:
    try:
        # 1) try to use your churn model
        if hasattr(model, "predict_proba"):
            # drop obvious non-feature columns, but don't crash if they don't exist
            X = df.drop(columns=["customerID", "Churn"], errors="ignore")
            churn_proba = model.predict_proba(X)[:, 1]
            df["churn_percent"] = churn_proba * 100
        else:
            raise ValueError("Model has no predict_proba")
    except Exception:
        # 2) fallback: if there's a Churn Yes/No column, convert to % scores
        if "Churn" in df.columns:
            df["churn_percent"] = df["Churn"].map({"Yes": 85.0, "No": 15.0})
        else:
            # 3) last-resort synthetic churn scores so chart looks realistic
            #    (for portfolio/demo when you only have synthetic data)
            n = len(df)
            base_scores = np.linspace(5, 95, n)   # spread from 5% to 95%
            np.random.shuffle(base_scores)
            df["churn_percent"] = base_scores

# ---- define health segment based on churn_percent ----
def success_segment(percent):
    if percent > 70:
        return "Churn list"
    elif percent > 40:
        return "At risk"
    else:
        return "Healthy"

df["success_segment"] = df["churn_percent"].apply(success_segment)

        

ticket_df = pd.read_csv("ticket_dataset_v2.csv")
ticket_df["created_at"] = pd.to_datetime(ticket_df["created_at"])
ticket_df["closed_at"]  = pd.to_datetime(ticket_df["closed_at"])



# Page config
st.set_page_config(page_title="AI Customer Success Dashboard", layout="wide")

inject_css()
st.sidebar.title("AI Customer Success")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🤖 AI Success Agents", "💬 Support AI Chatbot (coming soon)"]
)


# ===================== DASHBOARD PAGE =====================
if page == "📊 Dashboard":

    # ---------- HEADER ----------
    st.markdown(
        "<div class='cs-main-title'>AI Customer Success Command Center</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='cs-subtitle'>Enterprise-level insights powered by analytics + AI.</div>",
        unsafe_allow_html=True,
    )

    # small space
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- DASHBOARD TIME FILTER (BELOW TITLE) ----------
    st.markdown("#### Dashboard time range")

    dash_range = st.selectbox(
        "Select time range",
        ["Last 7 days", "Last 30 days", "Last 90 days", "Last 6 months", "Year to Date", "All time"],
        index=1,
        key="dash_time_filter",
    )

    today = datetime.today()

    if dash_range == "Last 7 days":
        start_date = today - timedelta(days=7)
    elif dash_range == "Last 30 days":
        start_date = today - timedelta(days=30)
    elif dash_range == "Last 90 days":
        start_date = today - timedelta(days=90)
    elif dash_range == "Last 6 months":
        start_date = today - timedelta(days=182)
    elif dash_range == "Year to Date":
        start_date = datetime(today.year, 1, 1)
    else:  # "All time"
        start_date = ticket_df["created_at"].min()

    # make sure created_at is datetime
    ticket_df["created_at"] = pd.to_datetime(ticket_df["created_at"])

    # tickets used ONLY for the dashboard
    dash_tickets = ticket_df[
        (ticket_df["created_at"] >= start_date) & (ticket_df["created_at"] <= today)
    ]

    st.caption(
        f"📅 Dashboard view: {start_date.date()} → {today.date()} · {len(dash_tickets)} tickets"
    )

    # ---------- KPI ROW (4 CARDS IN ONE LINE) ----------
    total_tickets = len(dash_tickets)
    sla_rate = (dash_tickets["sla_breach"] == "yes").mean() * 100

    avg_time = (
        (dash_tickets["closed_at"] - dash_tickets["created_at"])
        .dt.total_seconds()
        .mean()
        / 3600
    )

    # churn-based risk KPI
    risk_rate = (df["churn_percent"] > 40).mean() * 100

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        kpi_card("Total Tickets", total_tickets, "selected range", icon="📊")

    with k2:
        kpi_card("SLA Breach %", f"{sla_rate:.1f}%", "breached", icon="🚨")

    with k3:
        kpi_card("Avg Resolution", f"{avg_time:.1f} h", "hours", icon="⏱️")

    with k4:
        kpi_card("Accounts At Risk", f"{risk_rate:.1f}%", ">40% churn", icon="⚠️")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- BUILD CHART DATA (USING dash_tickets) ----------
    # Ticket summary pie
    status_counts = dash_tickets["status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]

    sla_breach_count = dash_tickets[dash_tickets["sla_breach"] == "yes"].shape[0]
    status_counts = pd.concat(
        [
            status_counts,
            pd.DataFrame({"status": ["SLA Breach"], "count": [sla_breach_count]}),
        ],
        ignore_index=True,
    )

    fig_status = px.pie(
        status_counts,
        names="status",
        values="count",
        hole=0.4,
    )

    # ---------- ACCOUNT HEALTH PIE ----------
    def categorize_churn(percent):
        if percent > 70:
            return "Likely to Churn"
        elif percent > 40:
            return "At Risk"
        else:
            return "Healthy"

    df["health_category"] = df["churn_percent"].apply(categorize_churn)
    health_counts = df["health_category"].value_counts().reset_index()
    health_counts.columns = ["category", "count"]

    fig_health = px.pie(
        health_counts,
        names="category",
        values="count",
        hole=0.4,
    )

    # ---------- TICKET TRENDS LINE ----------
    dash_tickets_sorted = dash_tickets.copy()
    dash_tickets_sorted["created_date"] = dash_tickets_sorted["created_at"].dt.date

    trend = dash_tickets_sorted.groupby("created_date").agg(
        {
            "sla_breach": lambda x: (x == "yes").sum(),
            "reopened": lambda x: (x == "yes").sum(),
            "status": lambda x: (x == "closed").sum(),
            "change_implementation": lambda x: (x == "yes").sum(),
        }
    )

    trend.columns = [
        "SLA Breaches",
        "Reopened Tickets",
        "Closed Tickets",
        "Change Implementation",
    ]

    fig_trend = px.line(
        trend,
        labels={"value": "Count", "created_date": "Date"},
    )

    # ---------- SINGLE ROW: 3 CHARTS ----------
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("<div class='cs-card'>", unsafe_allow_html=True)
        st.markdown("### Ticket Summary")
        st.plotly_chart(fig_status, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='cs-card'>", unsafe_allow_html=True)
        st.markdown("### Account Health Summary")
        st.plotly_chart(fig_health, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("<div class='cs-card'>", unsafe_allow_html=True)
        st.markdown("### Ticket Trends")
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)



# ===================== AI SUCCESS AGENTS PAGE =====================
if page == "🤖 AI Success Agents":

    st.markdown("## 🤖 AI Success Agents")
    st.markdown(
        "Use AI to analyse churn risk, generate success plans and full Customer Success playbooks.",
        unsafe_allow_html=True,
    )

        # ---------- ROW 1: Time range + Category + Customer selector + Metrics ----------
    c1, c2, c3 = st.columns([1.1, 1.1, 2])

    # Time range (UI context)
    with c1:
        time_filter = st.selectbox(
            "Time range",
            ["Last 7 days", "Last 30 days", "Last 90 days", "Last 6 months", "Year to Date", "All time"],
            index=1,
        )

    # Category + Customer selection
    with c2:
        category = st.selectbox(
            "Customer health category",
            ["Healthy", "At risk", "Churn list"],
            index=0,
        )

        # 🔹 filter customers by the chosen category
        filtered_df = df[df["success_segment"] == category]

        if filtered_df.empty:
            st.warning(f"No customers found in category '{category}'.")
            st.stop()

        # 🔹 dropdown now only shows customers from filtered_df, not all df
        customer_id = st.selectbox(
            "Select customer",
            filtered_df["customer_id"].unique()
        )

    # 🔹 also read the customer from filtered_df, not df
    customer_data = filtered_df[filtered_df["customer_id"] == customer_id].iloc[0]



    # Customer metrics mini-KPIs
    with c3:
        st.markdown("#### Customer metrics")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Tickets (30d)", int(customer_data["tickets_last_30_days"]))
        with m2:
            st.metric("SLA violations", int(customer_data["sla_violations"]))
        with m3:
            st.metric("Reopened tickets", int(customer_data["reopened_tickets"]))

        m4, m5, m6 = st.columns(3)
        with m4:
            st.metric("Avg response (min)", round(float(customer_data["avg_response_time"]), 1))
        with m5:
            st.metric("Sentiment score", round(float(customer_data["sentiment_score"]), 2))
        with m6:
            st.metric("Usage drop %", round(float(customer_data["usage_drop_percent"]), 1))

    st.markdown("---")

    # ---------- ROW 2: Churn risk ----------
    st.markdown("### 🔍 Churn risk")

    feature_cols = [
        "tickets_last_30_days",
        "sla_violations",
        "reopened_tickets",
        "avg_response_time",
        "sentiment_score",
        "usage_drop_percent",
    ]
    input_data = customer_data[feature_cols].values.reshape(1, -1)
    churn_prob = model.predict_proba(input_data)[0][1]
    churn_percent = round(churn_prob * 100, 2)

    if churn_percent >= 70:
        risk_label = "🔥 High Risk"
        color = "red"
    elif churn_percent >= 40:
        risk_label = "⚠️ Medium Risk"
        color = "orange"
    else:
        risk_label = "🟢 Low Risk"
        color = "green"

    st.markdown(
        f"#### Churn Risk: <span style='color:{color}'>{risk_label}</span>",
        unsafe_allow_html=True,
    )
    st.write(f"**Probability of churn:** **{churn_percent}%**")

    st.markdown("---")

    # ---------- ROW 3: 3 AI agents in one row ----------
    st.markdown("### 🧠 AI Assistants")

    a1, a2, a3 = st.columns(3)

    # 1) Churn Explanation Agent
    with a1:
        st.markdown("#### 1. Churn explanation")
        if st.button("Generate explanation", use_container_width=True):
            prompt = f"""
            You are an expert Customer Success AI Agent.
            Explain why this customer has a churn probability of {churn_percent}%.
            Use these metrics:

            Tickets (last 30 days): {customer_data['tickets_last_30_days']}
            SLA violations: {customer_data['sla_violations']}
            Reopened tickets: {customer_data['reopened_tickets']}
            Avg response time: {customer_data['avg_response_time']}
            Sentiment score: {customer_data['sentiment_score']}
            Usage drop %: {customer_data['usage_drop_percent']}

            Provide:
            1. Key churn risk drivers
            2. Customer behavior insights
            3. Recommended CS actions
            4. Tone: professional, confident, helpful
            """
            explanation = run_agent(prompt)
            st.write(explanation)

    # 2) Success Plan Agent
    with a2:
        st.markdown("#### 2. Success plan")
        if st.button("Generate success plan", use_container_width=True):
            prompt = f"""
            You are an expert Customer Success Manager AI Agent.
            Create a proactive success plan for this customer based on their metrics:

            Tickets (last 30 days): {customer_data['tickets_last_30_days']}
            SLA violations: {customer_data['sla_violations']}
            Reopened tickets: {customer_data['reopened_tickets']}
            Avg response time: {customer_data['avg_response_time']}
            Sentiment score: {customer_data['sentiment_score']}
            Usage drop %: {customer_data['usage_drop_percent']}
            Churn probability: {churn_percent}%

            Provide:
            1. Immediate mitigation actions
            2. Short-term Customer Success strategy
            3. Long-term adoption & product engagement steps
            4. Communication plan for CSM → Customer
            5. Recommendations for training, onboarding, or product usage
            6. Risks to monitor and how to mitigate them

            Tone: professional, proactive, strategic.
            """
            plan = run_agent(prompt)
            st.write(plan)

    # 3) Full Playbook Agent
    with a3:
        st.markdown("#### 3. Full CS playbook")
        if st.button("Generate playbook", use_container_width=True):
            full_prompt = f"""
            You are a senior Customer Success leader AI Agent.
            Generate a complete Customer Success Playbook for this customer.

            - Tickets last 30 days: {customer_data['tickets_last_30_days']}
            - SLA violations: {customer_data['sla_violations']}
            - Reopened tickets: {customer_data['reopened_tickets']}
            - Avg response time (minutes): {customer_data['avg_response_time']}
            - Sentiment score: {customer_data['sentiment_score']}
            - Usage drop percent: {customer_data['usage_drop_percent']}
            - Churn probability: {churn_percent}%

            The playbook should include:
            1. Customer Overview
            2. Churn Risk Analysis
            3. Key Risk Drivers
            4. Customer Behavior & Ticket Trends
            5. Health Assessment with reasoning
            6. Immediate Mitigation Actions (next 7 days)
            7. 30-Day Success Plan
            8. 90-Day Adoption & Growth Plan
            9. Communication Strategy
            10. Product/Feature Adoption Plan
            11. Internal Follow-up Tasks
            12. KPIs and Success Metrics

            Tone: professional, strategic, and clear.
            """
            playbook = run_agent(full_prompt)
            st.write(playbook)

            # Download buttons
            st.download_button(
                label="⬇️ Download Playbook (TXT)",
                data=playbook,
                file_name=f"{customer_id}_playbook_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

            md_report = f"# Customer Success Playbook for {customer_id}\n\n" + playbook
            st.download_button(
                label="⬇️ Download Playbook (MD)",
                data=md_report,
                file_name=f"{customer_id}_playbook_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
# ===================== AI REPLY STUDIO PAGE =====================
if page == "💬 Support AI Chatbot (coming soon)":

    st.markdown("<div class='cs-main-title'>AI Reply Studio</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='cs-subtitle'>Paste a ticket ID from your CRM, let AI pull the latest message and draft a reply.</div>",
        unsafe_allow_html=True,
    )

    # Load tickets once (our "CRM")
    tickets_df = load_tickets_data()
    id_col = pick_column(tickets_df, ["Ticket ID", "ticket_id", "ID", "id"])
    subject_col = pick_column(tickets_df, ["Ticket name", "Subject", "subject", "Ticket subject"], id_col)
    desc_col = pick_column(
        tickets_df,
        ["Ticket description", "Description", "description", "Latest message"],
        fallback=subject_col,   # if no description column, use subject as message
    )

    left, right = st.columns([1.1, 1.4])

    # ---------- LEFT: TICKET ID & MESSAGE ----------
    with left:
        st.markdown("### 1️⃣ Ticket from CRM")

        # init state
        if "customer_message" not in st.session_state:
            st.session_state["customer_message"] = ""
        if "ticket_id_input" not in st.session_state:
            st.session_state["ticket_id_input"] = ""
        if "issue_type" not in st.session_state:
            st.session_state["issue_type"] = "Auto-detect"
        if "detected_issue_type" not in st.session_state:
            st.session_state["detected_issue_type"] = "Not detected yet"

        ticket_id = st.text_input(
            "Ticket ID (from HubSpot / CRM)",
            key="ticket_id_input",
            placeholder="e.g. 2922297xxxxx",
        )

        if st.button("🔍 Load ticket from CRM"):
            if not ticket_id.strip():
                st.warning("Please paste a Ticket ID first.")
            else:
                match = tickets_df[tickets_df[id_col].astype(str) == ticket_id.strip()]
                if match.empty:
                    st.error("No ticket found with that ID in hubspot_tickets.csv.")
                else:
                    row = match.iloc[0]
                    # latest message (or subject fallback)
                    st.session_state["customer_message"] = str(row.get(desc_col, "")) or str(row.get(subject_col, ""))
                    # reset issue type so AI can auto-detect on generate
                    st.session_state["issue_type"] = "Auto-detect"
                    st.session_state["detected_issue_type"] = "Not detected yet"
                    st.success(f"Loaded ticket {row[id_col]} – {row[subject_col]}")

        customer_message = st.text_area(
            "Latest customer message / conversation",
            height=250,
            key="customer_message",
        )

        issue_type = st.selectbox(
            "Issue type (optional)",
            ["Auto-detect", "Login / access", "Billing / invoice", "Bug / error", "Feature question", "Other"],
            key="issue_type",
        )

        tone = st.selectbox(
            "Preferred tone",
            ["Professional & friendly", "Extra empathetic", "Very concise", "More technical"],
        )

        st.caption(f"Detected issue type: **{st.session_state.get('detected_issue_type', 'Not detected yet')}**")

    # ---------- RIGHT: AI DRAFTED REPLY ----------
    with right:
        st.markdown("### 2️⃣ AI drafted reply")

        if "reply_text" not in st.session_state:
            st.session_state["reply_text"] = ""

        # Generate AI reply (also auto-detect issue type when needed)
        if st.button("✨ Generate AI reply", use_container_width=True):
            if not st.session_state.customer_message.strip():
                st.warning("Please load a ticket or paste a customer message first.")
            else:
                with st.spinner("Thinking..."):

                    # 1) Auto-detect issue type if still on Auto-detect
                    if st.session_state.issue_type == "Auto-detect":
                        classify_prompt = f"""
You are a support triage assistant.

Classify the main issue in this message into EXACTLY one of these categories:
- Login / access
- Billing / invoice
- Bug / error
- Feature question
- Other

Return only the category text, nothing else.

Message:
{st.session_state.customer_message}
"""
                        detected_type = run_agent(classify_prompt).splitlines()[0].strip()
                        st.session_state["detected_issue_type"] = detected_type
                        type_for_reply = detected_type
                    else:
                        type_for_reply = st.session_state.issue_type
                        st.session_state["detected_issue_type"] = type_for_reply

                    # 2) Generate reply
                    reply_prompt = f"""
You are a Senior Customer Support Engineer.

Customer message:
---
{st.session_state.customer_message}
---

Issue type: {type_for_reply}
Preferred tone: {tone}

Write an email reply to the customer:
- Acknowledge the issue and show empathy.
- Ask any clarifying questions you really need.
- Provide next steps or troubleshooting guidance.
- Keep it {tone.lower()}.
- Do NOT mention that you're an AI system.
- End with a warm closing and generic signature like "Best regards, Your Support Team".
"""
                    reply = run_agent(reply_prompt)
                    st.session_state["reply_text"] = reply

        # main reply editor
        reply_text = st.text_area(
            "Edit the reply before sending",
            key="reply_text",
            height=260,
        )

        # ---- Only "Make shorter" button ----
        if st.button("Make shorter"):
            if st.session_state.reply_text.strip():
                shorter = run_agent(
                    "Rewrite this reply to be shorter but keep all key information:\n\n"
                    + st.session_state.reply_text
                )
                st.session_state.reply_text = shorter
            else:
                st.warning("No reply text to shorten yet.")

            
               
                # ---------- READY TO REPLY ----------
        st.markdown("### 3️⃣ Ready to reply")

        send_via = st.radio(
            "Where should this reply go?",
            ["CRM tool (via Ticket ID)", "Outlook / email client"],
            index=0,
        )

        if st.button("✅ Ready to reply", use_container_width=True):
            # Get the current reply text safely
            reply_text_value = st.session_state.get("reply_text", "").strip()

            if not reply_text_value:
                st.warning("Generate and/or edit a reply before marking it as ready.")
            else:
                # ----- OUTLOOK / EMAIL CLIENT -----
                if send_via.startswith("Outlook"):
                    subject = urllib.parse.quote("Customer support reply")
                    body = urllib.parse.quote(reply_text_value)
                    mailto_link = f"mailto:?subject={subject}&body={body}"

                    # Show a button that opens the email client when clicked
                    st.markdown(
                        f"""
                        <a href="{mailto_link}">
                            <button style="
                                background-color:#2563EB;
                                color:white;
                                border:none;
                                padding:8px 16px;
                                border-radius:6px;
                                cursor:pointer;
                                font-size:14px;
                                margin-top:8px;
                            ">
                                📧 Open in Outlook / email client
                            </button>
                        </a>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.success(
                        "Reply is ready. Click the button above to open your email client (e.g., Outlook) with the text pre-filled."
                    )

                # ----- CRM TOOL BRANCH -----
                else:
                    ticket_id_val = st.session_state.get("ticket_id_input", "").strip()
                    if not ticket_id_val:
                        st.warning("Add a Ticket ID on the left so the CRM knows which ticket to update.")
                    else:
                        # 1) Copy reply text to clipboard
                        js_text = json.dumps(reply_text_value)  # safe JS string
                        st.markdown(
                            f"""
                            <script>
                            (function() {{
                                try {{
                                    if (navigator && navigator.clipboard) {{
                                        navigator.clipboard.writeText({js_text});
                                    }}
                                }} catch (e) {{}}
                            }})();
                            </script>
                            """,
                            unsafe_allow_html=True,
                        )

                        # 2) Build HubSpot ticket URL from template constant
                        ticket_url = HUBSPOT_TICKET_URL_TEMPLATE.format(ticket_id=ticket_id_val)

                        # 3) Show a button/link to open HubSpot ticket in new tab
                        st.markdown(
                            f"""
                            <a href="{ticket_url}" target="_blank">
                                <button style="
                                    background-color:#10B981;
                                    color:white;
                                    border:none;
                                    padding:8px 16px;
                                    border-radius:6px;
                                    cursor:pointer;
                                    font-size:14px;
                                    margin-top:8px;
                                ">
                                    🔗 Open ticket in HubSpot
                                </button>
                            </a>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.success(
                            f"Reply is ready for CRM. We've copied the text to your clipboard. "
                            f"Click the green button above to open the HubSpot page for ticket **{ticket_id_val}**, "
                            "then paste the reply into the ticket's reply box and send."
                        )











