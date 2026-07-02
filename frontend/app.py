"""
Streamlit frontend for the Sensitive Data Detection & Compliance Assistant.

Talks to the FastAPI backend over HTTP. Run the backend first, then:
    streamlit run app.py
"""
import os
import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Sensitive Data & Compliance Assistant", page_icon="🛡️", layout="wide")

if "document_id" not in st.session_state:
    st.session_state.document_id = None
if "upload_result" not in st.session_state:
    st.session_state.upload_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🛡️ Sensitive Data Detection & Compliance Assistant")
st.caption("Upload a document to detect sensitive data, assess risk, and chat with it.")

with st.sidebar:
    st.header("Backend")
    st.code(BACKEND_URL, language="text")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3)
        st.success("Backend is reachable ✅") if health.ok else st.error("Backend unreachable ❌")
    except requests.exceptions.RequestException:
        st.error("Backend unreachable ❌ — start it with `uvicorn main:app --reload`")

    st.divider()
    st.header("Upload")
    uploaded_file = st.file_uploader("Choose a PDF, TXT, or CSV file", type=["pdf", "txt", "csv"])
    analyze_clicked = st.button("Analyze Document", type="primary", use_container_width=True)

    st.divider()
    with st.expander("📜 Audit Log"):
        if st.button("Refresh audit log"):
            try:
                resp = requests.get(f"{BACKEND_URL}/audit-log", timeout=5)
                if resp.ok:
                    df = pd.DataFrame(resp.json())
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.error("Could not fetch audit log.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {e}")

if analyze_clicked:
    if uploaded_file is None:
        st.warning("Please choose a file first.")
    else:
        with st.spinner("Uploading and analyzing document..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
                if resp.ok:
                    st.session_state.upload_result = resp.json()
                    st.session_state.document_id = st.session_state.upload_result["document_id"]
                    st.session_state.chat_history = []
                    st.success(f"Analysis complete for **{uploaded_file.name}**")
                else:
                    st.error(f"Backend error ({resp.status_code}): {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach backend: {e}")

result = st.session_state.upload_result

if result:
    tab_overview, tab_detections, tab_summary, tab_chat = st.tabs(
        ["📊 Overview", "🔍 Detections", "📋 Compliance Summary", "💬 Ask Questions"]
    )

    # ---------------- Overview ----------------
    with tab_overview:
        risk = result["risk"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Level", risk["risk_level"])
        col2.metric("Risk Score", risk["risk_score"])
        col3.metric("Characters Scanned", result["char_count"])

        risk_color = {"Low": "green", "Medium": "orange", "High": "red"}.get(risk["risk_level"], "gray")
        st.markdown(f"### Overall Risk: :{risk_color}[{risk['risk_level']}]")

        if risk["breakdown"]:
            chart_df = pd.DataFrame(
                [{"Category": k.replace("_", " ").title(), "Count": v} for k, v in risk["breakdown"].items()]
            ).sort_values("Count", ascending=False)
            st.bar_chart(chart_df.set_index("Category"))
        else:
            st.info("No sensitive data detected in this document.")

        with st.expander("Masked document preview"):
            st.text(result["masked_preview"])

    # ---------------- Detections ----------------
    with tab_detections:
        if not result["detections"]:
            st.info("No sensitive data patterns were found.")
        for det in result["detections"]:
            with st.expander(f"{det['type'].replace('_', ' ').title()}  —  {det['count']} found"):
                for ex in det["examples"]:
                    st.markdown(f"- **Masked value:** `{ex['masked_value']}`  \n  **Context:** _{ex['context']}_")

    # ---------------- Compliance Summary ----------------
    with tab_summary:
        summ = result["summary"]
        st.subheader("Narrative")
        st.write(summ["narrative"])

        st.subheader("✅ Compliance Observations")
        for item in summ["compliance_observations"]:
            st.markdown(f"- {item}")

        st.subheader("⚠️ Security Risks")
        for item in summ["security_risks"]:
            st.markdown(f"- {item}")

        st.subheader("🛠️ Suggested Remediation Steps")
        for item in summ["remediation_steps"]:
            st.markdown(f"- {item}")

    # ---------------- Chat / Q&A ----------------
    with tab_chat:
        st.write("Ask things like *“What sensitive data exists in the document?”* or *“Summarize this document.”*")

        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                st.write(turn["answer"])

        question = st.chat_input("Ask a question about this document...")
        if question:
            with st.chat_message("user"):
                st.write(question)
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/ask",
                        json={"document_id": st.session_state.document_id, "question": question},
                        timeout=60,
                    )
                    if resp.ok:
                        answer = resp.json()["answer"]
                    else:
                        answer = f"Backend error ({resp.status_code}): {resp.text}"
                except requests.exceptions.RequestException as e:
                    answer = f"Could not reach backend: {e}"

            with st.chat_message("assistant"):
                st.write(answer)

            st.session_state.chat_history.append({"question": question, "answer": answer})
else:
    st.info("👈 Upload a PDF, TXT, or CSV file from the sidebar and click **Analyze Document** to get started.")
