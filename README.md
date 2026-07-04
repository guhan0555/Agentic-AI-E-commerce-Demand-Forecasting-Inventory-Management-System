# Agentic AI E-commerce Demand Forecasting & Inventory Management System

## Run it
pip install -r requirements.txt
streamlit run app.py

Then open the URL shown in the terminal (usually http://localhost:8501).

## Deploy it for free (Streamlit Community Cloud)
1. Push this `app/` folder to a GitHub repo.
2. Go to https://share.streamlit.io, connect the repo, and set the main file to `app.py`.
3. Streamlit Cloud installs `requirements.txt` automatically and gives you a public URL.

## Files
- agentic_ecommerce.py  -> the 7 agents + orchestrator (core logic, runnable standalone)
- app.py                -> Streamlit dashboard wrapping the agents (the "app")
- make_charts.py        -> generates the charts used in the PDF report
- build_report.py       -> generates the full PDF report
- sample_data_online_retail.csv -> real UCI "Online Retail" dataset (541,909 rows) used to validate everything

## Standalone run (no UI)
python3 agentic_ecommerce.py
