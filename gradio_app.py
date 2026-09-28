import os
import joblib
import pandas as pd
import gradio as gr

from aml_inference import AMLStatefulInference
from utils.recent_state import RecentTransactionState, WindowState

# ============================================================
# PATHS & INITIAL DATA LOAD
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "artifacts", "model", "aml_xgb_model.pkl"
)
STATE_PATH = os.path.join(
    BASE_DIR, "artifacts", "initial_state.pkl"
)
TEST_DATA_PATH = os.path.join(
    BASE_DIR, "artifacts", "test_transactions.parquet"
)

# Set up model bundle for once-only loading to avoid repeated disk I/O
MODEL_BUNDLE = joblib.load(MODEL_PATH)

def load_test_data(path):
    df = pd.read_parquet(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df.sort_values("Datetime").reset_index(drop=True)


def create_pipeline():
    state = joblib.load(STATE_PATH)
    return AMLStatefulInference(
        bundle=MODEL_BUNDLE,
        sender_state=state["sender_state"],
        receiver_state=state["receiver_state"],
        sender_seen_receivers=state["sender_seen_receivers"],
        receiver_seen_senders=state["receiver_seen_senders"],
        recent_state=state["recent_state"],
        pair_state=state.get("pair_state", {}),
        categorical_features_dict=state.get(
            "categorical_features_dict", None
        ),
    )


# Load dataset sekali di awal
try:
    test_df = load_test_data(TEST_DATA_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load test transactions: {e}")


# ============================================================
# LOGIC FUNCTIONS FOR GRADIO
# ============================================================

def init_state():
    """Inisialisasi pipeline dan state awal."""
    pipeline = create_pipeline()
    processed_df = pd.DataFrame()
    last_batch_df = pd.DataFrame()
    return pipeline, processed_df, last_batch_df


def reset_simulation():
    """Mereset seluruh state simulasi dan output tampilan."""
    pipeline, processed_df, last_batch_df = init_state()
    total_tx = len(test_df)

    return (
        pipeline,
        processed_df,
        last_batch_df,
        f"**Model:** {pipeline.model_type} | **Threshold:** {pipeline.threshold:.4f} | **Test transactions:** {total_tx:,}",
        f"{total_tx:,}",
        "0",
        f"{total_tx:,}",
        "0",
        0.0,
        "Ready to start simulation.",
        "0",
        "0",
        "0.00%",
        "0.00%",
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )


def process_next_batch(batch_size, pipeline, processed_df, last_batch_df):
    """Menjalankan batch inference berikutnya."""
    if pipeline is None:
        pipeline, processed_df, last_batch_df = init_state()

    total_transactions = len(test_df)
    previous_position = pipeline.inference_position

    if previous_position >= total_transactions:
        status_msg = "✅ All test transactions have been processed."
        # Hitung angka akhir
        flagged = (
            int(processed_df["prediction"].sum())
            if not processed_df.empty
            else 0
        )
        return (
            pipeline,
            processed_df,
            last_batch_df,
            gr.update(),
            f"{total_transactions:,}",
            f"{previous_position:,}",
            "0",
            f"{flagged:,}",
            1.0,
            status_msg,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    # Predict batch
    batch_result = pipeline.predict_batch(test_df, n_rows=batch_size)
    new_position = pipeline.inference_position

    # Dynamic update DataFrame cumulative
    if processed_df.empty:
        processed_df = batch_result.copy()
    else:
        processed_df = pd.concat(
            [processed_df, batch_result], ignore_index=True
        )

    last_batch_df = batch_result.copy()

    # Metrics Global
    remaining = max(total_transactions - new_position, 0)
    progress = new_position / total_transactions if total_transactions > 0 else 0
    flagged_total = int(processed_df["prediction"].sum())

    # Metrics Batch
    batch_alerts = int(last_batch_df["prediction"].sum())
    batch_avg_prob = last_batch_df["aml_probability"].mean()
    batch_max_prob = last_batch_df["aml_probability"].max()

    status_msg = f"Processed transactions {previous_position + 1:,} → {new_position:,}"

    # Prepare DataFrame display: Latest Batch
    disp_cols = [
        c
        for c in [
            "Datetime",
            "Sender_account",
            "Receiver_account",
            "Amount",
            "aml_probability",
            "prediction",
        ]
        if c in last_batch_df.columns
    ]
    display_batch = last_batch_df[disp_cols].copy()

    if "Amount" in display_batch.columns:
        display_batch["Amount"] = display_batch["Amount"].map(
            lambda x: f"{x:,.2f}"
        )
    if "aml_probability" in display_batch.columns:
        display_batch["aml_probability"] = display_batch[
            "aml_probability"
        ].map(lambda x: f"{x:.2%}")
    if "prediction" in display_batch.columns:
        display_batch["prediction"] = display_batch["prediction"].map(
            {0: "Normal", 1: "⚠️ AML Alert"}
        )

    # Prepare DataFrame display: Alerts
    alerts = last_batch_df[last_batch_df["prediction"] == 1].copy()
    if not alerts.empty:
        alert_cols = [
            c
            for c in [
                "Datetime",
                "Sender_account",
                "Receiver_account",
                "Amount",
                "aml_probability",
            ]
            if c in alerts.columns
        ]
        alert_df = alerts[alert_cols].copy()
        if "Amount" in alert_df.columns:
            alert_df["Amount"] = alert_df["Amount"].map(
                lambda x: f"{x:,.2f}"
            )
        if "aml_probability" in alert_df.columns:
            alert_df["aml_probability"] = alert_df[
                "aml_probability"
            ].map(lambda x: f"{x:.2%}")
    else:
        alert_df = pd.DataFrame()

    # Chart Data Preparation
    line_chart_df = processed_df[["Datetime", "aml_probability"]].sort_values(
        "Datetime"
    )

    counts = (
        processed_df["prediction"]
        .value_counts()
        .rename(index={0: "Normal", 1: "AML Alert"})
        .reset_index()
    )
    counts.columns = ["Status", "Count"]

    return (
        pipeline,
        processed_df,
        last_batch_df,
        f"**Model:** {pipeline.model_type} | **Threshold:** {pipeline.threshold:.4f} | **Test transactions:** {total_transactions:,}",
        f"{total_transactions:,}",
        f"{new_position:,}",
        f"{remaining:,}",
        f"{flagged_total:,}",
        progress,
        status_msg,
        f"{len(last_batch_df):,}",
        f"{batch_alerts:,}",
        f"{batch_avg_prob:.2%}",
        f"{batch_max_prob:.2%}",
        display_batch,
        alert_df,
        line_chart_df,
        counts,
    )


# ============================================================
# GRADIO UI BUILD
# ============================================================

custom_css = """
.hero {
    padding: 1.5rem 1.8rem;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
    border: 1px solid rgba(148, 163, 184, 0.20);
    margin-bottom: 1.5rem;
}
.hero-title { font-size: 2.2rem; font-weight: 700; color: white; }
.hero-subtitle { font-size: 1rem; color: #cbd5e1; line-height: 1.6; }
"""

with gr.Blocks(title="AML Transaction Monitoring", css=custom_css) as demo:

    # State variables
    pipeline_state = gr.State(create_pipeline())
    processed_state = gr.State(pd.DataFrame())
    last_batch_state = gr.State(pd.DataFrame())

    # Header
    gr.HTML(
        """
        <div class="hero">
            <div class="hero-title">🔎 AML Transaction Monitoring</div>
            <div class="hero-subtitle">
                Stateful streaming inference using XGBoost, historical transaction state,
                cumulative features, rolling time-window features, and micro-batched inference.
            </div>
        </div>
        """
    )

    # Sidebar Controls
    with gr.Sidebar():
        gr.Markdown("## Simulation Control")
        batch_slider = gr.Slider(
            minimum=1,
            maximum=250,
            value=10,
            step=1,
            label="Transactions per batch",
        )

        gr.Markdown("---")
        gr.Markdown("### Model Info")
        init_pipe = create_pipeline()
        model_info = gr.Markdown(
            f"**Model:** {init_pipe.model_type}\n\n"
            f"**Threshold:** {init_pipe.threshold:.4f}\n\n"
            f"**Test transactions:** {len(test_df):,}"
        )

        gr.Markdown("---")
        reset_btn = gr.Button("🔄 Reset Simulation", variant="secondary")

    # Global Metrics Section
    with gr.Row():
        m_total = gr.Textbox(
            label="Total Transactions",
            value=f"{len(test_df):,}",
            interactive=False,
        )
        m_processed = gr.Textbox(
            label="Processed", value="0", interactive=False
        )
        m_remaining = gr.Textbox(
            label="Remaining",
            value=f"{len(test_df):,}",
            interactive=False,
        )
        m_alerts = gr.Textbox(
            label="AML Alerts", value="0", interactive=False
        )

    progress_bar = gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=0.0,
        label="Simulation Progress",
        interactive=False,
    )

    # Simulation Controls
    gr.Markdown("### Transaction Stream")
    with gr.Row():
        btn_process = gr.Button("▶ Process Next Batch", variant="primary")
        status_box = gr.Textbox(
            label="Status",
            value="Ready to start simulation.",
            interactive=False,
        )

    # Latest Batch Metrics & Table
    gr.Markdown("### Latest Batch")
    with gr.Row():
        b_size = gr.Textbox(
            label="Batch Size", value="0", interactive=False
        )
        b_alerts = gr.Textbox(
            label="AML Alerts", value="0", interactive=False
        )
        b_avg = gr.Textbox(
            label="Avg Probability", value="0.00%", interactive=False
        )
        b_max = gr.Textbox(
            label="Highest Probability", value="0.00%", interactive=False
        )

    batch_table = gr.Dataframe(interactive=False)

    # Detected Alerts Section
    gr.Markdown("### ⚠️ Detected AML Transactions")
    alert_table = gr.Dataframe(interactive=False)

    # Charts Section
    gr.Markdown("### Monitoring Overview")
    with gr.Row():
        line_chart = gr.LinePlot(
            x="Datetime",
            y="aml_probability",
            title="AML Probability over Time",
            tooltip=["Datetime", "aml_probability"],
        )
        bar_chart = gr.BarPlot(
            x="Status",
            y="Count",
            title="Normal vs AML Alert Distribution",
            tooltip=["Status", "Count"],
        )

    # Technical Explanation Expander
    with gr.Accordion("ℹ️ How this inference works", open=False):
        gr.Markdown(
            """
            **Stateful inference architecture**
            
            The simulation does not rebuild the historical dataset every time a transaction is processed.
            The application starts from a pre-computed inference state (`initial_state.pkl`) containing:
            
            - sender cumulative state
            - receiver cumulative state
            - sender/receiver relationship state
            - sender-receiver pair state
            - recent transaction window state
            
            Each new transaction follows this sequence:
            1. **Read current state**
            2. **Build transaction features**
            3. **Run XGBoost inference**
            4. **Store AML probability**
            5. **Update cumulative and recent state**
            """
        )

    # Bind Events
    btn_process.click(
        fn=process_next_batch,
        inputs=[
            batch_slider,
            pipeline_state,
            processed_state,
            last_batch_state,
        ],
        outputs=[
            pipeline_state,
            processed_state,
            last_batch_state,
            model_info,
            m_total,
            m_processed,
            m_remaining,
            m_alerts,
            progress_bar,
            status_box,
            b_size,
            b_alerts,
            b_avg,
            b_max,
            batch_table,
            alert_table,
            line_chart,
            bar_chart,
        ],
    )

    reset_btn.click(
        fn=reset_simulation,
        inputs=[],
        outputs=[
            pipeline_state,
            processed_state,
            last_batch_state,
            model_info,
            m_total,
            m_processed,
            m_remaining,
            m_alerts,
            progress_bar,
            status_box,
            b_size,
            b_alerts,
            b_avg,
            b_max,
            batch_table,
            alert_table,
            line_chart,
            bar_chart,
        ],
    )

if __name__ == "__main__":
    demo.launch()