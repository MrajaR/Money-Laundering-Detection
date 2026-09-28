import os
import joblib
import pandas as pd
import streamlit as st

from aml_inference import AMLStatefulInference
from utils.recent_state import RecentTransactionState, WindowState

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AML Transaction Monitoring",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "model",
    "aml_xgb_model.pkl",
)

STATE_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "initial_state.pkl",
)

TEST_DATA_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "test_transactions.parquet",
)

MODEL_BUNDLE = joblib.load(MODEL_PATH)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            rgba(30, 41, 59, 0.95),
            rgba(15, 23, 42, 0.98)
        );
        border: 1px solid rgba(148, 163, 184, 0.20);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        color: white;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #cbd5e1;
        line-height: 1.6;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
    }

    .status-card {
        padding: 1rem;
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.2);
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(148, 163, 184, 0.18);
        padding: 1rem;
        border-radius: 12px;
        background: rgba(148, 163, 184, 0.04);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD TEST DATA
# ============================================================

@st.cache_data
def load_test_data(path):
    df = pd.read_parquet(path)

    df["Datetime"] = pd.to_datetime(
        df["Datetime"]
    )

    return (
        df
        .sort_values("Datetime")
        .reset_index(drop=True)
    )


# ============================================================
# CREATE PIPELINE
# ============================================================

def create_pipeline():

    state = joblib.load(
        STATE_PATH
    )

    pipeline = AMLStatefulInference(
        bundle=MODEL_BUNDLE,

        sender_state=state["sender_state"],
        receiver_state=state["receiver_state"],

        sender_seen_receivers=state[
            "sender_seen_receivers"
        ],

        receiver_seen_senders=state[
            "receiver_seen_senders"
        ],

        recent_state=state["recent_state"],

        pair_state=state.get(
            "pair_state",
            {}
        ),

        categorical_features_dict=state.get(
            "categorical_features_dict",
            None,
        ),
    )

    return pipeline


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "pipeline" not in st.session_state:
    st.session_state.pipeline = create_pipeline()

if "processed_results" not in st.session_state:
    st.session_state.processed_results = pd.DataFrame()

if "last_batch" not in st.session_state:
    st.session_state.last_batch = pd.DataFrame()


# ============================================================
# LOAD DATA
# ============================================================

try:
    test_df = load_test_data(
        TEST_DATA_PATH
    )

except Exception as e:
    st.error(
        f"Failed to load test transactions: {e}"
    )

    st.stop()


pipeline = st.session_state.pipeline


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="hero">
    <div class="hero-title">
        🔎 AML Transaction Monitoring
    </div>
    <div class="hero-subtitle">
        Stateful streaming inference using XGBoost,
        historical transaction state, cumulative features,
        rolling time-window features, and micro-batched inference.
    </div>
</div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## Simulation Control")

    batch_size = st.select_slider(
        "Transactions per batch",
        options=[
            1,
            5,
            10,
            25,
            50,
            100,
            250,
        ],
        value=10,
    )

    st.divider()

    st.markdown("### Model")

    st.write(
        f"**Model:** {pipeline.model_type}"
    )

    st.write(
        f"**Threshold:** {pipeline.threshold:.4f}"
    )

    st.write(
        f"**Test transactions:** {len(test_df):,}"
    )

    st.divider()

    if st.button(
        "🔄 Reset Simulation",
        use_container_width=True,
    ):

        st.session_state.pipeline = (
            create_pipeline()
        )

        st.session_state.processed_results = (
            pd.DataFrame()
        )

        st.session_state.last_batch = (
            pd.DataFrame()
        )

        st.rerun()


# ============================================================
# CURRENT STATE
# ============================================================

total_transactions = len(test_df)

current_position = (
    pipeline.inference_position
)

remaining = max(
    total_transactions - current_position,
    0,
)

progress = (
    current_position / total_transactions
    if total_transactions > 0
    else 0
)


# ============================================================
# METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Transactions",
        f"{total_transactions:,}",
    )

with col2:
    st.metric(
        "Processed",
        f"{current_position:,}",
    )

with col3:
    st.metric(
        "Remaining",
        f"{remaining:,}",
    )

with col4:

    if not st.session_state.processed_results.empty:

        processed_df = (
            st.session_state.processed_results
        )

        flagged = int(
            processed_df["prediction"].sum()
        )

    else:
        flagged = 0

    st.metric(
        "AML Alerts",
        f"{flagged:,}",
    )


# ============================================================
# PROGRESS
# ============================================================

st.progress(
    progress,
    text=(
        f"Simulation progress: "
        f"{current_position:,} / "
        f"{total_transactions:,} transactions"
    ),
)


# ============================================================
# SIMULATION CONTROL
# ============================================================

st.markdown(
    '<div class="section-title">Transaction Stream</div>',
    unsafe_allow_html=True,
)

control_col1, control_col2 = st.columns(
    [2, 1]
)


with control_col1:

    if current_position < total_transactions:

        if st.button(
            f"▶ Process Next {batch_size} Transactions",
            type="primary",
            use_container_width=True,
        ):

            previous_position = (
                pipeline.inference_position
            )

            batch_result = pipeline.predict_batch(
                test_df,
                n_rows=batch_size,
            )

            new_position = (
                pipeline.inference_position
            )

            st.session_state.last_batch = (
                batch_result
            )

            if (
                st.session_state.processed_results
                .empty
            ):

                st.session_state.processed_results = (
                    batch_result.copy()
                )

            else:

                st.session_state.processed_results = (
                    pd.concat(
                        [
                            st.session_state.processed_results,
                            batch_result,
                        ],
                        ignore_index=True,
                    )
                )

            st.success(
                f"Processed transactions "
                f"{previous_position + 1:,} "
                f"→ "
                f"{new_position:,}"
            )

            st.rerun()

    else:

        st.success(
            "✅ All test transactions have been processed."
        )


with control_col2:

    st.info(
        "Each batch updates the inference state "
        "before the next batch is processed."
    )


# ============================================================
# LAST BATCH
# ============================================================

last_batch = (
    st.session_state.last_batch
)

if not last_batch.empty:

    st.markdown(
        '<div class="section-title">Latest Batch</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Batch metrics
    # --------------------------------------------------------

    batch_col1, batch_col2, batch_col3, batch_col4 = (
        st.columns(4)
    )

    batch_alerts = int(
        last_batch["prediction"].sum()
    )

    batch_avg_probability = (
        last_batch["aml_probability"].mean()
    )

    batch_max_probability = (
        last_batch["aml_probability"].max()
    )

    with batch_col1:
        st.metric(
            "Batch Size",
            f"{len(last_batch):,}",
        )

    with batch_col2:
        st.metric(
            "AML Alerts",
            f"{batch_alerts:,}",
        )

    with batch_col3:
        st.metric(
            "Avg Probability",
            f"{batch_avg_probability:.2%}",
        )

    with batch_col4:
        st.metric(
            "Highest Probability",
            f"{batch_max_probability:.2%}",
        )

    # --------------------------------------------------------
    # Transaction table
    # --------------------------------------------------------

    display_columns = [
        "Datetime",
        "Sender_account",
        "Receiver_account",
        "Amount",
        "aml_probability",
        "prediction",
    ]

    available_columns = [
        col
        for col in display_columns
        if col in last_batch.columns
    ]

    display_df = last_batch[
        available_columns
    ].copy()

    if "Amount" in display_df.columns:

        display_df["Amount"] = (
            display_df["Amount"]
            .map(lambda x: f"{x:,.2f}")
        )

    if "aml_probability" in display_df.columns:

        display_df["aml_probability"] = (
            display_df["aml_probability"]
            .map(lambda x: f"{x:.2%}")
        )

    if "prediction" in display_df.columns:

        display_df["prediction"] = (
            display_df["prediction"]
            .map(
                {
                    0: "Normal",
                    1: "⚠️ AML Alert",
                }
            )
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ALERT SECTION
# ============================================================

if not last_batch.empty:

    alerts = last_batch[
        last_batch["prediction"] == 1
    ].copy()

    if not alerts.empty:

        st.markdown(
            '<div class="section-title">⚠️ Detected AML Transactions</div>',
            unsafe_allow_html=True,
        )

        alert_columns = [
            "Datetime",
            "Sender_account",
            "Receiver_account",
            "Amount",
            "aml_probability",
        ]

        alert_columns = [
            col
            for col in alert_columns
            if col in alerts.columns
        ]

        alert_df = alerts[
            alert_columns
        ].copy()

        if "Amount" in alert_df.columns:

            alert_df["Amount"] = (
                alert_df["Amount"]
                .map(lambda x: f"{x:,.2f}")
            )

        if "aml_probability" in alert_df.columns:

            alert_df["aml_probability"] = (
                alert_df["aml_probability"]
                .map(lambda x: f"{x:.2%}")
            )

        st.dataframe(
            alert_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# CUMULATIVE MONITORING
# ============================================================

all_results = (
    st.session_state.processed_results
)

if not all_results.empty:

    st.markdown(
        '<div class="section-title">Monitoring Overview</div>',
        unsafe_allow_html=True,
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        probability_chart = (
            all_results[
                ["Datetime", "aml_probability"]
            ]
            .set_index("Datetime")
            .sort_index()
        )

        st.line_chart(
            probability_chart,
            y="aml_probability",
        )

    with chart_col2:

        alert_counts = (
            all_results["prediction"]
            .value_counts()
            .rename(
                index={
                    0: "Normal",
                    1: "AML Alert",
                }
            )
        )

        st.bar_chart(
            alert_counts,
        )


# ============================================================
# TECHNICAL INFORMATION
# ============================================================

with st.expander(
    "ℹ️ How this inference works"
):

    st.markdown(
        """
        **Stateful inference architecture**

        The simulation does not rebuild the historical dataset
        every time a transaction is processed.

        The application starts from a pre-computed inference state:

        `initial_state.pkl`

        containing:

        - sender cumulative state
        - receiver cumulative state
        - sender/receiver relationship state
        - sender-receiver pair state
        - recent transaction window state

        Each new transaction follows this sequence:

        **1. Read current state**

        **2. Build transaction features**

        **3. Run XGBoost inference**

        **4. Store AML probability**

        **5. Update cumulative and recent state**

        The next transaction therefore sees the state generated
        by all previously processed transactions.

        The model prediction itself is performed in batches,
        while feature generation and state updates remain
        chronological.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AML Transaction Monitoring • Stateful XGBoost Inference • "
    "Micro-batched simulation"
)