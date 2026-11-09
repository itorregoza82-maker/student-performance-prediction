"""
Streamlit interface for the Student Performance Prediction System.
Start after running train_model.py:
    streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_data.csv"
MODEL_PATH = BASE_DIR / "student_model.pkl"
METRICS_PATH = BASE_DIR / "model_evaluation.json"

st.set_page_config(
    page_title="Student Performance | Risk Insights",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {background-color: #f6f8fc;}
    .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.6rem 1.8rem; border-radius: 18px;
        background: linear-gradient(120deg, #13294b 0%, #245b83 100%);
        color: white; margin-bottom: 1.2rem;
    }
    .hero h1 {color: white; margin-bottom: .35rem;}
    .hero p {color: #e8f1fa; margin-bottom: 0;}
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e7ebf2;
        padding: 14px 16px; border-radius: 14px;
    }
    .note {
        background: #fff8e6; border-left: 4px solid #e5a000;
        padding: 12px 15px; border-radius: 8px; color: #57420b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


def missing_project_files():
    missing = [p.name for p in [DATA_PATH, MODEL_PATH] if not p.exists()]
    if missing:
        st.error(
            "Required project file(s) not found: "
            + ", ".join(missing)
            + ". Run `python train_model.py` in the project folder first."
        )
        st.stop()


missing_project_files()
model = load_model()
data = load_dataset()
required_columns = list(model.feature_names_in_)
data_without_target = data.drop(columns=["G3"], errors="ignore")

st.sidebar.title("🎓 Student Performance")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Dataset Preview", "Student Prediction"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Classroom prototype • Mathematics")
st.sidebar.caption("Predictions support review; they are not diagnoses.")

if page == "Dashboard":
    st.markdown(
        """
        <div class="hero">
          <h1>Student Performance Prediction</h1>
          <p>Explore historical academic patterns and flag records that may need
          additional academic support.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    total = len(data)
    avg_grade = pd.to_numeric(data["G3"], errors="coerce").mean()
    historical_risk = int((pd.to_numeric(data["G3"], errors="coerce") < 10).sum())
    risk_rate = (historical_risk / total * 100) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Student records", f"{total:,}")
    c2.metric("Average final grade", f"{avg_grade:.2f} / 20")
    c3.metric("Historical AT RISK", f"{historical_risk:,}")
    c4.metric("Historical risk rate", f"{risk_rate:.1f}%")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Grade distribution")
        grade_counts = data["G3"].value_counts().sort_index()
        st.bar_chart(grade_counts)
    with right:
        st.subheader("Historical risk categories")
        risk_counts = pd.Series(
            {
                "NOT AT RISK (G3 ≥ 10)": int((data["G3"] >= 10).sum()),
                "AT RISK (G3 < 10)": historical_risk,
            }
        )
        st.bar_chart(risk_counts)
        st.caption("Historical categories are based on the known final grade G3.")

    st.markdown(
        '<div class="note"><b>Responsible use:</b> This is a classroom '
        'demonstration based on historical records from two Portuguese schools. '
        'A prediction should prompt supportive review, not automatic decisions '
        'about a student.</div>',
        unsafe_allow_html=True,
    )

elif page == "Dataset Preview":
    st.title("Dataset Preview")
    st.write("Mathematics records retrieved from the UCI Student Performance dataset.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{data.shape[0]:,}")
    c2.metric("Columns", f"{data.shape[1]:,}")
    c3.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")

    st.subheader("Student records")
    st.dataframe(data, use_container_width=True, height=430)

    with st.expander("Column data types"):
        dtype_table = pd.DataFrame(
            {"Column": data.dtypes.index, "Data type": data.dtypes.astype(str).values}
        )
        st.dataframe(dtype_table, use_container_width=True)
    with st.expander("Missing values by column"):
        missing_table = data.isna().sum().rename("Missing values").to_frame()
        st.dataframe(missing_table, use_container_width=True)

    st.download_button(
        "Download full dataset",
        data.to_csv(index=False).encode("utf-8"),
        "student_data.csv",
        "text/csv",
    )

elif page == "Student Prediction":
    st.title("Student Risk Prediction")
    st.write(
        "Upload a CSV containing the same predictor columns used during training. "
        "Do not include G3; it is the outcome being estimated."
    )

    sample_csv = data_without_target.head(5).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download sample input CSV",
        sample_csv,
        "sample_prediction_input.csv",
        "text/csv",
        use_container_width=False,
    )

    uploaded = st.file_uploader("Choose a student CSV file", type=["csv"])
    if uploaded is not None:
        try:
            new_data = pd.read_csv(uploaded)
            if new_data.empty:
                st.error("The uploaded CSV has no student records.")
                st.stop()

            missing = [column for column in required_columns if column not in new_data.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
                st.info("Download the sample input CSV to see the required column names.")
            else:
                # Extra columns, including a supplied G3, are not used by the model.
                prediction_input = new_data.loc[:, required_columns].copy()
                predictions = model.predict(prediction_input)
                results = new_data.copy()
                results["Prediction"] = [
                    "AT RISK" if int(value) == 1 else "NOT AT RISK"
                    for value in predictions
                ]

                st.subheader("Prediction results")
                at_risk_count = int((predictions == 1).sum())
                not_at_risk_count = int((predictions == 0).sum())
                a, b, c = st.columns(3)
                a.metric("Records processed", f"{len(results):,}")
                b.metric("Predicted AT RISK", f"{at_risk_count:,}")
                c.metric("Predicted NOT AT RISK", f"{not_at_risk_count:,}")

                st.dataframe(results, use_container_width=True, height=400)
                st.download_button(
                    "⬇ Download prediction results",
                    results.to_csv(index=False).encode("utf-8"),
                    "student_predictions.csv",
                    "text/csv",
                    use_container_width=False,
                )
                st.caption(
                    "The model output is an estimate based on historical patterns. "
                    "It should be checked by a qualified faculty member."
                )
        except UnicodeDecodeError:
            st.error("The file could not be read as text. Please upload a UTF-8 CSV file.")
        except pd.errors.ParserError:
            st.error("The CSV format appears invalid. Check the file and try again.")
        except Exception as exc:
            st.error(f"Unable to process this file: {exc}")
