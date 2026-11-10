from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import numpy as np

import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    shap = None

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

st.set_page_config(
    page_title="HCLTech Data Analytics Project",
    page_icon="📊",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "loan_recovery_model.pkl"

PROCESSED_DATA_PATH = ROOT / "data" / "03_processed_data.pkl"   

FEATURE_COLUMNS = [
    "age",
    "income",
    "credit_score",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "debt_to_income",
    "existing_loans",
    "previous_defaults",
    "payment_delays",
    "employment_years",
    "collateral_value",
    "contact_success_rate",
    "employment_type_Contract",
    "employment_type_Salaried",
    "employment_type_Self-Employed",
    "employment_type_Unemployed",
    "loan_purpose_Education",
    "loan_purpose_Home",
    "loan_purpose_Personal",
    "loan_purpose_Vehicle",
]

EMPLOYMENT_OPTIONS = [
    "Salaried",
    "Self-Employed",
    "Contract",
    "Unemployed",
]

PURPOSE_OPTIONS = [
    "Personal",
    "Vehicle",
    "Home",
    "Education",
]


@st.cache_resource
def load_artifact(path):
    return joblib.load(path)


st.title("📊 HCLTech Data Analytics Project")
st.caption(
    "Loan Recovery Prediction | Random Forest Classification"
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    artifact = load_artifact(str(MODEL_PATH))
    model = artifact["model"]
    features = list(
        artifact.get("feature_columns", FEATURE_COLUMNS)
    )

    if set(features) != set(FEATURE_COLUMNS):
        st.error("Model feature columns do not match the app.")
        st.stop()

except Exception as exc:
    st.error(f"Unable to load model: {exc}")
    st.stop()



with st.sidebar:

    st.markdown("## 📊 HCLTech")
    st.caption("LOAN RECOVERY ANALYTICS")

    st.divider()

    st.markdown("### Model Overview")

    st.markdown("**Algorithm**")
    st.write("Random Forest Classifier")

    st.markdown("**Input Features**")
    st.write("21")

    st.markdown("**Model Status**")
    st.success("Trained and Ready")

    st.divider()

    st.markdown("### Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Recall", "91.8%")

    with col2:
        st.metric("ROC-AUC", "0.913")

    st.divider()

    st.markdown("### How It Works")

    st.markdown("""
    **01** · Enter borrower details

    **02** · Predict recovery probability

    **03** · Explore SHAP explanations

    **04** · Review recovery insights
    """)

    st.divider()

    st.caption(
        "HCLTech Data Analytics Capstone Project"
    )

    with st.expander("Responsible Use"):
        st.caption(
            "Predictions are estimates, not guarantees. "
            "All decisions require appropriate human review."
        )



st.subheader("Enter Customer and Loan Details")

with st.form("loan_details"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age (years)", 18, 100, 40
        )
        income = st.number_input(
            "Income", min_value=0.0,
            value=61000.0, step=1000.0
        )
        credit_score = st.number_input(
            "Credit Score", 300.0, 900.0, 650.0,
            step=1.0
        )
        loan_amount = st.number_input(
            "Loan Amount", min_value=0.0,
            value=250000.0, step=5000.0
        )
        loan_term_months = st.selectbox(
            "Loan Term (months)",
            [6, 12, 18, 24, 36, 48, 60, 72, 84, 120],
            index=5,
        )

    with col2:
        interest_rate = st.number_input(
            "Interest Rate (%)", 0.0, 100.0, 10.0,
            step=0.25
        )
        debt_to_income = st.number_input(
            "Debt-to-Income Ratio", 0.0, 10.0, 0.30,
            step=0.01, format="%.3f"
        )
        existing_loans = st.number_input(
            "Existing Loans", 0, 100, 1
        )
        previous_defaults = st.number_input(
            "Previous Defaults", 0, 100, 0
        )
        payment_delays = st.number_input(
            "Payment Delays", 0, 100, 1
        )

    with col3:
        employment_years = st.number_input(
            "Years Employed", 0.0, 80.0, 5.0,
            step=1.0
        )
        collateral_value = st.number_input(
            "Collateral Value", min_value=0.0,
            value=100000.0, step=5000.0
        )
        contact_success_rate = st.number_input(
            "Contact Success Rate (0–1)",
            0.0, 1.0, 0.50, step=0.01,
            format="%.3f"
        )
        employment_type = st.selectbox(
            "Employment Type", EMPLOYMENT_OPTIONS
        )
        loan_purpose = st.selectbox(
            "Loan Purpose", PURPOSE_OPTIONS
        )

    submitted = st.form_submit_button(
        "Predict Recovery",
        type="primary",
        use_container_width=True,
    )


if submitted:
    values = {
        "age": float(age),
        "income": float(income),
        "credit_score": float(credit_score),
        "loan_amount": float(loan_amount),
        "loan_term_months": int(loan_term_months),
        "interest_rate": float(interest_rate),
        "debt_to_income": float(debt_to_income),
        "existing_loans": float(existing_loans),
        "previous_defaults": float(previous_defaults),
        "payment_delays": float(payment_delays),
        "employment_years": float(employment_years),
        "collateral_value": float(collateral_value),
        "contact_success_rate": float(contact_success_rate),
    }

    # One-hot encode categorical inputs.
    for option in EMPLOYMENT_OPTIONS:
        values[f"employment_type_{option}"] = int(
            employment_type == option
        )

    for option in PURPOSE_OPTIONS:
        values[f"loan_purpose_{option}"] = int(
            loan_purpose == option
        )

    # Preserve the model's exact feature order.
    row = pd.DataFrame([values]).reindex(
        columns=features, fill_value=0
    )

    try:
        prediction = int(model.predict(row)[0])

        probabilities = dict(
            zip(model.classes_, model.predict_proba(row)[0])
        )

        recovered_prob = float(probabilities.get(1, 0.0))
        not_recovered_prob = float(probabilities.get(0, 0.0))

        st.divider()
        st.subheader("Prediction Result")

        result_col, probability_col = st.columns(2)

        with result_col:
            if prediction == 1:
                st.success(
                    "Predicted outcome: Loan likely to be recovered"
                )
            else:
                st.error(
                    "Predicted outcome: Loan may not be recovered"
                )

            st.metric(
                "Predicted Class",
                "Recovered (1)"
                if prediction == 1
                else "Not Recovered (0)",
            )

        with probability_col:
            st.write(
                f"**Estimated recovery probability:** "
                f"{recovered_prob:.1%}"
            )
            st.progress(recovered_prob)

            st.write(
                f"**Estimated non-recovery probability:** "
                f"{not_recovered_prob:.1%}"
            )
            st.progress(not_recovered_prob)

        st.caption(
            "These are model estimates, not guarantees. "
            "Their reliability depends on the training data "
            "and model validation."
        )

        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "Feature": features,
                "Importance": model.feature_importances_,
            }).sort_values(
                "Importance", ascending=False
            ).head(10)

            st.subheader("Top Overall Feature Importances")
            st.bar_chart(
                importance.set_index("Feature")["Importance"]
            )
            st.caption(
                "These importances summarize the model overall. "
                "They do not establish causation or explain an "
                "individual prediction."
            )

        
        with st.expander("View Encoded Model Input"):
            st.dataframe(row, use_container_width=True)

        # SHAP explanation for this individual prediction
        st.divider()
        st.subheader("Explainable AI — SHAP")

        if shap is None:
            st.warning(
                "SHAP is not installed. Run: python -m pip install shap"
            )
        else:
            try:
                explainer = shap.TreeExplainer(model)
                explanation = explainer(row)

                shap_values = np.asarray(explanation.values)
                classes = list(model.classes_)
                recovery_index = classes.index(1)

                if shap_values.ndim == 3:
                    contributions = shap_values[
                        0, :, recovery_index
                    ]
                elif shap_values.ndim == 2:
                    contributions = shap_values[0, :]
                else:
                    raise ValueError(
                        f"Unexpected SHAP output shape: {shap_values.shape}"
                    )

                contribution_df = pd.DataFrame({
                    "Feature": features,
                    "SHAP Contribution": contributions
                })

                contribution_df["Absolute Contribution"] = (
                    contribution_df["SHAP Contribution"].abs()
                )

                top_contributions = (
                    contribution_df
                    .nlargest(10, "Absolute Contribution")
                    .sort_values("SHAP Contribution")
                )

                st.write(
                    "The chart shows which features influenced this "
                    "individual prediction most strongly."
                )

                st.bar_chart(
                    top_contributions.set_index("Feature")[
                        "SHAP Contribution"
                    ],
                    horizontal=True
                )

                st.dataframe(
                    top_contributions[
                        ["Feature", "SHAP Contribution"]
                    ].sort_values(
                        "SHAP Contribution",
                        ascending=False
                    ),
                    hide_index=True,
                    use_container_width=True
                )

                st.caption(
                    "Positive contributions push the model output toward "
                    "the recovery class; negative contributions push it "
                    "away. These are model explanations, not causal "
                    "relationships or guarantees."
                )

            except Exception as exc:
                st.warning(
                    f"SHAP explanation could not be generated: {exc}"
                )


        # Recovery Strategy Insights
        st.divider()
        st.subheader("💡 Recovery Strategy Insights")

        st.write(
            "Suggested follow-up actions based on the model's "
            "estimated recovery probability and borrower inputs."
        )

        # Recovery outlook
        if recovered_prob >= 0.70:
            st.success("Recovery Outlook: Higher estimated probability")

            st.write(
                "The model estimates a relatively higher likelihood "
                "of recovery."
            )

            st.markdown("""
            **Suggested actions:**
            - Send a respectful payment reminder.
            - Confirm the preferred payment channel.
            - Continue routine repayment monitoring.
            """)

        elif recovered_prob >= 0.40:
            st.warning("Recovery Outlook: Moderate estimated probability")

            st.write(
                "The model indicates an intermediate recovery outlook. "
                "Additional review may be useful."
            )

            st.markdown("""
            **Suggested actions:**
            - Review the borrower's recent payment history.
            - Understand possible reasons for payment delays.
            - Discuss a manageable repayment arrangement.
            - Schedule an appropriate follow-up.
            """)

        else:
            st.error("Recovery Outlook: Lower estimated probability")

            st.write(
                "The model estimates a lower likelihood of recovery. "
                "Consider a careful human review before deciding "
                "on the next steps."
            )

            st.markdown("""
            **Suggested actions:**
            - Verify the accuracy of borrower and loan information.
            - Understand potential financial difficulties.
            - Explore affordable repayment or restructuring options.
            - Consider appropriate hardship support.
            """)

        # Identify input-based indicators
        st.subheader("Borrower Indicators to Review")

        indicators_found = False

        if payment_delays >= 3:
            indicators_found = True
            st.info(
                "Payment history: Multiple payment delays were entered. "
                "Review recent repayment patterns."
            )

        if previous_defaults > 0:
            indicators_found = True
            st.info(
                "Previous defaults: Earlier defaults were reported. "
                "Review their circumstances and subsequent repayment history."
            )

        if debt_to_income >= 0.40:
            indicators_found = True
            st.info(
                "Debt-to-income ratio: The entered ratio is relatively high. "
                "Review repayment affordability."
            )

        if contact_success_rate < 0.30:
            indicators_found = True
            st.info(
                "Contact success: Previous contact success is low. "
                "Verify the customer's preferred communication channel."
            )

        if not indicators_found:
            st.success(
                "No indicators meeting the current review rules "
                "were identified."
            )

        st.caption(
            "These are illustrative decision-support rules, not "
            "validated collection policies. Probability bands and "
            "indicator thresholds require business validation. "
            "All actions require human review and must respect "
            "customer rights and applicable regulations."
        )



    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


# Model Performance Dashboard
st.divider()
st.header("📊 Model Performance Dashboard")

if not PROCESSED_DATA_PATH.exists():
    st.warning(
        f"Test dataset not found: {PROCESSED_DATA_PATH}"
    )
else:
    try:
        test_artifact = load_artifact(
            str(PROCESSED_DATA_PATH)
        )

        X_test = test_artifact["X_test"].reindex(
            columns=features,
            fill_value=0
        )
        y_test = test_artifact["y_test"]

        y_pred = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        classes = list(model.classes_)
        recovery_index = classes.index(1)
        recovery_scores = probabilities[:, recovery_index]

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(
            y_test, y_pred,
            pos_label=1,
            zero_division=0
        )
        roc_auc = roc_auc_score(
            y_test, recovery_scores
        )

        st.caption(
            "Metrics are calculated on the saved test set "
            "using the loaded model."
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Accuracy",
            f"{accuracy:.1%}"
        )
        metric2.metric(
            "Recall",
            f"{recall:.1%}"
        )
        metric3.metric(
            "ROC-AUC",
            f"{roc_auc:.3f}"
        )

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual: Not Recovered (0)",
                "Actual: Recovered (1)"
            ],
            columns=[
                "Predicted: Not Recovered (0)",
                "Predicted: Recovered (1)"
            ]
        )

        st.dataframe(
            cm_df,
            use_container_width=True
        )

        st.caption(
            "Rows represent actual classes; columns represent "
            "predicted classes. Recall measures the proportion "
            "of actual recovered cases correctly identified."
        )

        st.subheader("ROC Curve")

        fpr, tpr, _ = roc_curve(
            y_test,
            recovery_scores,
            pos_label=1
        )

        
        # Improved ROC Curve Visualization
        st.subheader("ROC Curve")

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            fpr,
            tpr,
            label=f"Random Forest (AUC = {roc_auc:.3f})",
            linewidth=2
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Random Baseline"
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Loan Recovery Model")

        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)


        st.caption(
            "A higher ROC-AUC generally indicates better "
            "discrimination between the two classes. "
            "It does not guarantee real-world performance."
        )

    except Exception as exc:
        st.error(
            f"Could not calculate model metrics: {exc}"
        )


st.divider()
st.caption(
    "Loan Recovery Prediction | Random Forest Classification"
)
