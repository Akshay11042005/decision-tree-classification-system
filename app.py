
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from algorithms import (
    GainRatioTreeClassifier,
    InformationGainTreeClassifier,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Decision Tree Classification System",
    page_icon="🌳",
    layout="wide",
)

st.title("🌳 Decision Tree Classification System")
st.caption(
    "Upload a CSV dataset, train ID3 and C4.5-style decision trees, "
    "visualize the trees, make predictions, and compare performance."
)

# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    uploaded = st.file_uploader(
        "Upload classification dataset",
        type=["csv"],
    )

    test_size = st.slider(
        "Test set percentage",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
    ) / 100

    random_state = st.number_input(
        "Random seed",
        min_value=0,
        max_value=99999,
        value=42,
    )

    max_depth = st.slider(
        "Maximum tree depth",
        min_value=1,
        max_value=20,
        value=8,
    )

    min_samples_leaf = st.slider(
        "Minimum samples per leaf",
        min_value=1,
        max_value=10,
        value=1,
    )

    st.info(
        "Both algorithms use the same train/test split "
        "and preprocessing for a fair comparison."
    )

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Unable to read CSV: {exc}")
        st.stop()

else:
    st.subheader("Get Started")

    st.write(
        "Upload a CSV file or load the built-in Iris "
        "dataset to test the application."
    )

    if st.button("Load Sample Dataset"):
        from sklearn.datasets import load_iris

        iris = load_iris(as_frame=True)
        data = iris.data.copy()
        data["species"] = [
            iris.target_names[i] for i in iris.target
        ]

        st.session_state["sample_df"] = data

    if "sample_df" not in st.session_state:
        st.info("Upload a CSV file or click Load Sample Dataset.")
        st.stop()

    data = st.session_state["sample_df"].copy()

# --------------------------------------------------
# VALIDATE DATASET
# --------------------------------------------------
if data.empty or len(data.columns) < 2:
    st.error(
        "The dataset must contain at least one feature "
        "and one target column."
    )
    st.stop()

# --------------------------------------------------
# APPLICATION TABS
# --------------------------------------------------
tab_data, tab_train, tab_predict, tab_docs = st.tabs(
    [
        "1. Dataset & Preprocessing",
        "2. Train & Evaluate",
        "3. Predict",
        "4. About",
    ]
)

# --------------------------------------------------
# DATASET TAB
# --------------------------------------------------
with tab_data:
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rows", f"{len(data):,}")
    col2.metric("Total Columns", len(data.columns))
    col3.metric(
        "Missing Cells",
        f"{int(data.isna().sum().sum()):,}",
    )

    st.subheader("Dataset Preview")
    st.dataframe(data.head(100), use_container_width=True)

    st.subheader("Column Information")

    summary = pd.DataFrame(
        {
            "Data Type": data.dtypes.astype(str),
            "Missing Values": data.isna().sum(),
            "Unique Values": data.nunique(dropna=True),
        }
    )

    st.dataframe(summary, use_container_width=True)

    target = st.selectbox(
        "Select Target / Class Column",
        options=list(data.columns),
        index=len(data.columns) - 1,
    )

    if data[target].dropna().nunique() < 2:
        st.warning(
            "The target column must contain at least two classes."
        )

    st.markdown(
        """
        **Preprocessing**
        - Numeric missing values: median imputation.
        - Categorical missing values: most-frequent imputation.
        - Categorical features: one-hot encoding.
        - Dataset: training and testing split.
        """
    )

# Reset saved models if the dataset or target changes.
dataset_signature = (
    tuple(data.columns),
    len(data),
    target,
    str(data[target].dtype),
)

if st.session_state.get("dataset_signature") != dataset_signature:
    st.session_state["trained"] = False
    st.session_state["dataset_signature"] = dataset_signature

if "trained" not in st.session_state:
    st.session_state["trained"] = False

# --------------------------------------------------
# TRAINING AND EVALUATION TAB
# --------------------------------------------------
with tab_train:
    st.subheader("Train Decision Tree Models")

    if st.button("Train ID3 and C4.5", type="primary"):
        clean = data.dropna(subset=[target]).copy()

        if clean[target].nunique() < 2:
            st.error(
                "Choose a target column with at least two classes."
            )
            st.stop()

        X = clean.drop(columns=[target])
        y = clean[target].astype(str)

        if X.shape[1] == 0:
            st.error("At least one feature column is required.")
            st.stop()

        if len(clean) < 5:
            st.error(
                "Please provide at least five valid data rows."
            )
            st.stop()

        numeric_cols = X.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_cols = [
            col for col in X.columns
            if col not in numeric_cols
        ]

        transformers = []

        if numeric_cols:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(strategy="median"),
                            )
                        ]
                    ),
                    numeric_cols,
                )
            )

        if categorical_cols:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="most_frequent"
                                ),
                            ),
                            (
                                "onehot",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=False,
                                ),
                            ),
                        ]
                    ),
                    categorical_cols,
                )
            )

        preprocessor = ColumnTransformer(
            transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

        counts = y.value_counts()
        stratify = y if counts.min() >= 2 else None

        try:
            (
                X_train_raw,
                X_test_raw,
                y_train,
                y_test,
            ) = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=int(random_state),
                stratify=stratify,
            )

            X_train = preprocessor.fit_transform(X_train_raw)
            X_test = preprocessor.transform(X_test_raw)

            feature_names = (
                preprocessor.get_feature_names_out().tolist()
            )

        except Exception as exc:
            st.error(f"Preprocessing failed: {exc}")
            st.stop()

        id3 = InformationGainTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )

        c45 = GainRatioTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )

        try:
            id3.fit(
                X_train,
                y_train.to_numpy(),
                feature_names,
            )

            c45.fit(
                X_train,
                y_train.to_numpy(),
                feature_names,
            )

            pred_id3 = id3.predict(X_test)
            pred_c45 = c45.predict(X_test)

        except Exception as exc:
            st.error(f"Model training failed: {exc}")
            st.stop()

        labels = sorted(y.unique().tolist())

        def calculate_metrics(name, predictions):
            return {
                "Model": name,
                "Accuracy": accuracy_score(
                    y_test, predictions
                ),
                "Precision (Weighted)": precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                "Recall (Weighted)": recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                "F1-Score (Weighted)": f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
            }

        results = pd.DataFrame(
            [
                calculate_metrics(
                    "ID3 (Information Gain)",
                    pred_id3,
                ),
                calculate_metrics(
                    "C4.5-style (Gain Ratio)",
                    pred_c45,
                ),
            ]
        )

        st.session_state.update(
            {
                "trained": True,
                "id3": id3,
                "c45": c45,
                "preprocessor": preprocessor,
                "feature_columns": X.columns.tolist(),
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "target": target,
                "labels": labels,
                "X_test": X_test,
                "y_test": y_test.to_numpy(),
                "pred_id3": pred_id3,
                "pred_c45": pred_c45,
                "results": results,
                "feature_names": feature_names,
            }
        )

        st.success("Both models trained successfully!")

    # DISPLAY RESULTS
    if st.session_state["trained"]:
        results = st.session_state["results"]

        st.subheader("Performance Comparison")

        st.dataframe(
            results.style.format(
                {
                    col: "{:.3f}"
                    for col in results.columns
                    if col != "Model"
                }
            ),
            use_container_width=True,
        )

        st.bar_chart(
            results.set_index("Model")
        )

        st.caption(
            "Precision, recall, and F1-score use weighted "
            "averaging to support multiclass datasets."
        )

        selected_model = st.radio(
            "Select a model to inspect",
            ["ID3", "C4.5-style"],
            horizontal=True,
        )

        if selected_model == "ID3":
            model = st.session_state["id3"]
            predictions = st.session_state["pred_id3"]
        else:
            model = st.session_state["c45"]
            predictions = st.session_state["pred_c45"]

        st.subheader(f"{selected_model} Decision Tree")

        fig, ax = plt.subplots(figsize=(18, 9))
        model.plot(ax=ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            st.session_state["y_test"],
            predictions,
            labels=st.session_state["labels"],
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=st.session_state["labels"],
        ).plot(
            ax=ax,
            xticks_rotation=45,
            colorbar=False,
        )

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("Classification Report")

        st.code(
            classification_report(
                st.session_state["y_test"],
                predictions,
                labels=st.session_state["labels"],
                zero_division=0,
            )
        )

        st.download_button(
            "Download Comparison Metrics (CSV)",
            data=results.to_csv(index=False),
            file_name="model_comparison.csv",
            mime="text/csv",
        )

# --------------------------------------------------
# PREDICTION TAB
# --------------------------------------------------
with tab_predict:
    st.subheader("Predict a Class for a New Example")

    if not st.session_state["trained"]:
        st.info(
            "Train both models in the Train & Evaluate tab first."
        )

    else:
        input_values = {}
        cols = st.columns(2)

        for i, col in enumerate(
            st.session_state["feature_columns"]
        ):
            with cols[i % 2]:
                if col in st.session_state["numeric_cols"]:
                    values = pd.to_numeric(
                        data[col], errors="coerce"
                    ).dropna()

                    default = (
                        float(values.median())
                        if not values.empty
                        else 0.0
                    )

                    input_values[col] = st.number_input(
                        col,
                        value=default,
                        key=f"predict_{col}",
                    )

                else:
                    options = (
                        data[col]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    if options:
                        input_values[col] = st.selectbox(
                            col,
                            options,
                            key=f"predict_{col}",
                        )
                    else:
                        input_values[col] = ""

        if st.button("Predict Class", type="primary"):
            row = pd.DataFrame(
                [input_values],
                columns=st.session_state["feature_columns"],
            )

            try:
                encoded = (
                    st.session_state["preprocessor"]
                    .transform(row)
                )

                p1 = st.session_state["id3"].predict(
                    encoded
                )[0]

                p2 = st.session_state["c45"].predict(
                    encoded
                )[0]

                col1, col2 = st.columns(2)

                col1.metric("ID3 Prediction", str(p1))
                col2.metric("C4.5-style Prediction", str(p2))

                if p1 == p2:
                    st.success(
                        "Both models predict the same class."
                    )
                else:
                    st.warning(
                        "The models predict different classes."
                    )

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

# --------------------------------------------------
# DOCUMENTATION TAB
# --------------------------------------------------
with tab_docs:
    st.subheader("Algorithms and Implementation")

    st.markdown(
        """
        **ID3**
        - Uses entropy and information gain.
        - Chooses the split with the greatest information gain.

        **C4.5-style**
        - Uses information gain ratio.
        - Helps reduce the preference for splits with many outcomes.

        **Preprocessing**
        - Median imputation for numerical columns.
        - Most-frequent imputation and one-hot encoding for
          categorical columns.

        **Evaluation**
        - Accuracy
        - Weighted precision
        - Weighted recall
        - Weighted F1-score
        - Confusion matrix and classification report

        **Implementation limitation**

        These are educational implementations of information-gain
        and gain-ratio decision trees. The C4.5-style algorithm is
        not a complete reproduction of the original C4.5 algorithm,
        which includes additional features such as pruning and
        specialized handling of continuous and missing values.
        """
    )

    st.subheader("GitHub Repository")

    st.markdown(
        "[Open your GitHub profile](https://github.com/Akshay11042005)"
    )

    st.code(
        """git init
git add .
git commit -m "Initial decision tree classification system"
git branch -M main
git remote add origin https://github.com/Akshay11042005/decision-tree-classification-system.git
git push -u origin main""",
        language="bash",
    )
