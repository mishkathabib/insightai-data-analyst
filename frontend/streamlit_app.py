import streamlit as st
import pandas as pd
from app.tools.data_tools import (
    summarize_dataset,
    check_missing_values,
    get_data_types,
    describe_numeric,
    list_columns,
    correlation_with_target,
)
from app.tools.chart_tools import generate_histogram
from app.tools.ml_tools import train_classification_model
from app.tools.sql_tools import run_sql_query
from app.agent import decide_tool, explain_result, generate_sql_query

DEBUG = False

st.set_page_config(page_title="InsightAI", layout="wide")

st.title("InsightAI")
st.write("Upload a dataset and ask questions in plain English.")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Ask InsightAI")

    question = st.text_input(
        "Ask a question about your dataset"
    )

    if question:
        question_lower = question.lower()
        tool = decide_tool(question, list(df.columns))

        if DEBUG:
            st.caption(f"Tool selected: {tool}")

        if tool == "missing":
            st.write(check_missing_values(df))

        elif tool == "summary":
            result = summarize_dataset(df)
            st.write(result)
            explanation = explain_result(question, tool, result)
            st.info(explanation)

        elif tool == "columns":
            st.write(list_columns(df))

        elif tool == "types":
            st.write(get_data_types(df))

        elif tool == "statistics":
            st.write(describe_numeric(df))

        elif tool == "histogram":
            numeric_columns = df.select_dtypes(include="number").columns.tolist()

            selected_column = None
            for col in numeric_columns:
                if col.lower() in question_lower:
                    selected_column = col
                    break

            if selected_column:
                fig = generate_histogram(df, selected_column)
                st.pyplot(fig)
            else:
                st.write(f"Please mention a numeric column. Options: {numeric_columns}")

        elif tool == "correlation":
            numeric_columns = df.select_dtypes(include="number").columns.tolist()

            target = None
            for col in numeric_columns:
                if col.lower() in question_lower:
                    target = col
                    break

            if target:
                st.write(correlation_with_target(df, target))
            else:
                st.write(f"Please specify a numeric column. Options: {numeric_columns}")

        elif tool == "classification":
            target = None

            for col in df.columns:
                if col.lower() in question_lower:
                    target = col
                    break

            if target:
                result = train_classification_model(df, target)

                st.subheader("Model Comparison")

                comparison_df = pd.DataFrame(
                    result["model_results"].items(),
                    columns=["Model", "Accuracy"]
                )

                comparison_df["Accuracy"] = comparison_df["Accuracy"].apply(
                    lambda x: f"{x:.1%}"
                )

                st.dataframe(comparison_df, use_container_width=True)

                st.metric(
                    label="Best Model Accuracy",
                    value=f"{result['best_accuracy']:.1%}"
                )

                st.subheader("Best Model")
                st.write(result["best_model"])

                st.subheader("Most Important Features")
                st.dataframe(result["top_features"], use_container_width=True)

                explanation = explain_result(question, tool, result)
                st.info(explanation)
            else:
                st.write(f"Please specify a target column. Options: {list(df.columns)}")

        elif tool == "sql":
            sql_query = generate_sql_query(question, list(df.columns))

            st.write("Generated SQL")
            st.code(sql_query, language="sql")

            result = run_sql_query(df, sql_query)

            st.write("Query Results")
            st.dataframe(result)

            explanation = explain_result(question, tool, result)
            st.info(explanation)        

        else:
            st.write("I don't know how to answer that yet.")
