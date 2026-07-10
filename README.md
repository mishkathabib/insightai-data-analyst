# InsightAI

AI-powered data analyst agent that lets users upload a CSV, ask questions in natural language, generate SQL, create visualizations, and train machine learning models.

## Features

- CSV upload and dataset preview
- Dataset summaries and missing-value analysis
- Natural-language tool selection using an OpenAI model
- Histogram and correlation generation
- SQL generation and execution with SQLite
- Automated preprocessing
- Classification model comparison
- Feature importance
- Plain-English AI explanations

## Architecture

User → Streamlit → LLM Router → Python/SQL/ML Tools → AI Explanation

## Tech Stack

Python, Streamlit, FastAPI, OpenAI API, Pandas, scikit-learn, Matplotlib, SQLite

## Example Prompts

- Give me an overview of this dataset
- Which columns have missing values?
- Visualize the age distribution
- Show average fare by sex
- Train a model to predict survival

## Installation

```bash
git clone https://github.com/mishkathabib/insightai-data-analyst.git
cd insightai-data-analyst
python3 -m venv insightai_env
source insightai_env/bin/activate
pip install -r requirements.txt
