# InsightAI

AI-powered data analyst agent that lets users upload a CSV, ask questions in natural language, generate SQL, create visualizations, and train machine learning models.
---

### Architecture 

<img width="1536" height="1024" alt="ChatGPT Image Jul 14, 2026, 10_22_45 AM" src="https://github.com/user-attachments/assets/b982ea75-dbd3-44f9-bf84-a8ceb4896554" />


### Home Page
<img width="2555" height="1434" alt="Screenshot 2026-07-14 at 10 13 40 AM" src="https://github.com/user-attachments/assets/6838080a-c504-409c-ae0c-3a5a26164c1f" />

### Dataset Upload
<img width="2558" height="1417" alt="Screenshot 2026-07-14 at 10 14 34 AM" src="https://github.com/user-attachments/assets/62fbecef-ae36-40cd-bb11-3a4b9b8c597a" />

### Machine Learning Model Comparison
<img width="2048" height="701" alt="Screenshot 2026-07-10 at 4 13 26 PM" src="https://github.com/user-attachments/assets/7649880b-c2f2-4ef7-9524-a0a70f148ff8" />

### Natural Language SQL
<img width="2048" height="346" alt="Screenshot 2026-07-10 at 4 13 47 PM" src="https://github.com/user-attachments/assets/a4580ac9-a8be-4434-bd4b-e645bcd4997d" />
---

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

- Python
- Streamlit
- FastAPI
- OpenAI API
- Pandas
- scikit-learn
- Matplotlib
- SQLite

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
```
