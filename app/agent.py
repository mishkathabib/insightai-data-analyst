import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def decide_tool(question: str, columns: list[str]) -> str:
    prompt = f"""
You are an AI data analyst agent.

User question:
{question}

Dataset columns:
{columns}

Choose exactly one tool:
summary
missing
columns
types
statistics
correlation
histogram
classification
sql
unknown

Return only the tool name.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response.choices[0].message.content.strip().lower()

def explain_result(question: str, tool: str, result) -> str:
    prompt = f"""
You are an AI data analyst explaining results to a non-technical user.

User question:
{question}

Tool used:
{tool}

Raw result:
{result}

Explain the result clearly in 3-6 sentences.
Mention the most important numbers or patterns.
Do not overstate conclusions.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()

def generate_sql_query(question: str, columns: list[str]) -> str:
    prompt = f"""
You are a SQL assistant.

The user asked:
{question}

The SQLite table is named:
uploaded_data

Available columns:
{columns}

Write one SQLite SELECT query that answers the question.

Rules:
- Return only SQL.
- Use the table name uploaded_data.
- Do not use markdown.
- Do not modify the database.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response.choices[0].message.content.strip()