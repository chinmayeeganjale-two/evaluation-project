import os
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

app = FastAPI(title="AI Complaint Analysis Service")


# Groq API client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


# Input model
class ComplaintRequest(BaseModel):
    complaint: str


# Output model
class ComplaintResponse(BaseModel):
    issue: str
    severity: str
    sentiment: str
    recommended_action: str


# Q21 - Python implementation
def clean_complaint(complaint: str) -> str:
    if not complaint or not complaint.strip():
        raise ValueError("Complaint cannot be empty")

    return " ".join(complaint.strip().split())


# Q23 - Prompt design
def create_prompt(complaint: str) -> str:
    return f"""
Analyze the following customer complaint.

Return ONLY valid JSON with exactly these fields:
- issue
- severity
- sentiment
- recommended_action

Severity must be one of: Low, Medium, High.
Sentiment must be one of: Positive, Neutral, Negative.

Do not include any explanation or additional text.

Customer complaint:
{complaint}
"""


# Q22 + Q24 - FastAPI endpoint and LLM API integration
@app.post("/analyze-complaint", response_model=ComplaintResponse)
def analyze_complaint(data: ComplaintRequest):

    try:
        # Clean and validate complaint
        complaint = clean_complaint(data.complaint)

        # Create prompt
        prompt = create_prompt(complaint)

        # Call Groq LLM
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        # Get model response
        output = response.choices[0].message.content

        # Convert JSON string into Python dictionary
        result = json.loads(output)

        # Validate structured response using Pydantic
        return ComplaintResponse(**result)

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="LLM returned invalid JSON"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"API error: {str(e)}"
        )