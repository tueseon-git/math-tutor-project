# NJ Private AI Math Tutor

Learn. Play. Grow with Math!

A private AI-powered multiplication tutor for children.

## Features

- Learn a multiplication table
- Ask for a memory trick
- Take a multiplication quiz
- Explain a student's answer
- Local private LLM using Ollama
- Private document knowledge
- FastAPI backend
- GoDaddy-compatible HTML frontend

## Architecture

GoDaddy Frontend  
↓  
FastAPI API on Mac  
↓  
Private document retrieval  
↓  
Ollama local LLM  
↓  
Child-friendly response

## Privacy

The private knowledge document, environment variables and Ollama
models remain on the local Mac and are not uploaded to GitHub.

## API

### Endpoint

`POST /api/v1/math`

### Example request

```json
{
  "mode": "learn",
  "table": 7,
  "message": "Teach me the table of 7.",
  "question": "",
  "student_answer": ""
}