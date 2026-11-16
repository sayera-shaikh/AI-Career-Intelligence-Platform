# AI Career Intelligence Platform

An AI-assisted career guidance and first-stage resume screening application built with Python and Streamlit.

> **Certification project:** CPA 2  
> **Status:** Functional prototype; validated manually across the Candidate and Recruiter portals.

## Overview

The platform brings two workflows into one web application:

### Candidate Portal
- Select a target career role from 24 available profiles.
- Upload a PDF resume and extract its text.
- Identify skills and calculate a role skill-match score.
- View missing skills, a learning roadmap, and role-specific resume improvement tips.
- Ask career questions using the AI Career Assistant, with voice input and text-to-speech support.

### Recruiter Portal
- Enter or dictate a job description.
- Upload multiple PDF resumes.
- Extract and compare required and candidate skills.
- Rank resumes using a transparent skill-overlap score.
- Adjust a shortlisting threshold and export results to CSV.

## How it works

1. The user supplies a PDF resume or job description.
2. The app extracts readable PDF text.
3. Rule-based skill dictionaries and aliases identify skills.
4. The app compares detected skills with role/JD requirements and calculates a percentage match.
5. Candidate guidance or recruiter ranking is displayed.
6. The Career Assistant uses `all-MiniLM-L6-v2` embeddings with FAISS similarity retrieval over the bundled knowledge base.

**Important:** Match scores are skill-overlap indicators. They are not predictions of job performance and should not be used as the sole basis for hiring decisions.

## Technology stack

- Python
- Streamlit
- Pandas and NumPy
- PyPDF
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- `streamlit-mic-recorder`

## Repository structure

```text
.
├── app.py
├── data/
│   └── knowledge_base_v3_expanded.pkl
├── docs/
│   └── AI_Career_Intelligence_Platform_CPA2_Presentation.pptx
├── assets/                 # Add application screenshots here
├── requirements.txt
├── README.md
└── .gitignore
```

## Run locally

### 1. Create and activate a virtual environment

**Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start the application

```bash
streamlit run app.py
```

The app will open in your browser, usually at `http://localhost:8501`.

On first launch, Sentence Transformers may download the `all-MiniLM-L6-v2` model. An internet connection is needed for that initial download unless the model is already cached.

## Data and privacy

The application requires the bundled `data/knowledge_base_v3_expanded.pkl` file for its Career Assistant.

The original resume-ranking CSV and `Resume.zip` from the development folder are **not included** in this public-ready package. The CSV contains fields such as addresses and online links, and the ZIP contains resume files. These may contain personal information. Do not publish real candidate resumes or personally identifiable information. If a public sample dataset is needed, use a properly licensed, anonymized dataset and document its source and license.

Uploaded resumes should be handled securely and deleted when no longer needed. Do not commit API keys, passwords, `.env` files, private resumes, or confidential data.

## Validation notes

Manual functional checks have covered:
- Candidate PDF upload and resume analysis
- Skill match, missing skills, learning roadmap, and resume tips
- Career Assistant text interaction and voice features
- Recruiter job-description input and multiple PDF uploads
- Candidate ranking, adjustable shortlisting threshold, and CSV export

Testing used sample inputs. Results depend on PDF text quality and the skill vocabulary configured in the application. Scanned PDFs may require OCR. Skill matching is primarily keyword/alias based.

## Limitations and future improvements

- Add OCR support for scanned resumes.
- Improve semantic resume-to-job-description matching.
- Expand and validate skill aliases across more job descriptions.
- Add secure user accounts and controlled storage.
- Integrate verified live job data sources.
- Expand multilingual career assistance and interview practice.

## Responsible use

This prototype is intended to assist career exploration and organize initial resume review. Recruiters must review the underlying resumes and make fair, human-led decisions. Candidates' personal information must be protected.

## Project presentation

See `docs/AI_Career_Intelligence_Platform_CPA2_Presentation.pptx` for the CPA 2 project presentation.

## Author

**Sayera Bano A Nazir Shaikh**  
CPA 2 Certification Project
