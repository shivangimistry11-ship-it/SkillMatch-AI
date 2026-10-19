# SkillMatch AI

An educational NLP prototype that compares a resume with a job description, detects catalog skills, calculates text similarity, and explains possible skill gaps.

## Features
- Extract text from selectable-text PDF resumes
- Detect skills from a customizable vocabulary
- Compare resume and job-description text using sentence embeddings and cosine similarity
- Fall back to TF-IDF cosine similarity if the embedding model cannot load
- List matching skills and skills mentioned in the job description but not detected in the resume
- Download a CSV skill-gap report

## Technology
Python, Streamlit, PyMuPDF, pandas, scikit-learn, Sentence Transformers.

## Setup (Windows PowerShell)
```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If `py` is unavailable, use `python` to create the virtual environment.

The first embedding-based analysis may download the `all-MiniLM-L6-v2` model, so an internet connection may be required. If model loading fails, the application attempts TF-IDF as a fallback.

## How to use
1. Start the application.
2. Upload a readable PDF resume.
3. Paste the target job description.
4. Click **Analyze Match**.
5. Review similarity, matching skills, skill gaps, and suggested next steps.
6. Download the CSV report if needed.

## Dataset and limitations
The initial `data/skills.csv` file is a manually assembled skill vocabulary for demonstrating the workflow, not a benchmark dataset. Expand and cite it if you incorporate an external dataset. Test with several resumes and job descriptions, and document limitations in your report.

The similarity percentage is a text-similarity score, not the probability of being hired. Skill detection depends on the vocabulary and exact phrase matching. Scanned PDFs, tables, abbreviations, and synonyms may not be handled correctly in this baseline. The tool must not use protected personal characteristics to make recommendations and should support human review.

## NLP techniques demonstrated
- Text extraction and preprocessing
- Skill phrase matching and normalization
- Sentence embeddings
- Cosine similarity
- Explainable skill-gap reporting

## Team contributions
Update this section with the actual contributions of each team member before submission.

## Academic integrity
Acknowledge external datasets, pretrained models, libraries, and sources used. Evaluate the system on a documented test set before making performance claims.
