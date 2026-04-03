import io
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import re
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords

app = FastAPI()

# Allow browser-based frontends (including file:// during local dev) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

ps = PorterStemmer()


programming_skills = [
    "python", "java", "c", "c++", "javascript", "typescript",
    "go", "rust", "kotlin", "swift", "php", "ruby", "scala",
    "bash", "matlab"
]
web_skills = [
    "flask", "django", "fastapi", "spring", "spring boot",
    "nodejs", "express", "react", "angular", "vue",
    "html", "css", "bootstrap", "tailwind",
    "rest api", "graphql"
]
database_skills = [
    "mysql", "postgresql", "mongodb", "sqlite",
    "redis", "oracle", "firebase", "dynamodb"
]
ml_skills = [
    "machine learning", "deep learning", "nlp",
    "computer vision", "data analysis", "data science",
    "scikit-learn", "tensorflow", "pytorch",
    "pandas", "numpy", "matplotlib", "seaborn"
]
tools_skills = [
    "git", "github", "docker", "kubernetes",
    "linux", "aws", "azure", "gcp",
    "ci cd", "jenkins"
]
cs_skills = [
    "data structures", "algorithms", "oops",
    "operating systems", "dbms", "computer networks",
    "system design"
]
ALL_SKILLS = (
    programming_skills +
    web_skills +
    database_skills +
    ml_skills +
    tools_skills +
    cs_skills
)

def generate_feedback(resume_skills, job_skills):
    matched = set(resume_skills) & set(job_skills)
    missing = set(job_skills) - set(resume_skills)

    return {
        "matched": list(matched),
        "missing": list(missing)
    }


def skill_matching(resume_skills, job_skills):
    match = set(resume_skills) & set(job_skills)
    if len(match) == 0:
        return 0.0, set()
    return (len(match)/len(job_skills)), match


def extract_skills(text):
    found = []
    for skill in ALL_SKILLS:
        if skill in text:
            found.append(skill)
    return found

def compute_similarity(job, resume):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume, job])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return score


def remove_stop_words(text):
    text = text.lower()
    text = text.replace('\n',' ')
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)


def extract_text_from_bytes(file_bytes: bytes) -> str:
    pdf = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text


def find_similarity_from_text(resume_text: str, job_desc: str):
    job = remove_stop_words(job_desc)
    resume = remove_stop_words(resume_text)
    sim = compute_similarity(job, resume)

    resume_skills = extract_skills(resume)
    job_skills = extract_skills(job)
    match, _lst = skill_matching(resume_skills, job_skills)
    final_score = 0.6 * sim + 0.4 * match

    feedback = generate_feedback(resume_skills, job_skills)
    return feedback, final_score

def find_similarity(filepath, job_desc):
    reader = PdfReader(filepath)
    number_of_pages = len(reader.pages)
    resume_text = ""
    for page in reader.pages:
        resume_text += page.extract_text() or ""

    job = remove_stop_words(job_desc)
    resume = remove_stop_words(resume_text)
    sim = compute_similarity(job, resume)

    resume_skills = extract_skills(resume)
    job_skills = extract_skills(job)   
    match, lst = skill_matching(resume_skills, job_skills)
    # lst is set
    final_score = 0.4 * sim + 0.6 * match

    feedback = generate_feedback(resume_skills, job_skills)

    return feedback, final_score

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
def upload_files(
    file: UploadFile = File(...),
    job_desc: str = Form(...)
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file detected")

    content = file.file.read()  # bytes in memory
    resume_text = extract_text_from_bytes(content)
    feedback, final_score = find_similarity_from_text(resume_text, job_desc)
    return {"feedback": feedback, "final_score": final_score}

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run("app:app", host="0.0.0.0", port=port)