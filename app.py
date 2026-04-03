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
    "python", "java", "c++", "javascript", "typescript",
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


_SKILL_PATTERNS = {
    skill: re.compile(rf"(?<![a-zA-Z0-9]){re.escape(skill)}(?![a-zA-Z0-9])")
    for skill in ALL_SKILLS
}

SKILL_WEIGHTS = {


    "python": 2.5, "java": 2.0, "c++": 2.0, "javascript": 1.8,
    "typescript": 1.8, "go": 1.8, "rust": 1.8, "kotlin": 1.8,
    "swift": 1.8, "php": 1.5, "ruby": 1.5, "scala": 1.8,
    "bash": 1.2, "matlab": 1.5,


    "flask": 2.5, "django": 2.5, "fastapi": 2.5,
    "spring": 2.2, "spring boot": 2.2,
    "nodejs": 2.0, "express": 2.0,
    "react": 1.8, "angular": 1.8, "vue": 1.8,
    "html": 1.2, "css": 1.2, "bootstrap": 1.0, "tailwind": 1.0,
    "rest api": 2.3, "graphql": 2.0,


    "mysql": 2.0, "postgresql": 2.0, "mongodb": 2.0,
    "sqlite": 1.8, "redis": 1.8, "oracle": 1.8,
    "firebase": 1.5, "dynamodb": 1.8,


    "machine learning": 2.5, "deep learning": 2.5,
    "nlp": 2.3, "computer vision": 2.3,
    "data analysis": 2.0, "data science": 2.2,
    "scikit-learn": 2.3, "tensorflow": 2.3, "pytorch": 2.3,
    "pandas": 2.0, "numpy": 2.0,
    "matplotlib": 1.5, "seaborn": 1.5,


    "git": 1.8, "github": 1.8,
    "docker": 2.0, "kubernetes": 2.2,
    "linux": 1.8,
    "aws": 2.2, "azure": 2.2, "gcp": 2.2,
    "ci cd": 1.8, "jenkins": 1.8,


    "data structures": 1.2, "algorithms": 1.2,
    "oops": 1.2,
    "operating systems": 1.2,
    "dbms": 1.2,
    "computer networks": 1.2,
    "system design": 2.0
}


def generate_feedback(resume_skills, job_skills):
    matched = set(resume_skills) & set(job_skills)
    missing = set(job_skills) - set(resume_skills)

    return {
        "matched": list(matched),
        "missing": list(missing)
    }


# def skill_matching(resume_skills, job_skills):
#     match = set(resume_skills) & set(job_skills)
#     if len(match) == 0:
#         return 0.0, set()
#     return (len(match)/len(job_skills)), match


def weighted_skill_score(resume_skills, job_skills):
    score = 0.0
    total = 0.0

    resume_set = set(resume_skills)
    job_set = set(job_skills)
    matched = resume_set & job_set

    for skill in job_skills:
        weight = float(SKILL_WEIGHTS.get(skill, 1.0))
        total += weight
        if skill in resume_set:
            score += weight

    if total == 0.0:
        return 0.0, matched
    return (score / total), matched


def extract_skills(text):
    text = text.lower()
    found = []
    for skill, pattern in _SKILL_PATTERNS.items():
        if pattern.search(text):
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
    match, _lst = weighted_skill_score(resume_skills, job_skills)
    final_score = 0.3 * sim + 0.7 * match

    feedback = generate_feedback(resume_skills, job_skills)
    return feedback, final_score

# def find_similarity(filepath, job_desc):
#     reader = PdfReader(filepath)
#     number_of_pages = len(reader.pages)
#     resume_text = ""
#     for page in reader.pages:
#         resume_text += page.extract_text() or ""

#     job = remove_stop_words(job_desc)
#     resume = remove_stop_words(resume_text)
#     sim = compute_similarity(job, resume)

#     resume_skills = extract_skills(resume)
#     job_skills = extract_skills(job)   
#     match, lst = weighted_skill_score(resume_skills, job_skills)
#     # lst is set
#     final_score = 0.4 * sim + 0.6 * match

#     feedback = generate_feedback(resume_skills, job_skills)

#     return feedback, final_score

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