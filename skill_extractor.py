
import re
import unicodedata
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# SKILL ALIASES
# Canonical skill name -> accepted aliases.
# Closely related skills remain separate.
# =========================================================

SKILL_ALIASES = {
    # Programming languages
    "Python": ["python programming", "python programming language"],
    "Java": ["java programming"],
    "C++": ["cpp", "cplusplus"],
    "C": ["c language", "programming in c"],
    "C#": ["c sharp", "c-sharp"],
    "PHP": ["php programming", "php language"],
    "JavaScript": ["js", "ecmascript"],
    "TypeScript": ["ts", "typescript programming"],
    "Dart": ["dart language"],
    "R": ["r programming", "r language"],
    "Go": ["golang", "go programming language"],
    "Ruby": ["ruby programming"],
    "Kotlin": ["kotlin programming"],
    "Swift": ["swift programming"],

    # Databases
    "SQL": [
        "structured query language",
        "sql queries",
        "writing sql queries",
    ],
    "MySQL": ["mysql database", "mysql server"],
    "PostgreSQL": ["postgres", "postgresql database", "postgre sql"],
    "SQLite": ["sqlite database"],
    "Microsoft SQL Server": ["ms sql server", "mssql", "sql server"],
    "Oracle Database": ["oracle db"],
    "MongoDB": ["mongo db", "mongodb database"],
    "Redis": ["redis database", "redis cache"],
    "Firebase": ["google firebase", "firebase database"],
    "Elasticsearch": ["elastic search"],

    # Backend and web development
    "Django": ["django framework", "django web framework"],
    "Flask": ["flask framework"],
    "FastAPI": ["fast api"],
    "REST API": [
        "rest apis",
        "restful api",
        "restful apis",
        "rest api development",
        "restful web services",
    ],
    "GraphQL": ["graph ql"],
    "Node.js": ["nodejs", "node js"],
    "Express.js": ["expressjs", "express js"],
    "Spring Boot": ["springboot", "spring boot framework"],
    "ASP.NET": ["asp net", "asp.net core"],
    "Laravel": ["laravel framework"],
    "HTML": ["html5"],
    "CSS": ["css3", "cascading style sheets"],
    "Sass": ["scss", "syntactically awesome stylesheets"],
    "React": ["react.js", "reactjs"],
    "Angular": ["angular.js", "angularjs"],
    "Vue.js": ["vuejs", "vue js"],
    "Bootstrap": ["bootstrap css"],
    "Tailwind CSS": ["tailwind", "tailwindcss"],
    "jQuery": [],

    # IDEs and development tools
    "Visual Studio Code": ["vs code", "vscode", "visual studio code editor"],
    "Visual Studio": ["visual studio ide"],
    "Android Studio": ["android studio ide"],
    "Jupyter Notebook": ["jupyter", "jupyter notebooks", "ipython notebook"],
    "Git": ["git version control", "version control with git"],
    "GitHub": ["git hub", "github repositories"],
    "GitLab": ["git lab"],
    "Bitbucket": ["bit bucket"],
    "Postman": ["postman api client"],
    "Swagger": ["swagger ui", "openapi swagger"],
    "Jira": ["atlassian jira"],
    "Confluence": ["atlassian confluence"],

    # DevOps and cloud
    "CI/CD": [
        "continuous integration",
        "continuous delivery",
        "continuous deployment",
        "ci cd",
        "ci-cd",
        "cicd",
    ],
    "Jenkins": ["jenkins pipeline", "jenkins ci"],
    "Docker": ["docker containers", "containerization with docker"],
    "Kubernetes": ["k8s", "kubernetes orchestration"],
    "Terraform": ["terraform iac", "infrastructure as code"],
    "Ansible": ["ansible automation"],
    "GitHub Actions": ["github action", "gh actions"],
    "GitLab CI/CD": ["gitlab ci", "gitlab ci cd", "gitlab pipelines"],
    "Linux": ["linux operating system"],
    "Bash": ["bash scripting", "shell scripting"],
    "PowerShell": ["powershell scripting"],
    "Nginx": ["engine x"],
    "Apache HTTP Server": ["apache server", "apache httpd"],
    "AWS": ["amazon web services"],
    "Microsoft Azure": ["azure cloud"],
    "Google Cloud": ["gcp", "google cloud platform"],
    "Cloud Computing": ["cloud platforms", "cloud services"],
    "Serverless": ["serverless computing", "serverless architecture"],

    # Monitoring and testing
    "Monitoring": [
        "system monitoring",
        "application monitoring",
        "infrastructure monitoring",
        "performance monitoring",
    ],
    "Logging": ["application logging", "log management", "log analysis"],
    "Prometheus": ["prometheus monitoring"],
    "Grafana": ["grafana dashboards"],
    "Datadog": ["datadog monitoring"],
    "Splunk": ["splunk logging", "splunk observability"],
    "ELK Stack": ["elastic stack"],
    "OpenTelemetry": ["otel", "open telemetry"],
    "Unit Testing": ["unit tests", "unit test"],
    "Integration Testing": ["integration tests", "integration test"],
    "PyTest": ["python pytest"],
    "Selenium": ["selenium webdriver", "selenium testing"],
    "Robot Framework": ["robotframework"],
    "Jest": ["jest testing framework"],
    "CI Testing": ["continuous integration testing"],

    # Data science, AI, and NLP
    "Machine Learning": ["ml", "machine-learning"],
    "Deep Learning": ["dl", "deep neural learning"],
    "Natural Language Processing": ["nlp", "natural-language processing"],
    "Computer Vision": ["image recognition"],
    "Generative AI": ["gen ai", "genai", "generative artificial intelligence"],
    "Large Language Models": ["llm", "llms", "large language model"],
    "Retrieval-Augmented Generation": ["rag", "retrieval augmented generation"],
    "TensorFlow": ["tensor flow"],
    "PyTorch": ["py torch"],
    "Scikit-learn": ["scikit learn", "sklearn"],
    "Pandas": ["python pandas"],
    "NumPy": ["num py"],
    "Matplotlib": ["mat plot lib"],
    "Seaborn": ["seaborn library"],
    "OpenCV": ["open cv"],
    "spaCy": ["spacy nlp"],
    "NLTK": ["natural language toolkit"],
    "Hugging Face": ["huggingface", "hugging face transformers"],
    "Sentence Transformers": ["sentence transformer", "sbert"],
    "FAISS": ["facebook ai similarity search"],
    "Vector Database": ["vector databases", "vector store"],
    "Data Analysis": ["data analytics", "data analysing", "data analyzing"],
    "Data Visualization": ["data visualisation", "data viz"],
    "Power BI": ["powerbi", "microsoft power bi"],
    "Tableau": ["tableau software"],
    "Excel": ["microsoft excel", "ms excel"],
    "Statistics": ["statistical analysis"],
    "Big Data": ["big data analytics"],
    "Apache Hadoop": ["hadoop"],
    "Apache Spark": ["spark framework", "pyspark"],
    "ETL": ["extract transform load", "data pipelines"],

    # Software engineering
    "Data Structures": ["data structure"],
    "Algorithms": ["algorithm design"],
    "Object-Oriented Programming": ["oop", "object oriented programming"],
    "Software Development Life Cycle": ["sdlc", "software development lifecycle"],
    "Agile": ["agile methodology", "agile methodologies"],
    "Scrum": ["scrum framework"],
    "Problem Solving": ["problem-solving"],
    "Communication": ["communication skills"],
    "Teamwork": ["team work", "collaboration"],
    "Streamlit": [],
}


# =========================================================
# SKILL-SPECIFIC RECOMMENDATIONS
# =========================================================

SKILL_RECOMMENDATIONS = {
    "python": (
        "Practise functions, collections, file handling, and object-oriented "
        "programming. Build a small Python application."
    ),
    "java": (
        "Practise Java classes, inheritance, exception handling, collections, "
        "and object-oriented programming."
    ),
    "c++": (
        "Practise C++ functions, pointers, classes, STL, and data structures."
    ),
    "sql": (
        "Practise SELECT, JOIN, GROUP BY, subqueries, and database design."
    ),
    "mysql": (
        "Create tables and practise CRUD operations, joins, relationships, "
        "and query optimization."
    ),
    "mongodb": (
        "Practise documents, collections, CRUD operations, filtering, "
        "and aggregation pipelines."
    ),
    "machine learning": (
        "Build a supervised-learning project and evaluate it using "
        "accuracy, precision, recall, and F1-score."
    ),
    "deep learning": (
        "Study neural networks, activation functions, and backpropagation. "
        "Train a simple deep-learning model."
    ),
    "tensorflow": (
        "Build a basic neural network using TensorFlow and Keras, "
        "then evaluate it on a classification dataset."
    ),
    "pytorch": (
        "Practise tensors, datasets, loss functions, and optimizers "
        "by training a classification model."
    ),
    "scikit-learn": (
        "Practise preprocessing, train-test splitting, classification, "
        "regression, and model evaluation."
    ),
    "computer vision": (
        "Build an image-classification project and learn image "
        "preprocessing, OpenCV, and basic CNN concepts."
    ),
    "natural language processing": (
        "Build a text-classification project using tokenization, TF-IDF, "
        "text preprocessing, and model evaluation."
    ),
    "nltk": (
        "Practise tokenization, stemming, lemmatization, stop-word removal, "
        "and text analysis with NLTK."
    ),
    "spacy": (
        "Build a text-processing application using spaCy tokenization "
        "and named entity recognition."
    ),
    "generative ai": (
        "Build a small generative-AI application and study prompt design, "
        "model limitations, and responsible AI."
    ),
    "large language models": (
        "Explore tokenization and embeddings, then build a small "
        "LLM-based question-answering application."
    ),
    "pandas": (
        "Practise filtering, grouping, merging, and cleaning datasets "
        "with Pandas."
    ),
    "numpy": (
        "Practise NumPy arrays, indexing, broadcasting, and numerical "
        "operations."
    ),
    "data analysis": (
        "Clean a dataset, calculate summary statistics, and communicate "
        "findings with charts."
    ),
    "data visualization": (
        "Create clear charts and dashboards to communicate data patterns."
    ),
    "streamlit": (
        "Build an interactive Streamlit application with inputs, charts, "
        "data tables, and downloadable reports."
    ),
    "aws": (
        "Learn core AWS services and deploy a small application using "
        "an appropriate AWS service."
    ),
    "microsoft azure": (
        "Learn Azure fundamentals and deploy a small application "
        "using an appropriate cloud service."
    ),
    "google cloud": (
        "Learn Google Cloud fundamentals and deploy a small application."
    ),
    "cloud computing": (
        "Study cloud service models, deployment models, scalability, "
        "and cloud deployment."
    ),
    "linux": (
        "Practise shell commands, file permissions, processes, "
        "package management, and troubleshooting."
    ),
    "docker": (
        "Write a Dockerfile, build an image, and run a container "
        "for a small Python application."
    ),
    "kubernetes": (
        "Learn Pods, Deployments, and Services, then deploy a "
        "containerized application in a practice environment."
    ),
    "ci/cd": (
        "Create a CI/CD pipeline that runs automated tests and builds "
        "an application after a code change."
    ),
    "jenkins": (
        "Create a Jenkins pipeline that checks out code, runs tests, "
        "and builds a small application."
    ),
    "github actions": (
        "Create a GitHub Actions workflow that automatically runs tests "
        "when code is pushed."
    ),
    "terraform": (
        "Learn infrastructure as code and practise creating resources "
        "with Terraform."
    ),
    "ansible": (
        "Write Ansible playbooks to automate configuration and "
        "repetitive administration tasks."
    ),
    "monitoring": (
        "Practise monitoring application health, response times, errors, "
        "and resource usage."
    ),
    "logging": (
        "Add structured logging to an application and use logs "
        "to diagnose errors."
    ),
    "prometheus": (
        "Collect application metrics with Prometheus and practise "
        "writing basic PromQL queries."
    ),
    "grafana": (
        "Create a Grafana dashboard using sample metrics and "
        "configure useful visualizations."
    ),
    "django": (
        "Build a Django application with models, views, forms, "
        "database operations, and authentication."
    ),
    "flask": (
        "Build a Flask application with routes, templates, and API endpoints."
    ),
    "fastapi": (
        "Create a FastAPI service with request validation, documented "
        "endpoints, and automated tests."
    ),
    "rest api": (
        "Build REST endpoints, validate requests, handle errors, "
        "and test endpoints with an API client."
    ),
    "html": (
        "Create accessible web pages using semantic HTML elements "
        "and well-structured forms."
    ),
    "css": (
        "Practise layouts, Flexbox, Grid, responsive design, and reusable CSS."
    ),
    "react": (
        "Build a React application using components, props, state, hooks, "
        "and API requests."
    ),
    "angular": (
        "Build an Angular application using components, services, routing, "
        "and forms."
    ),
    "git": (
        "Practise commits, branches, merges, and resolving merge conflicts."
    ),
    "github": (
        "Create a repository, write a README, use branches, "
        "and practise pull requests."
    ),
    "visual studio code": (
        "Practise debugging, extensions, integrated terminals, "
        "and Git integration in VS Code."
    ),
    "android studio": (
        "Create and run a small Android application and practise debugging."
    ),
    "selenium": (
        "Automate a browser workflow with Selenium and add assertions "
        "to verify expected results."
    ),
    "robot framework": (
        "Create a Robot Framework test suite with reusable keywords, "
        "test data, and pass/fail checks."
    ),
    "unit testing": (
        "Write unit tests for individual functions, including normal cases, "
        "edge cases, and failures."
    ),
    "communication": (
        "Practise explaining technical concepts clearly and presenting "
        "a project in a short demonstration."
    ),
    "problem solving": (
        "Solve programming problems regularly and explain your approach, "
        "complexity, and test cases."
    ),
    "data structures": (
        "Practise arrays, linked lists, stacks, queues, trees, hash tables, "
        "and their common operations."
    ),
    "algorithms": (
        "Practise searching, sorting, recursion, and algorithm complexity."
    ),
}


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):
    """Normalize Unicode, case, and whitespace."""
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = text.replace("–", "-").replace("—", "-").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def compile_phrase(alias):
    """Match a phrase with flexible spaces or hyphens."""
    alias = normalize_text(alias)
    parts = re.split(r"[\s-]+", alias)
    phrase = r"[\s-]+".join(
        re.escape(part) for part in parts if part
    )
    return re.compile(r"(?<!\w)" + phrase + r"(?!\w)", re.IGNORECASE)


@lru_cache(maxsize=1)
def get_compiled_aliases():
    """Compile all skill aliases once for efficient extraction."""
    compiled = {}

    for canonical, aliases in SKILL_ALIASES.items():
        terms = set([canonical] + aliases)
        compiled[canonical] = [
            compile_phrase(term)
            for term in terms
            if term.strip()
        ]

    return compiled


# =========================================================
# SKILL EXTRACTION
# =========================================================

def extract_skills(text, skill_catalog=None):
    """
    Extract canonical skill names using aliases and word boundaries.

    Compatible with app.py:
        extract_skills(text, skills_catalog)
    """
    normalized = normalize_text(text)

    if not normalized:
        return []

    patterns = dict(get_compiled_aliases())

    # Also support additional skills added to data/skills.csv.
    for skill in skill_catalog or []:
        skill = str(skill).strip()

        if skill and skill not in patterns:
            patterns[skill] = [compile_phrase(skill)]

    found = []

    for canonical, alias_patterns in patterns.items():
        if any(pattern.search(normalized) for pattern in alias_patterns):
            found.append(canonical)

    return sorted(set(found), key=str.casefold)


# =========================================================
# SENTENCE EMBEDDING MODEL
# =========================================================

@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the sentence-transformer model only when required."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# RESUME-JOB TEXT SIMILARITY
# =========================================================

def calculate_similarity(resume_text, job_text):
    """
    Calculate semantic similarity with Sentence Transformers.

    If the embedding model cannot be loaded, use TF-IDF cosine
    similarity as a fallback.

    Returns:
        (score_percentage, method_description)
    """
    resume_text = str(resume_text or "").strip()
    job_text = str(job_text or "").strip()

    if not resume_text or not job_text:
        return 0.0, "One or both texts are empty"

    # Preferred method: sentence embeddings and cosine similarity.
    try:
        model = get_embedding_model()

        embeddings = model.encode(
            [resume_text, job_text],
            normalize_embeddings=True,
        )

        score = float(
            cosine_similarity(
                [embeddings[0]],
                [embeddings[1]],
            )[0][0]
        )

        return max(0.0, min(100.0, score * 100)), (
            "Sentence Transformers + cosine similarity"
        )

    except Exception:
        # Fallback: TF-IDF cosine similarity.
        try:
            vectors = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
            ).fit_transform([resume_text, job_text])

            score = float(
                cosine_similarity(
                    vectors[0:1],
                    vectors[1:2],
                )[0][0]
            )

            return max(0.0, min(100.0, score * 100)), (
                "TF-IDF + cosine similarity (fallback)"
            )

        except ValueError:
            return 0.0, "Insufficient text for similarity calculation"


# =========================================================
# SKILL RECOMMENDATIONS
# =========================================================

def recommend_skills(missing_skills):
    """
    Return learning recommendations for skills missing from a resume.

    Each output item includes the skill name and a practical next step.
    """
    recommendations = []

    for skill in missing_skills or []:
        skill_name = str(skill).strip()

        if not skill_name:
            continue

        key = skill_name.casefold()

        recommendation = SKILL_RECOMMENDATIONS.get(
            key,
            (
                f"Study the fundamentals of {skill_name}, complete "
                "a guided exercise, and build a small practical project "
                "to demonstrate your understanding."
            ),
        )

        recommendations.append(
            f"{skill_name}: {recommendation}"
        )

    return recommendations
