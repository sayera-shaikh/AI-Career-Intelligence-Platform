
import streamlit as st
import streamlit.components.v1 as components
from streamlit_mic_recorder import speech_to_text
import pandas as pd
import pickle
import faiss
import numpy as np
import re

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ---------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="AI Career Intelligence",
    page_icon="💼",
    layout="wide"
)

# CUSTOM PROFESSIONAL THEME
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f4f3ff, #eef6ff);
}

h1 {
    color: #392477;
    font-weight: 800;
}

h2, h3 {
    color: #3949ab;
}

div.stButton > button {
    background: linear-gradient(90deg, #673ab7, #2962ff);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: bold;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #512da8, #1565c0);
    color: white;
    border: none;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dedbff;
    box-shadow: 0 3px 10px rgba(60, 50, 120, 0.08);
}

div[data-testid="stFileUploader"] {
    background: white;
    padding: 12px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# 2. LOAD KNOWLEDGE BASE AND MODEL
# ---------------------------------

@st.cache_resource
def load_resources():

    with open("data/knowledge_base_v3_expanded.pkl", "rb") as file:
        knowledge_base = pickle.load(file)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    kb_texts = [
        item["text"] for item in knowledge_base
    ]

    kb_embeddings = model.encode(
        kb_texts,
        show_progress_bar=False
    ).astype("float32")

    faiss.normalize_L2(kb_embeddings)

    kb_index = faiss.IndexFlatIP(
        kb_embeddings.shape[1]
    )

    kb_index.add(kb_embeddings)

    return model, knowledge_base, kb_index

try:
    model, knowledge_base, kb_index = load_resources()

except Exception as error:
    st.error(f"Could not load project resources: {error}")
    st.stop()

# ---------------------------------
# 3. CAREER ASSISTANT
# ---------------------------------


def career_assistant(question):

    question_clean = question.strip()

    if not question_clean:
        return "Please enter a career question."

    question_lower = question_clean.lower()

    if len(knowledge_base) == 0:
        return "Knowledge Base is empty."

    # Detect career role
    asks_analyst = any(term in question_lower for term in [
        "data analyst", "data analytics"
    ])

    asks_scientist = any(term in question_lower for term in [
        "data scientist", "data science"
    ])

    asks_comparison = (
        any(term in question_lower for term in [
            "compare", "comparison", "difference",
            "differentiate", "versus"
        ])
        and asks_analyst
        and asks_scientist
    )

    # Detect topic intent
    topic_keywords = {
        "Data Analyst Learning Roadmap": [
            "roadmap", "learning path", "beginner",
            "how should i learn"
        ],
        "Data Scientist Learning Roadmap": [
            "scientist roadmap", "data science roadmap"
        ],
        "Skill Gap Analysis": [
            "skill gap", "missing skills",
            "skills should i improve", "skills should i learn"
        ],
        "Python Interview Preparation": [
            "python interview", "python questions"
        ],
        "SQL Interview Preparation": [
            "sql interview", "sql questions"
        ],
        "Power BI Interview Preparation": [
            "power bi interview", "power bi questions"
        ],
        "Interview Question Generation": [
            "generate interview questions",
            "interview questions"
        ],
        "Recruiter Candidate Screening": [
            "candidate screening", "shortlisting",
            "recruiter"
        ],
        "Job Description Analysis": [
            "job description", "job requirements"
        ],
        "Resume Improvement": [
            "improve my resume", "resume improvement",
            "resume suggestions"
        ]
    }
    preferred_topics = []

    for topic, keywords in topic_keywords.items():
        if any(word in question_lower for word in keywords):
            preferred_topics.append(topic)

    # Detect focused technical topic
    focused_topic = None

    if "sql" in question_lower:
        focused_topic = "sql"

    elif "python" in question_lower:
        focused_topic = "python"

    elif "power bi" in question_lower:
        focused_topic = "power bi"

    elif "machine learning" in question_lower:
        focused_topic = "machine learning"
       
    # Create question embedding
    question_vector = model.encode(
        [question_clean],
        show_progress_bar=False
    ).astype("float32")

    faiss.normalize_L2(question_vector)

    scores, indices = kb_index.search(
        question_vector,
        len(knowledge_base)
    )

    question_terms = set(re.findall(
        r"[a-zA-Z][a-zA-Z0-9+#.-]+",
        question_lower
    ))

    ranked_results = []
    seen_texts = set()

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        item = knowledge_base[idx]

        topic = item.get("topic", "")
        item_text = item.get("text", "").strip()

        if not item_text:
            continue

        text_lower = item_text.lower()
        topic_lower = topic.lower()

        # Filter unrelated technical topics
        if focused_topic and focused_topic not in topic_lower:
            continue

        # Avoid unrelated role information
        if not asks_comparison:

            if asks_analyst and (
                "data scientist" in topic_lower
                or "machine learning engineer" in topic_lower
            ):
                continue

            if asks_scientist and (
                "data analyst" in topic_lower
                and "data scientist" not in topic_lower
            ):
                continue

        # Remove duplicate text
        normalized = re.sub(r"\s+", " ", text_lower)

        if normalized in seen_texts:
            continue

        seen_texts.add(normalized)

        # Calculate relevance
        relevance = float(score)

        if topic in preferred_topics:
            relevance += 0.25

        elif any(word in topic_lower for word in question_terms):
            relevance += 0.08

        text_terms = set(re.findall(
            r"[a-zA-Z][a-zA-Z0-9+#.-]+",
            text_lower
        ))

        overlap = len(question_terms.intersection(text_terms))

        relevance += min(overlap * 0.01, 0.08)

        if asks_analyst and "data analyst" in topic_lower:
            relevance += 0.12

        if asks_scientist and "data scientist" in topic_lower:
            relevance += 0.12

        ranked_results.append((
            relevance,
            topic,
            item_text
        ))

    ranked_results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # Select relevant but distinct knowledge-base entries. Semantic search can
    # otherwise return several near-identical entries for the same role.
    from difflib import SequenceMatcher

    selected = []
    selected_topics = set()
    selected_texts = []

    for relevance, topic, item_text in ranked_results:
        topic_key = re.sub(r"[^a-z0-9]+", " ", topic.lower()).strip()
        clean_text = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", item_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        # Avoid repeated topic labels and near-duplicate content.
        if topic_key in selected_topics:
            continue
        if any(
            SequenceMatcher(None, clean_text.lower(), previous.lower()).ratio() >= 0.58
            for previous in selected_texts
        ):
            continue

        selected.append((relevance, topic, clean_text))
        selected_topics.add(topic_key)
        selected_texts.append(clean_text)

        if len(selected) == 3:
            break

    if not selected or selected[0][0] < 0.20:
        return (
            "I could not find sufficiently relevant information "
            "in the current Knowledge Base. Please try asking "
            "about Data Analyst, Data Scientist, Python, SQL "
            "or Power BI."
        )

    response = []

    for number, (score, topic, text) in enumerate(selected, 1):
        display_text = text
        if display_text.lower().startswith(topic.lower()):
            display_text = display_text[len(topic):].lstrip(" :-—")
        response.append(f"{number}. {topic}\\n{display_text}")

    return "\n\n".join(response)

# ---------------------------------
# 4. RESUME TEXT EXTRACTION
# ---------------------------------

def extract_resume_text(pdf_file):

    pdf_file.seek(0)

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text

# ---------------------------------
# 5. SKILL DICTIONARY AND ALIASES
# ---------------------------------

skill_aliases = {

    "Python": ["python"],

    "SQL": ["sql", "structured query language"],

    "Excel": ["excel", "microsoft excel", "ms excel"],

    "Power BI": ["power bi", "powerbi"],

    "Statistics": ["statistics", "statistical analysis"],

    "Data Visualization": [
        "data visualization",
        "data visualisation"
    ],

    "Pandas": ["pandas"],

    "NumPy": ["numpy"],

    "Machine Learning": [
        "machine learning",
        "machine-learning",
        "ml"
    ],

    "Deep Learning": ["deep learning"],

    "CNN": [
        "cnn",
        "convolutional neural network",
        "convolutional neural networks"
    ],

    "Transformers": [
        "transformer",
        "transformers",
        "transformer model",
        "transformer models"
    ],

    "Hugging Face": [
        "hugging face",
        "huggingface",
        "hugging face transformers"
    ],

    "Tokenization": [
        "tokenization",
        "tokenisation",
        "tokenizer",
        "tokenizers"
    ],

    "Embeddings": [
        "embedding",
        "embeddings",
        "word embedding",
        "word embeddings",
        "text embedding",
        "text embeddings"
    ],

    "BERT": ["bert"],

    "GPT": ["gpt", "generative pre-trained transformer"],

    "TensorFlow": ["tensorflow"],

    "Scikit-learn": [
        "scikit-learn",
        "scikit learn",
        "sklearn"
    ],

    "Tableau": ["tableau"],

    "Matplotlib": ["matplotlib"],

    "Seaborn": ["seaborn"],

    "NLP": [
        "nlp",
        "natural language processing"
    ],

    "R": ["r"],

    "Java": ["java"],

    "C++": ["c++"],

    "Data Analysis": [
        "data analysis",
        "data analytics"
    ],

    "Data Cleaning": [
        "data cleaning",
        "data cleansing"
    ],

    "Data Mining": ["data mining"],

    "ETL": [
        "etl",
        "extract transform load"
    ],

    "AWS": [
        "aws",
        "amazon web services"
    ],

    "Azure": [
        "azure",
        "microsoft azure"
    ],

    "AWS Glue": ["aws glue"],
    "Data Warehousing": ["data warehousing", "data warehouse"],
    "Data Modeling": ["data modeling", "data modelling"],
    "dbt": ["dbt", "data build tool"],
    "Airflow": ["airflow", "apache airflow"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Linux": ["linux"],
    "Jupyter": ["jupyter", "jupyter notebook"],
    "FastAPI": ["fastapi"],
    "MLOps": ["mlops", "machine learning operations"],
    "Model Deployment": ["model deployment", "deploying models"],
    "Generative AI": ["generative ai", "gen ai"],
    "Vector Database": ["vector database", "vector databases", "faiss", "chroma"],

    "Git": ["git", "github"],

    "Flask": ["flask"],

    "Streamlit": ["streamlit"],

    "PyTorch": ["pytorch", "torch"],

    "Computer Vision": ["computer vision", "cv"],
    "OpenCV": ["opencv", "open cv"],

    "Big Data": ["big data"],

    "Hadoop": ["hadoop"],

    "Spark": [
        "apache spark",
        "pyspark",
        "spark"
    ],

    "MySQL": ["mysql"],

    "PostgreSQL": [
        "postgresql",
        "postgres"
    ],

    "MongoDB": ["mongodb"],

    "PowerPoint": [
        "powerpoint",
        "ms powerpoint"
    ],

    "Keras": ["keras"],

    "OpenAI": ["openai"],

    "LangChain": ["langchain"],

    "RAG": [
        "rag",
        "retrieval augmented generation"
    ],

    "LLM": [
        "llm",
        "large language model",
        "large language models"
    ],

    "HTML": ["html"],

    "CSS": ["css"],

    "JavaScript": ["javascript"],

    "Django": ["django"],

    "API": ["api", "apis"],

    "Data Structures": [
        "data structures",
        "data structure"
    ],

    "Problem Solving": ["problem solving"],

    "Time Series": ["time series"],

    "Regression": ["regression"],

    "Classification": ["classification"],

    "Neural Networks": [
        "neural networks",
        "neural network"
    ]
}

skills_list = list(skill_aliases.keys())

# ---------------------------------
# 6. SKILL EXTRACTION FUNCTION
# ---------------------------------

def extract_skills(text):

    if not isinstance(text, str):
        return []

    text = text.lower()

    found_skills = []

    for skill, aliases in skill_aliases.items():

        for alias in aliases:

            pattern = (
                r"(?<![a-zA-Z0-9])"
                + re.escape(alias.lower())
                + r"(?![a-zA-Z0-9])"
            )

            if re.search(pattern, text):

                found_skills.append(skill)
                break

    return found_skills

# ---------------------------------


# ==========================================
# AI CAREER INTELLIGENCE PLATFORM
# NEW DASHBOARD
# ==========================================

st.title("💼 AI Career Intelligence Platform")

st.sidebar.markdown("## 💼 Career Hub")
st.sidebar.caption("AI-powered career & recruitment")

page = st.sidebar.radio(
    "Select Portal",
    [
        "Home",
        "Candidate Portal",
        "Recruiter Portal"
    ]
)

# Keep Recruiter results available across Streamlit reruns.
if "recruiter_results" not in st.session_state:
    st.session_state["recruiter_results"] = None
if "recruiter_required_skills" not in st.session_state:
    st.session_state["recruiter_required_skills"] = []

# ==========================================
# HOME
# ==========================================


if page == "Home":

    st.markdown("""
    <div style="
        background: linear-gradient(120deg, #4338ca, #2563eb);
        padding: 35px;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
    ">
        <h1>AI Career Intelligence Platform</h1>
        <h3>Your Career Growth & Recruitment Partner</h3>
        <p>Smart tools for job seekers and recruiters — all in one place.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Welcome! Choose Your Portal")

    st.write(
        "Select the portal from the sidebar to access the tools "
        "designed for your needs."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #eef2ff, #dbeafe);
            padding: 22px;
            border-radius: 15px;
            border: 1px solid #a5b4fc;
            min-height: 220px;
        ">
            <h2 style="color:#3730a3;">👩‍💻 Candidate Portal</h2>
            <p style="color:#1e293b;">
                Build your career with smart resume and skill analysis.
            </p>
            <ul style="color:#1e293b;">
                <li>Resume Analysis</li>
                <li>Job Skill Matching</li>
                <li>Skill Gap Analysis</li>
                <li>AI Career Assistant</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f3e8ff, #ede9fe);
            padding: 22px;
            border-radius: 15px;
            border: 1px solid #c4b5fd;
            min-height: 220px;
        ">
            <h2 style="color:#6b21a8;">🏢 Recruiter Portal</h2>
            <p style="color:#1e293b;">
                Find and shortlist suitable candidates efficiently.
            </p>
            <ul style="color:#1e293b;">
                <li>Job Description Analysis</li>
                <li>Multiple Resume Screening</li>
                <li>Candidate Ranking</li>
                <li>Smart Shortlisting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("How to Get Started")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.info("**01 — Select Portal**\n\nChoose Candidate or Recruiter from the sidebar.")

    with step2:
        st.info("**02 — Enter Details**\n\nUpload your resume or enter a job description.")

    with step3:
        st.info("**03 — Get Results**\n\nView your analysis, matching and recommendations.")

    st.caption(
        "Note: Each portal has its own tools and results. "
        "The Home Page is only your starting point."
    )


# ==========================================
# CANDIDATE PORTAL
# ==========================================

elif page == "Candidate Portal":

    st.title("👩‍💻 Candidate Portal")
    st.write("Your personal career guide — analyze your resume and plan your next step.")

    # Career role -> role-specific skills and learning roadmap.
    # This is career guidance, not a live vacancy or employer-specific JD.
    role_skill_map = {
        "Data Analyst": ["Python", "SQL", "Excel", "Power BI", "Statistics", "Data Visualization", "Pandas", "NumPy"],
        "Junior Data Analyst": ["Excel", "SQL", "Data Analysis", "Data Cleaning", "Power BI", "Statistics"],
        "Business Analyst": ["Excel", "SQL", "Power BI", "Data Visualization", "Statistics", "Data Analysis"],
        "Business Intelligence (BI) Analyst": ["SQL", "Power BI", "Tableau", "Data Visualization", "Data Warehousing"],
        "Reporting Analyst": ["Excel", "SQL", "Power BI", "Data Visualization", "Data Cleaning"],
        "Financial Data Analyst": ["Excel", "SQL", "Statistics", "Data Analysis", "Power BI"],
        "Marketing Data Analyst": ["Excel", "SQL", "Python", "Data Analysis", "Statistics", "Data Visualization"],
        "Data Scientist": ["Python", "SQL", "Statistics", "Pandas", "NumPy", "Machine Learning", "Scikit-learn", "Data Visualization"],
        "Junior Data Scientist": ["Python", "Statistics", "Pandas", "NumPy", "SQL", "Machine Learning"],
        "Machine Learning Engineer": ["Python", "Machine Learning", "Scikit-learn", "TensorFlow", "PyTorch", "SQL"],
        "AI Engineer": ["Python", "Machine Learning", "Deep Learning", "API", "Generative AI"],
        "Deep Learning Engineer": ["Python", "Deep Learning", "TensorFlow", "PyTorch", "Neural Networks"],
        "NLP Engineer": ["Python", "Machine Learning", "Deep Learning", "NLP", "LLM"],
        "Computer Vision Engineer": ["Python", "Computer Vision", "OpenCV", "Deep Learning", "PyTorch"],
        "Data Engineer": ["Python", "SQL", "ETL", "AWS", "Azure", "Spark", "Hadoop", "PostgreSQL"],
        "Junior Data Engineer": ["Python", "SQL", "ETL", "Data Warehousing", "Airflow"],
        "Big Data Engineer": ["Python", "SQL", "Big Data", "Hadoop", "Spark"],
        "ETL Developer": ["SQL", "Python", "ETL", "Data Warehousing", "Airflow"],
        "Analytics Engineer": ["SQL", "Python", "Data Warehousing", "Data Modeling", "dbt"],
        "Generative AI Engineer": ["Python", "Generative AI", "LLM", "RAG", "LangChain", "API"],
        "LLM Engineer": ["Python", "LLM", "RAG", "LangChain", "API", "OpenAI"],
        "AI Application Developer": ["Python", "API", "OpenAI", "Generative AI", "LangChain"],
        "Prompt Engineer": ["Generative AI", "LLM", "OpenAI", "Problem Solving"],
        "RAG Developer": ["Python", "RAG", "LangChain", "LLM", "Vector Database", "API"],
    }

    role_roadmaps = {
        "Data Analyst": [
            "Strengthen Excel, SQL and data cleaning.",
            "Practice Python with Pandas and NumPy.",
            "Build two dashboards using Power BI or Tableau.",
            "Complete two end-to-end analysis projects and publish them on GitHub.",
            "Practice SQL, Excel and business-case interview questions."
        ],
        "Junior Data Analyst": [
            "Learn Excel formulas, pivot tables and data cleaning.",
            "Practice SQL SELECT, JOIN, GROUP BY and subqueries.",
            "Build a Power BI dashboard using a public dataset.",
            "Complete a small analysis project and explain your findings.",
            "Prepare basic SQL, Excel and analytical reasoning questions."
        ],
        "Business Analyst": [
            "Strengthen Excel and SQL.",
            "Learn dashboarding with Power BI.",
            "Practice requirements gathering and documentation.",
            "Build business reports using realistic datasets.",
            "Prepare business-case and communication interview examples."
        ],
        "Business Intelligence (BI) Analyst": [
            "Strengthen SQL queries and data aggregation.",
            "Learn Power BI and Tableau dashboard design.",
            "Understand data warehouses and dimensional modelling.",
            "Build an interactive KPI dashboard.",
            "Practice explaining business insights to stakeholders."
        ],
        "Reporting Analyst": [
            "Improve Excel formulas, pivot tables and reporting.",
            "Practice SQL for extracting and summarising data.",
            "Learn Power BI report creation and visualisation.",
            "Automate a recurring report using a practical dataset.",
            "Practice accuracy checks and explaining report findings."
        ],
        "Financial Data Analyst": [
            "Strengthen Excel, financial formulas and statistics.",
            "Learn SQL for financial data extraction.",
            "Practice analysing revenue, cost and profitability.",
            "Build a financial KPI dashboard in Power BI.",
            "Prepare budgeting, variance and financial analysis examples."
        ],
        "Marketing Data Analyst": [
            "Learn marketing metrics such as conversion and retention.",
            "Practice SQL and Excel for campaign analysis.",
            "Use Python and Pandas for data cleaning.",
            "Build a campaign performance dashboard.",
            "Practice explaining insights and recommendations."
        ],
        "Data Scientist": [
            "Build a strong foundation in Python, statistics and probability.",
            "Practice data analysis with Pandas and NumPy.",
            "Learn supervised and unsupervised machine learning.",
            "Build and explain two end-to-end ML projects.",
            "Practice model evaluation and technical interviews."
        ],
        "Junior Data Scientist": [
            "Strengthen Python, statistics and probability fundamentals.",
            "Practice Pandas, NumPy and exploratory data analysis.",
            "Learn core ML algorithms with Scikit-learn.",
            "Build one regression and one classification project.",
            "Practice explaining metrics, model results and limitations."
        ],
        "Machine Learning Engineer": [
            "Strengthen Python, data structures and SQL.",
            "Learn machine learning algorithms and evaluation.",
            "Practice Scikit-learn and one deep-learning framework.",
            "Build, test and document deployable ML projects.",
            "Learn APIs, Docker and model deployment basics."
        ],
        "AI Engineer": [
            "Strengthen Python and software development fundamentals.",
            "Learn machine learning and deep learning basics.",
            "Explore LLMs, prompt design and AI APIs.",
            "Build and test a practical AI-powered application.",
            "Learn deployment, evaluation and responsible AI practices."
        ],
        "Deep Learning Engineer": [
            "Strengthen Python, linear algebra and probability.",
            "Learn neural networks and backpropagation.",
            "Practice TensorFlow or PyTorch.",
            "Build CNN and sequence-model projects.",
            "Learn model evaluation, optimisation and deployment."
        ],
        "NLP Engineer": [
            "Strengthen Python and text preprocessing.",
            "Learn NLP fundamentals and language representations.",
            "Practice machine learning and deep learning for text.",
            "Build a text classification or information extraction project.",
            "Explore transformers, LLM evaluation and deployment."
        ],
        "Computer Vision Engineer": [
            "Strengthen Python, NumPy and image processing.",
            "Learn OpenCV and image augmentation.",
            "Study CNNs and computer vision architectures.",
            "Build an image classification or object detection project.",
            "Evaluate and deploy a vision model."
        ],
        "Data Engineer": [
            "Strengthen Python and advanced SQL.",
            "Learn ETL/ELT concepts and data pipeline design.",
            "Practice relational databases and data modelling.",
            "Learn a cloud platform and a big-data tool such as Spark.",
            "Build a small automated data pipeline project."
        ],
        "Junior Data Engineer": [
            "Practice Python scripting and SQL joins.",
            "Learn databases, data cleaning and ETL fundamentals.",
            "Understand data warehouses and data modelling.",
            "Build a simple scheduled pipeline with Airflow concepts.",
            "Document a small end-to-end data engineering project."
        ],
        "Big Data Engineer": [
            "Strengthen Python, SQL and distributed computing basics.",
            "Learn Hadoop ecosystem fundamentals.",
            "Practice Apache Spark and PySpark.",
            "Understand batch and streaming data pipelines.",
            "Build a scalable data processing project."
        ],
        "ETL Developer": [
            "Strengthen SQL and Python scripting.",
            "Learn ETL design, extraction and transformation.",
            "Practice data quality checks and error handling.",
            "Understand scheduling and orchestration with Airflow.",
            "Build and document an end-to-end ETL pipeline."
        ],
        "Analytics Engineer": [
            "Strengthen SQL and analytical data modelling.",
            "Learn warehouse concepts and dimensional modelling.",
            "Practice dbt models, tests and documentation.",
            "Build a transformation project using sample business data.",
            "Learn version control and data quality practices."
        ],
        "Generative AI Engineer": [
            "Strengthen Python and API development.",
            "Learn LLM fundamentals and prompt engineering.",
            "Practice embeddings, vector search and RAG.",
            "Build a grounded chatbot using a knowledge base.",
            "Evaluate answer quality, safety and deployment."
        ],
        "LLM Engineer": [
            "Strengthen Python and software engineering basics.",
            "Understand transformers, tokens and embeddings.",
            "Practice LLM APIs, prompting and evaluation.",
            "Build a retrieval-augmented generation application.",
            "Learn deployment, monitoring and cost optimisation."
        ],
        "AI Application Developer": [
            "Strengthen Python and application development.",
            "Learn REST APIs and AI model integration.",
            "Practice prompt design and structured outputs.",
            "Build an AI application with a clear user workflow.",
            "Test, document and deploy the application."
        ],
        "Prompt Engineer": [
            "Understand LLM capabilities and limitations.",
            "Practice clear instructions and context design.",
            "Learn structured outputs and prompt evaluation.",
            "Build a small prompt library for practical tasks.",
            "Measure reliability, accuracy and safety."
        ],
        "RAG Developer": [
            "Strengthen Python and document processing.",
            "Learn embeddings, chunking and vector databases.",
            "Practice retrieval and reranking concepts.",
            "Build a RAG application using LangChain.",
            "Evaluate retrieval quality and grounded responses."
        ],
    }

    st.markdown("### 1. Your Career Preference")
    st.caption("Choose the role you want. You do not need to paste a recruiter's full job description.")

    role_options = list(role_skill_map.keys())
    st.caption("You can also say your preference using the microphone.")
    pref_col, pref_mic_col = st.columns([0.86, 0.14])
    with pref_mic_col:
        spoken_preference = speech_to_text(
            language="en",
            start_prompt="🎤 Speak",
            stop_prompt="⏹️ Stop",
            just_once=True,
            use_container_width=True,
            key="candidate_preference_voice_v2"
        )

    if spoken_preference and spoken_preference.strip():
        spoken_lower = spoken_preference.lower()
        detected_role = next(
            (role for role in sorted(role_options, key=len, reverse=True) if role.lower() in spoken_lower),
            None
        )
        if detected_role:
            st.session_state["candidate_selected_role"] = detected_role
            st.success(f"Voice preference detected: {detected_role}")
        else:
            st.info(f"I heard: {spoken_preference}. Please choose the closest role above.")

    selected_role = st.selectbox(
        "Which job are you looking for?",
        role_options,
        key="candidate_selected_role"
    )

    st.markdown(f"**Selected career goal:** `{selected_role}`")
    target_skills = role_skill_map[selected_role]

    st.divider()
    st.markdown("### 2. Resume Analyzer")
    candidate_resume = st.file_uploader(
        "Upload Your Resume (PDF)",
        type=["pdf"],
        key="candidate_resume"
    )

    if st.button("🔍 Analyze My Resume", type="primary", key="analyze_candidate_resume"):
        if candidate_resume is None:
            st.warning("Please upload your resume first.")
        else:
            try:
                resume_text = extract_resume_text(candidate_resume)
                if not resume_text.strip():
                    st.error("No readable text was extracted. Please check your PDF.")
                else:
                    candidate_skills = extract_skills(resume_text)
                    matched_skills = [skill for skill in target_skills if skill in candidate_skills]
                    missing_skills = [skill for skill in target_skills if skill not in candidate_skills]
                    score = (len(matched_skills) / len(target_skills)) * 100

                    st.session_state["candidate_analysis"] = {
                        "role": selected_role,
                        "skills": candidate_skills,
                        "matched": matched_skills,
                        "missing": missing_skills,
                        "score": score
                    }
                    st.success("Resume analyzed successfully!")
            except Exception as error:
                st.error(f"Resume analysis error: {error}")

    analysis = st.session_state.get("candidate_analysis")
    if analysis:
        # Do not show stale results for a different selected role
        if analysis.get("role") != selected_role:
            st.info("Your selected role changed. Click Analyze My Resume again to refresh the career recommendations.")
        else:
            st.markdown("### 3. Your Career Match")
            m1, m2, m3 = st.columns(3)
            m1.metric("Resume-to-role skill match", f"{analysis['score']:.1f}%")
            m2.metric("Skills found", len(analysis["matched"]))
            m3.metric("Skills to develop", len(analysis["missing"]))

            st.progress(int(analysis["score"]))
            left, right = st.columns(2)
            with left:
                st.success("Skills you already have")
                st.write(", ".join(analysis["matched"]) if analysis["matched"] else "No target skills detected yet.")
            with right:
                st.warning("Skills to develop next")
                st.write(", ".join(analysis["missing"]) if analysis["missing"] else "Great! All listed target skills were found in your resume.")

            st.markdown("### 4. Personalized Learning Roadmap")
            for idx, item in enumerate(role_roadmaps[selected_role], 1):
                st.write(f"**Step {idx}.** {item}")

            st.markdown("### 5. Resume Improvement Tips")
            tips = []

            # Give advice based on the selected role and the actual skill gaps.
            missing_skills = analysis["missing"]
            matched_skills = analysis["matched"]

            if missing_skills:
                priority_skills = ", ".join(missing_skills[:4])
                tips.append(
                    f"For the {selected_role} role, prioritise learning and demonstrating: "
                    f"{priority_skills}."
                )
            else:
                tips.append(
                    f"Your resume includes all skills in this {selected_role} checklist. "
                    "Strengthen the evidence with project descriptions and measurable results."
                )

            role_project_tips = {
                "Data Analyst": "Add a data analysis project showing data cleaning, SQL queries and a Power BI or Excel dashboard.",
                "Junior Data Analyst": "Add one beginner-friendly project showing Excel, SQL and clear business insights.",
                "Business Analyst": "Include a project with a business problem, requirements, analysis and recommendations.",
                "Business Intelligence (BI) Analyst": "Show a dashboard project with KPIs, data modelling and interactive filters.",
                "Reporting Analyst": "Show an automated or repeatable report and explain how it helps users make decisions.",
                "Financial Data Analyst": "Add a financial analysis project covering revenue, costs, profitability or variance.",
                "Marketing Data Analyst": "Show campaign analysis using conversion, engagement or customer-retention metrics.",
                "Data Scientist": "Include an end-to-end ML project: problem, data preparation, model, evaluation metric and result.",
                "Junior Data Scientist": "Add one regression and one classification project, and explain the evaluation metrics.",
                "Machine Learning Engineer": "Show how you trained, evaluated, tested and deployed or served an ML model.",
                "AI Engineer": "Include a practical AI application and explain the model/API, workflow and evaluation.",
                "Deep Learning Engineer": "Show a neural-network project and explain the dataset, architecture and evaluation.",
                "NLP Engineer": "Add a text classification, sentiment analysis or information-extraction project.",
                "Computer Vision Engineer": "Add an image classification or object-detection project with evaluation results.",
                "Data Engineer": "Show an ETL/ELT pipeline, its data sources, transformations and output.",
                "Junior Data Engineer": "Add a small Python/SQL pipeline project and explain each processing stage.",
                "Big Data Engineer": "Show a Spark or PySpark project and describe the data volume and transformations.",
                "ETL Developer": "Document an ETL project with extraction, transformation, validation and loading steps.",
                "Analytics Engineer": "Show SQL transformations, data models and data quality tests in a project.",
                "Generative AI Engineer": "Include a GenAI project and explain prompting, retrieval, grounding and evaluation.",
                "LLM Engineer": "Show an LLM application with model/API choice, prompt design and evaluation.",
                "AI Application Developer": "Include an AI-powered application and explain its API integration and user workflow.",
                "Prompt Engineer": "Show before/after prompt examples and how you evaluated output quality.",
                "RAG Developer": "Show a RAG project with document chunking, embeddings, retrieval and answer evaluation."
            }
            tips.append(role_project_tips.get(
                selected_role,
                "Add a relevant project and explain the problem, tools, process and outcome."
            ))

            if "Git" not in matched_skills:
                tips.append("Add a GitHub link with a clear README, setup steps and screenshots for your strongest project, if available.")
            if len(analysis["skills"]) < 5:
                tips.append("Add a clearly labelled Technical Skills section, listing only tools you can explain and have genuinely used.")
            if missing_skills:
                tips.append("Do not just list missing tools on your resume; first practise them and add evidence through coursework or projects.")
            else:
                tips.append("Use quantified outcomes where possible, such as records analysed, time saved, model score or dashboard KPIs.")
            tips.append("Mention relevant internships and certifications, with dates and the practical work completed.")

            # Remove repeated tips while preserving their order.
            unique_tips = list(dict.fromkeys(tips))
            for tip in unique_tips[:5]:
                st.write(f"• {tip}")

            st.caption("The match is based on skill keywords found in the PDF and a role-based skill checklist. It is guidance, not a hiring prediction.")

    st.divider()
    st.markdown("### 6. AI Career Assistant")
    st.caption("Type a question or use the microphone. Example: What SQL topics should I prepare for a Data Analyst interview?")

    assistant_mic_col, assistant_input_col = st.columns([0.13, 0.87])
    with assistant_mic_col:
        st.write("")
        spoken_question = speech_to_text(
            language="en",
            start_prompt="🎙️ Speak",
            stop_prompt="⏹️ Stop",
            just_once=True,
            use_container_width=True,
            key="candidate_assistant_voice_v2"
        )
    if spoken_question and spoken_question.strip():
        st.session_state["career_question_input_v2"] = spoken_question.strip()

    with assistant_input_col:
        typed_question = st.text_input(
            "Ask a career-related question",
            placeholder="Example: How can I prepare for a Data Analyst interview?",
            key="career_question_input_v2"
        )

    if st.button("Ask AI Career Assistant", type="primary", key="ask_candidate_assistant"):
        question = typed_question.strip()
        if not question:
            st.warning("Please type or speak your question.")
        else:
            try:
                with st.spinner("Finding relevant information..."):
                    answer = career_assistant(question)
                st.subheader("💡 AI Assistant Response")
                st.write(answer)
                speech_text = str(answer)
                speech_html = f"""
                <div style="display:flex; gap:12px; flex-wrap:wrap;">
                    <button onclick="speakAnswer()" style="background:linear-gradient(90deg,#673ab7,#2196f3);color:white;padding:12px 22px;border:none;border-radius:10px;cursor:pointer;font-size:15px;">🔊 Listen to Answer</button>
                    <button onclick="stopAnswer()" style="background:#e53935;color:white;padding:12px 22px;border:none;border-radius:10px;cursor:pointer;font-size:15px;">⏹ Stop Speaking</button>
                </div>
                <script>
                function speakAnswer() {{
                    window.speechSynthesis.cancel();
                    let speech = new SpeechSynthesisUtterance({speech_text!r});
                    speech.lang = "en-IN";
                    speech.rate = 0.9;
                    window.speechSynthesis.speak(speech);
                }}
                function stopAnswer() {{ window.speechSynthesis.cancel(); }}
                </script>
                """
                components.html(speech_html, height=65)
            except Exception as error:
                st.error(f"Career Assistant Error: {error}")


# ==========================================
# RECRUITER PORTAL
# ==========================================
elif page == "Recruiter Portal":

    st.title("🏢 Recruiter Portal")
    st.write(
        "Upload multiple candidate resumes and compare their skills "
        "against one job description."
    )

    # 1. Job description input
    st.header("1. Job Description")
    st.caption("Type the job requirements or use the microphone.")

    jd_col, mic_col = st.columns([0.88, 0.12])

    with mic_col:
        spoken_job_text = speech_to_text(
            language="en",
            start_prompt="🎤",
            stop_prompt="⏹️",
            just_once=True,
            use_container_width=True,
            key="recruiter_job_voice"
        )

    # Put recognized speech into the text area before creating the widget.
    if spoken_job_text and spoken_job_text.strip():
        st.session_state["recruiter_job_description"] = spoken_job_text.strip()

    with jd_col:
        job_description = st.text_area(
            "Paste or edit the Job Description",
            height=170,
            key="recruiter_job_description",
            placeholder="Example: Hiring Junior Data Analyst. Required skills: Python, SQL, Excel, Power BI, Statistics and Data Visualization. Responsibilities: analyze datasets, clean data, write SQL queries, build dashboards and prepare reports. Qualification: Bachelor's degree. Freshers can apply."
        )

    st.divider()

    # 2. Resume uploader
    st.header("2. Upload Candidate Resumes")
    uploaded_resumes = st.file_uploader(
        "Select one or more candidate resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        key="recruiter_resumes"
    )

    if uploaded_resumes:
        st.success(f"{len(uploaded_resumes)} resume(s) uploaded.")

    # 3. Explicit analysis action
    analyze_clicked = st.button(
        "🔍 Analyze Candidate Resumes",
        type="primary",
        use_container_width=False
    )

    if analyze_clicked:
        if not job_description or not job_description.strip():
            st.warning("Please enter or speak a Job Description first.")
            st.session_state["recruiter_results"] = None
        elif not uploaded_resumes:
            st.warning("Please upload at least one candidate resume (PDF).")
            st.session_state["recruiter_results"] = None
        else:
            required_skills = extract_skills(job_description)

            if not required_skills:
                st.warning(
                    "No recognized skills were found in this Job Description. "
                    "Please include skills such as Python, SQL, Excel or Power BI."
                )
                st.session_state["recruiter_results"] = None
            else:
                st.session_state["recruiter_required_skills"] = required_skills
                candidate_results = []
                progress = st.progress(0, text="Analyzing resumes...")

                for index, resume_file in enumerate(uploaded_resumes):
                    try:
                        resume_text = extract_resume_text(resume_file)
                        if not resume_text.strip():
                            st.warning(
                                f"Could not extract readable text from {resume_file.name}. "
                                "It may be a scanned PDF."
                            )
                        else:
                            candidate_skills = extract_skills(resume_text)
                            matched_skills = [
                                skill for skill in required_skills
                                if skill in candidate_skills
                            ]
                            missing_skills = [
                                skill for skill in required_skills
                                if skill not in candidate_skills
                            ]
                            match_score = (
                                len(matched_skills) / len(required_skills)
                            ) * 100

                            candidate_results.append({
                                "Candidate": resume_file.name,
                                "Match Score": round(match_score, 2),
                                "Matched Skills": ", ".join(matched_skills),
                                "Missing Skills": ", ".join(missing_skills),
                                "All Identified Skills": ", ".join(candidate_skills)
                            })
                    except Exception as error:
                        st.warning(f"Could not analyze {resume_file.name}: {error}")

                    progress.progress(
                        (index + 1) / len(uploaded_resumes),
                        text=f"Processed {index + 1} of {len(uploaded_resumes)} resumes"
                    )

                progress.empty()
                if candidate_results:
                    results_df = pd.DataFrame(candidate_results)
                    results_df = results_df.sort_values(
                        by="Match Score", ascending=False
                    ).reset_index(drop=True)
                    results_df.insert(0, "Rank", range(1, len(results_df) + 1))
                    st.session_state["recruiter_results"] = results_df
                    st.success("Candidate screening completed successfully!")
                else:
                    st.session_state["recruiter_results"] = None
                    st.error("No resumes could be analyzed. Please check the PDF files.")

    # 4. Display stored results (also remain visible after slider changes)
    results_df = st.session_state.get("recruiter_results")
    required_skills = st.session_state.get("recruiter_required_skills", [])

    if results_df is not None:
        st.divider()
        st.header("3. Job Description — Required Skills")
        st.write(", ".join(required_skills))

        st.header("4. Candidate Ranking")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.header("5. Candidate Skill Details")
        for _, row in results_df.iterrows():
            with st.expander(
                f"Rank {row['Rank']} — {row['Candidate']} ({row['Match Score']}%)"
            ):
                st.write("**Matched Skills:**")
                st.write(row["Matched Skills"] or "No matched skills")
                st.write("**Missing Skills:**")
                st.write(row["Missing Skills"] or "No missing skills")
                st.write("**All Identified Skills:**")
                st.write(row["All Identified Skills"] or "No skills identified")

        st.header("6. Smart Shortlisting")
        threshold = st.slider(
            "Minimum Match Score for Shortlisting",
            min_value=0, max_value=100, value=70,
            key="recruiter_shortlist_threshold"
        )
        shortlisted_df = results_df[results_df["Match Score"] >= threshold]
        st.metric("Shortlisted Candidates", len(shortlisted_df))
        st.dataframe(shortlisted_df, use_container_width=True, hide_index=True)

        csv_data = results_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ Download Candidate Ranking CSV",
            data=csv_data,
            file_name="candidate_ranking.csv",
            mime="text/csv",
            key="download_candidate_ranking"
        )
        st.caption(
            "Match scores represent keyword skill overlap. They are not a "
            "guarantee of hiring or overall candidate suitability."
        )
    else:
        st.info(
            "First enter the job description, upload candidate PDF resumes, "
            "then click **Analyze Candidate Resumes**."
        )
