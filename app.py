
import streamlit as st
import fitz
import pandas as pd
from datetime import datetime
from pathlib import Path

from skill_extractor import (
    extract_skills,
    calculate_similarity,
    recommend_skills,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SkillMatch AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        color: #777;
        font-size: 1.05rem;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 10px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

@st.cache_data
def load_skills():
    """Load the skill catalog from data/skills.csv."""
    skills_path = Path("data") / "skills.csv"

    if not skills_path.exists():
        raise FileNotFoundError(
            "data/skills.csv was not found. "
            "Keep the data folder inside your project."
        )

    df = pd.read_csv(skills_path)

    if df.empty:
        return []

    if "skill" in df.columns:
        column = "skill"
    else:
        column = df.columns[0]

    skills = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    return sorted(
        set(skill for skill in skills if skill),
        key=str.casefold,
    )


def extract_pdf_text(uploaded_file):
    """Extract text from a PDF using PyMuPDF."""
    text_parts = []

    with fitz.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf",
    ) as pdf:
        for page in pdf:
            text_parts.append(page.get_text("text") or "")

    return "\n".join(text_parts).strip()


def clean_text(text):
    """Normalize unnecessary whitespace."""
    if not text:
        return ""

    return " ".join(str(text).split())


def normalize_skills(skills):
    """Remove duplicate skills without considering letter case."""
    if skills is None:
        return []

    if isinstance(skills, str):
        skills = [skills]

    normalized = {}
    for skill in skills:
        skill = str(skill).strip()
        if skill:
            normalized.setdefault(skill.casefold(), skill)

    return sorted(normalized.values(), key=str.casefold)


def compare_skills(resume_skills, job_skills):
    """Compare skills found in the resume and job description."""
    resume_map = {
        skill.casefold(): skill for skill in resume_skills
    }

    job_map = {
        skill.casefold(): skill for skill in job_skills
    }

    matched = [
        job_map[key]
        for key in job_map
        if key in resume_map
    ]

    missing = [
        job_map[key]
        for key in job_map
        if key not in resume_map
    ]

    return (
        sorted(matched, key=str.casefold),
        sorted(missing, key=str.casefold),
    )


def calculate_match(resume_text, job_text):
    """Use the similarity function from skill_extractor.py."""
    result = calculate_similarity(resume_text, job_text)

    method = "Existing similarity method"

    if isinstance(result, (tuple, list)):
        if len(result) >= 2:
            score, method = result[0], result[1]
        elif len(result) == 1:
            score = result[0]
        else:
            score = 0.0
    else:
        score = result

    if hasattr(score, "item"):
        score = score.item()

    score = float(score)

    # Convert a 0–1 similarity score to a percentage.
    if 0 <= score <= 1:
        score *= 100

    score = max(0.0, min(100.0, score))

    return score, str(method)


def create_recommendations(missing_skills):
    """Get suggestions from the existing recommendation function."""
    if not missing_skills:
        return []

    try:
        suggestions = recommend_skills(missing_skills)

        if suggestions is None:
            suggestions = []

        if isinstance(suggestions, str):
            suggestions = [suggestions]

        elif isinstance(suggestions, dict):
            suggestions = [
                f"{skill}: {suggestion}"
                for skill, suggestion in suggestions.items()
            ]

        else:
            suggestions = list(suggestions)

        if suggestions:
            return [str(item) for item in suggestions]

    except Exception:
        pass

    return [
        f"Learn {skill} and practise it through a small project."
        for skill in missing_skills
    ]


def create_report(
    similarity,
    method,
    resume_skills,
    job_skills,
    matched_skills,
    missing_skills,
    recommendations,
):
    """Create a CSV report of the analysis."""
    rows = [
        {
            "Category": "Overall",
            "Item": "Resume–Job Similarity",
            "Details": f"{similarity:.2f}%",
        },
        {
            "Category": "Overall",
            "Item": "Similarity Method",
            "Details": method,
        },
    ]

    for skill in resume_skills:
        rows.append({
            "Category": "Resume Skill",
            "Item": skill,
            "Details": "Detected in resume",
        })

    for skill in job_skills:
        rows.append({
            "Category": "Required Skill",
            "Item": skill,
            "Details": "Detected in job description",
        })

    for skill in matched_skills:
        rows.append({
            "Category": "Matched Skill",
            "Item": skill,
            "Details": "Detected in both texts",
        })

    for skill in missing_skills:
        rows.append({
            "Category": "Skill Gap",
            "Item": skill,
            "Details": "Not detected in resume",
        })

    for suggestion in recommendations:
        rows.append({
            "Category": "Recommendation",
            "Item": suggestion,
            "Details": "Suggested learning activity",
        })

    return pd.DataFrame(rows)


# =========================================================
# LOAD SKILL CATALOG
# =========================================================

try:
    skills_catalog = load_skills()
except Exception as error:
    st.error(f"Unable to load the skill catalog: {error}")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.title("🎯 SkillMatch AI")

    st.write(
        "An AI-assisted platform for resume matching "
        "and skill-gap analysis."
    )

    st.divider()

    st.subheader("NLP techniques")

    st.markdown(
        """
        - PDF text extraction
        - Skill extraction and normalization
        - Text similarity
        - Cosine similarity (if implemented by the model)
        - Explainable skill-gap analysis
        """
    )

    st.divider()

    st.subheader("How it works")

    st.markdown(
        """
        1. Upload a resume.
        2. Enter a job description.
        3. Extract relevant skills.
        4. Compare skills and text similarity.
        5. Review the results and download a report.
        """
    )

    st.divider()

    st.caption("NLP Practical CEC Project")
    st.caption("A match score is not a hiring decision.")


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎯 SkillMatch AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Explainable Resume–Job Matching and Skill Gap Analysis'
    '</div>',
    unsafe_allow_html=True,
)

st.info(
    "Upload your resume and paste a job description to discover "
    "matching skills, potential gaps, and learning recommendations."
)


# =========================================================
# INPUT SECTION — ONLY ONE RESUME UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">1. Provide your information</div>',
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    st.subheader("📄 Resume")

    resume_file = st.file_uploader(
        "Upload your resume (PDF)",
        type=["pdf"],
        help="Upload a text-based PDF resume.",
        key="resume_upload",
    )

    if resume_file is not None:
        st.success(f"Uploaded: {resume_file.name}")

with right_col:
    st.subheader("💼 Job description")

    job_description = st.text_area(
        "Paste the complete job description",
        height=230,
        placeholder=(
            "Example: We are looking for a Python developer "
            "with knowledge of Python, Django, SQL, Git, "
            "REST APIs, and problem-solving..."
        ),
        key="job_description",
    )


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze_button = st.button(
    "🔍 Analyze Resume",
    type="primary",
    use_container_width=True,
    key="analyze_resume_button",
)


# =========================================================
# RUN ANALYSIS
# =========================================================

if analyze_button:

    if resume_file is None:
        st.warning("Please upload your resume PDF.")

    elif not job_description.strip():
        st.warning("Please enter a job description.")

    else:
        try:
            with st.spinner("Analyzing your resume..."):

                resume_text = clean_text(
                    extract_pdf_text(resume_file)
                )

                job_text = clean_text(job_description)

                if len(resume_text) < 30:
                    st.warning(
                        "Very little text was extracted. "
                        "If your PDF is scanned, OCR may be required."
                    )

                if not resume_text:
                    st.error(
                        "No readable text was found in the PDF."
                    )
                    st.stop()

                # Extract skills using the existing project module.
                resume_skills = normalize_skills(
                    extract_skills(resume_text, skills_catalog)
                )

                job_skills = normalize_skills(
                    extract_skills(job_text, skills_catalog)
                )

                # Compare the detected skills.
                matched_skills, missing_skills = compare_skills(
                    resume_skills,
                    job_skills,
                )

                # Calculate text similarity.
                similarity, similarity_method = calculate_match(
                    resume_text,
                    job_text,
                )

                # Generate recommendations.
                recommendations = create_recommendations(
                    missing_skills
                )

                # Store results so they remain visible on reruns.
                st.session_state["skillmatch_results"] = {
                    "resume_name": resume_file.name,
                    "similarity": similarity,
                    "similarity_method": similarity_method,
                    "resume_skills": resume_skills,
                    "job_skills": job_skills,
                    "matched_skills": matched_skills,
                    "missing_skills": missing_skills,
                    "recommendations": recommendations,
                    "analysis_time": datetime.now().strftime(
                        "%d-%m-%Y %H:%M:%S"
                    ),
                }

        except Exception as error:
            st.error("Analysis failed.")
            st.exception(error)


# =========================================================
# DISPLAY ANALYSIS RESULTS
# =========================================================

results = st.session_state.get("skillmatch_results")

if results:

    st.divider()

    st.markdown(
        '<div class="section-title">2. Analysis dashboard</div>',
        unsafe_allow_html=True,
    )

    similarity = results["similarity"]
    similarity_method = results["similarity_method"]
    resume_skills = results["resume_skills"]
    job_skills = results["job_skills"]
    matched_skills = results["matched_skills"]
    missing_skills = results["missing_skills"]
    recommendations = results["recommendations"]

    total_job_skills = len(job_skills)
    total_matched = len(matched_skills)
    total_missing = len(missing_skills)

    coverage = (
        total_matched / total_job_skills * 100
        if total_job_skills > 0
        else 0
    )

    # -----------------------------------------------------
    # SUMMARY METRICS
    # -----------------------------------------------------

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Text Similarity",
        f"{similarity:.1f}%",
    )

    metric2.metric(
        "Matched Skills",
        total_matched,
    )

    metric3.metric(
        "Skill Gaps",
        total_missing,
    )

    metric4.metric(
        "Resume Skills",
        len(resume_skills),
    )

    st.progress(
        int(round(similarity)),
        text=f"Resume–job text similarity: {similarity:.1f}%",
    )

    st.caption(
        f"Similarity method: {similarity_method}. "
        "This score is not the probability of getting hired."
    )

    # -----------------------------------------------------
    # SKILL COVERAGE CHART
    # -----------------------------------------------------

    st.divider()
    st.subheader("📊 Required Skill Coverage")

    chart_col, summary_col = st.columns(2, gap="large")

    with chart_col:
        if total_job_skills > 0:
            chart_df = pd.DataFrame({
                "Skill status": ["Matched", "Not detected"],
                "Number of skills": [
                    total_matched,
                    total_missing,
                ],
            })

            st.bar_chart(
                chart_df.set_index("Skill status"),
                horizontal=True,
            )
        else:
            st.info(
                "No skills were detected in the job description. "
                "Check whether the skill catalog contains relevant terms."
            )

    with summary_col:
        st.metric(
            "Required Skill Coverage",
            f"{coverage:.1f}%",
        )

        st.write(f"**Resume:** {results['resume_name']}")
        st.write(f"**Job skills detected:** {total_job_skills}")
        st.write(f"**Resume skills detected:** {len(resume_skills)}")
        st.write(f"**Analysis time:** {results['analysis_time']}")

        if total_job_skills == 0:
            st.warning(
                "Skill coverage cannot be calculated because "
                "no job-description skills were detected."
            )

    # -----------------------------------------------------
    # MATCHED SKILLS
    # -----------------------------------------------------

    st.divider()
    st.subheader("✅ Matching Skills")

    if matched_skills:
        st.success(
            f"{total_matched} skill(s) were detected in both "
            "the resume and job description."
        )

        matched_columns = st.columns(3)

        for index, skill in enumerate(matched_skills):
            with matched_columns[index % 3]:
                st.markdown(f"- ✅ {skill}")
    else:
        st.info(
            "No matching catalog skills were detected. "
            "Different wording or a limited skill catalog may affect results."
        )

    # -----------------------------------------------------
    # SKILL GAP ANALYSIS
    # -----------------------------------------------------

    st.divider()
    st.subheader("📌 Skill Gap Analysis")

    if missing_skills:
        st.warning(
            f"{total_missing} job requirement skill(s) were not "
            "detected in the resume."
        )

        for skill in missing_skills:
            st.markdown(f"- ⚠️ {skill}")

        st.caption(
            "A missing skill means the current extractor did not find it "
            "in your resume. It does not prove that you lack the skill."
        )

    elif total_job_skills > 0:
        st.success(
            "All detected job-description skills were also detected "
            "in the resume."
        )

    # -----------------------------------------------------
    # RESUME SKILLS
    # -----------------------------------------------------

    st.divider()
    st.subheader("🧠 Extracted Resume Skills")

    if resume_skills:
        with st.expander(
            f"View all {len(resume_skills)} detected skills",
            expanded=True,
        ):
            st.write(", ".join(resume_skills))
    else:
        st.warning(
            "No catalog skills were detected in the resume. "
            "Review data/skills.csv and skill_extractor.py."
        )

    # -----------------------------------------------------
    # JOB DESCRIPTION SKILLS
    # -----------------------------------------------------

    st.subheader("💼 Extracted Job Skills")

    if job_skills:
        with st.expander(
            f"View all {len(job_skills)} required skills"
        ):
            st.write(", ".join(job_skills))
    else:
        st.info(
            "No catalog skills were detected in the job description."
        )

    # -----------------------------------------------------
    # RECOMMENDATIONS
    # -----------------------------------------------------

    st.divider()
    st.subheader("🚀 Recommended Next Steps")

    if recommendations:
        for index, suggestion in enumerate(recommendations, start=1):
            st.markdown(f"{index}. {suggestion}")

    elif missing_skills:
        for index, skill in enumerate(missing_skills, start=1):
            st.markdown(
                f"{index}. Learn {skill} and practise it through a project."
            )

    else:
        st.success(
            "There are no detected skill gaps requiring recommendations."
        )

    # -----------------------------------------------------
    # CSV DOWNLOAD
    # -----------------------------------------------------

    st.divider()
    st.subheader("📥 Download Analysis Report")

    report_df = create_report(
        similarity=similarity,
        method=similarity_method,
        resume_skills=resume_skills,
        job_skills=job_skills,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        recommendations=recommendations,
    )

    st.download_button(
        label="⬇️ Download CSV Report",
        data=report_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="skillmatch_ai_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Preview report"):
        st.dataframe(
            report_df,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "SkillMatch AI is a prototype decision-support tool. "
        "Review its results before making career decisions."
    )

else:

    # -----------------------------------------------------
    # INITIAL DASHBOARD
    # -----------------------------------------------------

    st.divider()
    st.markdown(
        '<div class="section-title">What you will get</div>',
        unsafe_allow_html=True,
    )

    feature1, feature2, feature3 = st.columns(3)

    with feature1:
        st.subheader("📊 Text Similarity")
        st.write(
            "Compare resume and job-description text using "
            "your project's existing similarity function."
        )

    with feature2:
        st.subheader("🧩 Skill Gap Analysis")
        st.write(
            "Identify required skills that were not detected "
            "in the resume."
        )

    with feature3:
        st.subheader("📥 CSV Report")
        st.write(
            "Download a report containing matching skills, "
            "skill gaps, and recommendations."
        )

    st.info(
        "Upload your resume and enter a job description above, "
        "then click Analyze Resume."
    )
