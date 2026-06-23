import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Career Path Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Cards */
    .card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Hero */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
        margin-bottom: 8px;
    }
    
    .hero-sub {
        font-size: 1.05rem;
        color: rgba(255,255,255,0.55);
        margin-bottom: 0;
    }
    
    /* Section headers */
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #a78bfa;
        margin-bottom: 12px;
        margin-top: 4px;
    }
    
    /* Result box */
    .result-box {
        background: linear-gradient(135deg, rgba(167,139,250,0.2), rgba(96,165,250,0.2));
        border: 1.5px solid rgba(167,139,250,0.5);
        border-radius: 20px;
        padding: 32px 28px;
        text-align: center;
    }
    
    .result-role {
        font-size: 2rem;
        font-weight: 800;
        color: #e2e8f0;
        margin: 8px 0 4px;
    }
    
    .result-badge {
        display: inline-block;
        background: rgba(167,139,250,0.25);
        color: #c4b5fd;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        padding: 4px 14px;
        border-radius: 99px;
        border: 1px solid rgba(167,139,250,0.4);
        margin-bottom: 14px;
    }
    
    /* Course cards */
    .course-card {
        background: rgba(52,211,153,0.1);
        border: 1px solid rgba(52,211,153,0.3);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
        color: #d1fae5;
        font-size: 0.9rem;
    }
    
    .course-card strong {
        display: block;
        color: #6ee7b7;
        font-size: 0.95rem;
        margin-bottom: 2px;
    }
    
    /* Metric cards */
    .metric-pill {
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 8px;
    }
    
    .metric-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #60a5fa;
    }
    
    .metric-lbl {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.45);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    /* Divider */
    hr.custom {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 20px 0;
    }
    
    /* Streamlit element overrides */
    .stSelectbox > div > div,
    .stSlider > div {
        color: #e2e8f0 !important;
    }
    
    label, .stRadio label {
        color: rgba(255,255,255,0.8) !important;
        font-size: 0.88rem !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #3b82f6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 0;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
        letter-spacing: 0.03em;
        transition: opacity 0.2s;
    }
    
    .stButton > button:hover {
        opacity: 0.88;
        color: white;
        border: none;
    }

    /* Confidence bar */
    .conf-bar-bg {
        background: rgba(255,255,255,0.1);
        border-radius: 99px;
        height: 8px;
        overflow: hidden;
        margin-top: 6px;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        transition: width 0.5s ease;
    }
    
    p, li {
        color: rgba(255,255,255,0.75);
    }
</style>
""", unsafe_allow_html=True)


# ─── Constants ───────────────────────────────────────────────────────────────
JOB_ROLES = [
    'Applications Developer', 'CRM Technical Developer', 'Database Developer',
    'Mobile Applications Developer', 'Network Security Engineer', 'Software Developer',
    'Software Engineer', 'Software Quality Assurance (QA) / Testing',
    'Systems Security Administrator', 'Technical Support', 'UX Designer', 'Web Developer'
]

COURSE_RECOMMENDATIONS = {
    'Applications Developer': [
        ("Full-Stack Web Development", "HTML, CSS, JavaScript, React, Node.js — the complete web stack."),
        ("Java / Kotlin for Android", "Build production-grade Android apps from scratch."),
        ("Software Design Patterns", "MVC, MVVM, microservices — architecture that scales."),
    ],
    'CRM Technical Developer': [
        ("Salesforce Developer Fundamentals", "Apex, SOQL, Lightning Web Components on the #1 CRM platform."),
        ("Microsoft Dynamics 365", "CRM/ERP customisation with Power Platform integrations."),
        ("Database Management & SQL", "Master relational databases essential for CRM backends."),
    ],
    'Database Developer': [
        ("Advanced SQL & Query Optimization", "Indexing, query plans, stored procedures, and performance tuning."),
        ("NoSQL Databases", "MongoDB, Cassandra, Redis — when relational isn't enough."),
        ("Database Administration (DBA)", "PostgreSQL / MySQL administration, backups, and replication."),
    ],
    'Mobile Applications Developer': [
        ("Flutter & Dart", "Cross-platform mobile apps for iOS and Android from one codebase."),
        ("React Native", "JavaScript-powered native mobile development by Meta."),
        ("iOS Development with Swift", "SwiftUI fundamentals to App Store deployment."),
    ],
    'Network Security Engineer': [
        ("CompTIA Security+", "Globally recognised baseline for cybersecurity professionals."),
        ("Ethical Hacking & Penetration Testing", "CEH / OSCP — think like an attacker to defend better."),
        ("Network Fundamentals (CCNA)", "TCP/IP, routing, switching, and firewall configuration."),
    ],
    'Software Developer': [
        ("Python for Software Development", "Scripting, OOP, libraries, packaging, and clean code."),
        ("Data Structures & Algorithms", "The foundation every software developer must master."),
        ("Agile & Scrum Methodology", "Ship faster with iterative, team-based development."),
    ],
    'Software Engineer': [
        ("System Design & Architecture", "Design scalable, distributed systems like FAANG engineers."),
        ("Computer Science Fundamentals", "OS, compilers, networks — the CS core that never goes stale."),
        ("Cloud Engineering (AWS / GCP)", "Architect and deploy production workloads on the cloud."),
    ],
    'Software Quality Assurance (QA) / Testing': [
        ("Software Testing Fundamentals", "Manual testing, test planning, bug lifecycle, SDLC."),
        ("Selenium & Automation Testing", "End-to-end test automation with Python / Java bindings."),
        ("API Testing with Postman", "REST API validation, collections, environment management."),
    ],
    'Systems Security Administrator': [
        ("Linux System Administration", "CLI mastery, user management, hardening, and scripting."),
        ("CISSP / CISM Certification", "Senior-level information security governance and risk."),
        ("Cloud Security (AWS Security)", "IAM, VPC, encryption, and compliance in cloud environments."),
    ],
    'Technical Support': [
        ("IT Support Professional (Google)", "Google's foundational IT support course on Coursera."),
        ("ITIL 4 Foundation", "Service management framework used by enterprises worldwide."),
        ("Networking Basics", "Understand the infrastructure you'll be supporting daily."),
    ],
    'UX Designer': [
        ("UI/UX Design Fundamentals (Figma)", "Wireframing, prototyping, and user-testing in Figma."),
        ("Human-Computer Interaction", "Cognitive principles behind intuitive product design."),
        ("Design Systems & Accessibility", "Build scalable, WCAG-compliant design systems."),
    ],
    'Web Developer': [
        ("The Complete Web Developer Bootcamp", "HTML, CSS, JavaScript, React, Node, MongoDB end-to-end."),
        ("Advanced CSS & Sass", "Flexbox, Grid, animations, and production-level CSS architecture."),
        ("Next.js / TypeScript", "Modern, SSR-first React framework for professional web apps."),
    ],
}

ROLE_ICONS = {
    'Applications Developer': '📱',
    'CRM Technical Developer': '🤝',
    'Database Developer': '🗄️',
    'Mobile Applications Developer': '📲',
    'Network Security Engineer': '🔐',
    'Software Developer': '💻',
    'Software Engineer': '⚙️',
    'Software Quality Assurance (QA) / Testing': '🧪',
    'Systems Security Administrator': '🛡️',
    'Technical Support': '🎧',
    'UX Designer': '🎨',
    'Web Developer': '🌐',
}

MODEL_OPTIONS = [
    "Random Forest",
    "XGBoost",
    "SVM (RBF Kernel)",
    "SVM (Linear Kernel)",
    "LightGBM",
    "CatBoost",
    "Ensemble (All Models)",
]

# Models that support predict_proba
PROBA_MODELS = {"Random Forest", "XGBoost", "LightGBM", "CatBoost"}


# ─── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {
        "Random Forest":       joblib.load("tuned_random_forest_model.joblib"),
        "XGBoost":             joblib.load("tuned_xgboost_model.joblib"),
        "SVM (RBF Kernel)":    joblib.load("svm_rbf_kernel_model.joblib"),
        "SVM (Linear Kernel)": joblib.load("svm_linear_model.joblib"),
        "LightGBM":            joblib.load("lightgbm_model.joblib"),
        "CatBoost":            joblib.load("catboost_model.joblib"),
    }
    return models


# ─── Feature vector builders ─────────────────────────────────────────────────

def _build_base_row(inputs: dict) -> tuple[dict, int, int, int, int]:
    """Build the shared 89-feature base row (spaces as separators)."""
    binary_map     = {"yes": 1, "no": 0}
    skill_level_map = {"poor": 0, "medium": 1, "excellent": 2}

    row = {}

    # Scalar encoded fields
    row["self-learning capability?"]           = binary_map[inputs["self_learning"]]
    row["Extra-courses did"]                   = binary_map[inputs["extra_courses"]]
    row["reading and writing skills"]          = skill_level_map[inputs["rw_skills"]]
    row["memory capability score"]             = skill_level_map[inputs["memory"]]
    row["Taken inputs from seniors or elders"] = binary_map[inputs["senior_inputs"]]
    row["worked in teams ever?"]               = binary_map[inputs["team_work"]]
    row["Introvert"]                           = binary_map[inputs["introvert"]]

    # One-hot: certifications
    for val in ['app development', 'distro making', 'full stack', 'hadoop',
                'information security', 'machine learning', 'python',
                'r programming', 'shell programming']:
        row[f"certifications_{val}"] = 1 if inputs["certifications"] == val else 0

    # One-hot: workshops
    for val in ['cloud computing', 'data science', 'database security',
                'game development', 'hacking', 'system designing',
                'testing', 'web technologies']:
        row[f"workshops_{val}"] = 1 if inputs["workshops"] == val else 0

    # One-hot: Interested subjects
    for val in ['Computer Architecture', 'IOT', 'Management', 'Software Engineering',
                'cloud computing', 'data engineering', 'hacking', 'networks',
                'parallel computing', 'programming']:
        row[f"Interested subjects_{val}"] = 1 if inputs["interested_subjects"] == val else 0

    # One-hot: interested career area
    for val in ['Business process analyst', 'cloud computing', 'developer',
                'security', 'system developer', 'testing']:
        row[f"interested career area_{val}"] = 1 if inputs["career_area"] == val else 0

    # One-hot: Type of company
    for val in ['BPA', 'Cloud Services', 'Finance', 'Product based', 'SAaS services',
                'Sales and Marketing', 'Service Based',
                'Testing and Maintainance Services', 'Web Services', 'product development']:
        row[f"Type of company want to settle in?_{val}"] = 1 if inputs["company_type"] == val else 0

    # One-hot: Books
    books = ['Action and Adventure', 'Anthology', 'Art', 'Autobiographies', 'Biographies',
             'Childrens', 'Comics', 'Cookbooks', 'Diaries', 'Dictionaries', 'Drama',
             'Encyclopedias', 'Fantasy', 'Guide', 'Health', 'History', 'Horror',
             'Journals', 'Math', 'Mystery', 'Poetry', 'Prayer books',
             'Religion-Spirituality', 'Romance', 'Satire', 'Science',
             'Science fiction', 'Self help', 'Series', 'Travel', 'Trilogy']
    for val in books:
        row[f"Interested Type of Books_{val}"] = 1 if inputs["books"] == val else 0

    # One-hot: Management or Technical
    row["Management or Technical_Management"] = 1 if inputs["mgmt_tech"] == "Management" else 0
    row["Management or Technical_Technical"]  = 1 if inputs["mgmt_tech"] == "Technical"  else 0

    # One-hot: hard/smart worker
    row["hard/smart worker_hard worker"]  = 1 if inputs["work_style"] == "hard worker" else 0
    row["hard/smart worker_smart worker"] = 1 if inputs["work_style"] == "smart worker" else 0

    # Numeric
    lq = inputs["logical_quotient"]
    hk = inputs["hackathons"]
    cs = inputs["coding_skills"]
    ps = inputs["public_speaking"]

    row["Logical quotient rating"] = lq
    row["hackathons"]              = hk
    row["coding skills rating"]    = cs
    row["public speaking points"]  = ps

    return row, lq, hk, cs, ps


def build_feature_vector_rf(inputs: dict, feature_names: list) -> pd.DataFrame:
    """99-feature vector for Random Forest (includes polynomial interactions)."""
    row, lq, hk, cs, ps = _build_base_row(inputs)

    # Polynomial degree-2 interaction terms
    row["Logical quotient rating^2"]                     = lq ** 2
    row["Logical quotient rating hackathons"]            = lq * hk
    row["Logical quotient rating coding skills rating"]  = lq * cs
    row["Logical quotient rating public speaking points"]= lq * ps
    row["hackathons^2"]                                  = hk ** 2
    row["hackathons coding skills rating"]               = hk * cs
    row["hackathons public speaking points"]             = hk * ps
    row["coding skills rating^2"]                        = cs ** 2
    row["coding skills rating public speaking points"]   = cs * ps
    row["public speaking points^2"]                      = ps ** 2

    df = pd.DataFrame([row])
    return df[feature_names]


def build_feature_vector_standard(inputs: dict, feature_names: list) -> pd.DataFrame:
    """89-feature vector for XGBoost, SVM (RBF/Linear), CatBoost (space-separated names)."""
    row, _, _, _, _ = _build_base_row(inputs)
    df = pd.DataFrame([row])
    return df[feature_names]


def build_feature_vector_lgbm(inputs: dict, feature_names: list) -> pd.DataFrame:
    """89-feature vector for LightGBM (underscore-separated names)."""
    row, _, _, _, _ = _build_base_row(inputs)

    # Rename keys: replace spaces with underscores to match LightGBM training names
    lgbm_row = {k.replace(" ", "_"): v for k, v in row.items()}

    df = pd.DataFrame([lgbm_row])
    return df[feature_names]


def predict_single(model, model_name: str, inputs: dict):
    """
    Returns (predicted_class_index, confidence_0_to_1, proba_array_or_None).

    - Models with predict_proba: returns full probability array.
    - SVM models (no probability): uses decision_function, softmax-normalises
      the scores to approximate a probability distribution for the UI.
    """
    # Build the correct feature vector
    if model_name == "Random Forest":
        X = build_feature_vector_rf(inputs, model.feature_names_in_.tolist())
    elif model_name == "LightGBM":
        X = build_feature_vector_lgbm(inputs, model.feature_names_in_.tolist())
    elif model_name == "CatBoost":
        # CatBoost uses feature_names_ (not feature_names_in_)
        feat_names = model.feature_names_
        X = build_feature_vector_standard(inputs, feat_names)
    else:
        # XGBoost, SVM RBF, SVM Linear
        X = build_feature_vector_standard(inputs, model.feature_names_in_.tolist())

    if model_name in PROBA_MODELS:
        proba = model.predict_proba(X)[0]
        pred  = int(np.argmax(proba))
        conf  = float(proba[pred])
        return pred, conf, proba
    else:
        # SVM: use decision_function scores; softmax to get pseudo-probabilities
        scores = model.decision_function(X)[0]
        # Softmax normalisation
        e = np.exp(scores - scores.max())
        proba = e / e.sum()
        pred  = int(np.argmax(proba))
        conf  = float(proba[pred])
        return pred, conf, proba


# ─── App layout ──────────────────────────────────────────────────────────────
def main():
    models = load_models()

    # Sidebar — model selector
    with st.sidebar:
        st.markdown('<p class="section-label">⚡ Configuration</p>', unsafe_allow_html=True)
        model_choice = st.radio(
            "Choose prediction model",
            MODEL_OPTIONS,
            index=6,  # Default: Ensemble
        )
        st.markdown("---")
        st.markdown('<p class="section-label">ℹ️ About</p>', unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.82rem; color:rgba(255,255,255,0.5);'>"
            "This app uses machine learning models trained on career survey data "
            "to suggest your best-fit job role and relevant learning resources."
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown('<p class="section-label">🤖 Models</p>', unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.78rem; color:rgba(255,255,255,0.45);'>"
            "Random Forest · XGBoost · SVM (RBF) · SVM (Linear) · LightGBM · CatBoost"
            "</p>",
            unsafe_allow_html=True,
        )

    # Hero
    st.markdown(
        '<div class="hero-title">Career Path Predictor 🎯</div>'
        '<p class="hero-sub">Fill in your profile below and discover which tech role suits you best.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='custom'>", unsafe_allow_html=True)

    # ── Two-column form layout ────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<p class="section-label">📊 Skills & Ratings</p>', unsafe_allow_html=True)
        with st.container():
            logical_quotient = st.slider("Logical Quotient Rating", 1, 9, 5)
            hackathons        = st.slider("Hackathons Participated", 0, 6, 2)
            coding_skills     = st.slider("Coding Skills Rating", 1, 9, 5)
            public_speaking   = st.slider("Public Speaking Points", 1, 9, 4)

        st.markdown('<p class="section-label" style="margin-top:20px;">🧠 Abilities</p>', unsafe_allow_html=True)
        rw_skills = st.select_slider(
            "Reading & Writing Skills",
            options=["poor", "medium", "excellent"], value="medium"
        )
        memory = st.select_slider(
            "Memory Capability Score",
            options=["poor", "medium", "excellent"], value="medium"
        )

        st.markdown('<p class="section-label" style="margin-top:20px;">📚 Learning Profile</p>', unsafe_allow_html=True)
        certifications = st.selectbox("Certification Area", [
            'app development', 'distro making', 'full stack', 'hadoop',
            'information security', 'machine learning', 'python',
            'r programming', 'shell programming'
        ])
        workshops = st.selectbox("Workshop Attended", [
            'cloud computing', 'data science', 'database security',
            'game development', 'hacking', 'system designing',
            'testing', 'web technologies'
        ])
        interested_subjects = st.selectbox("Interested Subject", [
            'Computer Architecture', 'IOT', 'Management', 'Software Engineering',
            'cloud computing', 'data engineering', 'hacking', 'networks',
            'parallel computing', 'programming'
        ])

    with col_right:
        st.markdown('<p class="section-label">🏢 Career Preferences</p>', unsafe_allow_html=True)
        career_area = st.selectbox("Interested Career Area", [
            'Business process analyst', 'cloud computing', 'developer',
            'security', 'system developer', 'testing'
        ])
        company_type = st.selectbox("Preferred Company Type", [
            'BPA', 'Cloud Services', 'Finance', 'Product based', 'SAaS services',
            'Sales and Marketing', 'Service Based',
            'Testing and Maintainance Services', 'Web Services', 'product development'
        ])
        mgmt_tech  = st.radio("Orientation", ["Management", "Technical"], horizontal=True)
        work_style = st.radio("Work Style", ["hard worker", "smart worker"], horizontal=True)

        st.markdown('<p class="section-label" style="margin-top:20px;">🧩 Personal Traits</p>', unsafe_allow_html=True)
        self_learning = st.radio("Self-Learning Capability?", ["yes", "no"], horizontal=True)
        extra_courses = st.radio("Completed Extra Courses?",  ["yes", "no"], horizontal=True)
        senior_inputs = st.radio("Taken Inputs from Seniors?", ["yes", "no"], horizontal=True)
        team_work     = st.radio("Worked in Teams?", ["yes", "no"], horizontal=True)
        introvert     = st.radio("Are You an Introvert?", ["yes", "no"], horizontal=True)

        st.markdown('<p class="section-label" style="margin-top:20px;">📖 Reading Preference</p>', unsafe_allow_html=True)
        books = st.selectbox("Favourite Type of Books", [
            'Action and Adventure', 'Anthology', 'Art', 'Autobiographies', 'Biographies',
            'Childrens', 'Comics', 'Cookbooks', 'Diaries', 'Dictionaries', 'Drama',
            'Encyclopedias', 'Fantasy', 'Guide', 'Health', 'History', 'Horror',
            'Journals', 'Math', 'Mystery', 'Poetry', 'Prayer books',
            'Religion-Spirituality', 'Romance', 'Satire', 'Science',
            'Science fiction', 'Self help', 'Series', 'Travel', 'Trilogy'
        ])

    # ── Predict button ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    predict_col, _ = st.columns([1, 2])
    with predict_col:
        predict_clicked = st.button("🔍  Predict My Career Path")

    # ── Results ───────────────────────────────────────────────────────────────
    if predict_clicked:
        inputs = dict(
            logical_quotient=logical_quotient,
            hackathons=hackathons,
            coding_skills=coding_skills,
            public_speaking=public_speaking,
            rw_skills=rw_skills,
            memory=memory,
            certifications=certifications,
            workshops=workshops,
            interested_subjects=interested_subjects,
            career_area=career_area,
            company_type=company_type,
            mgmt_tech=mgmt_tech,
            work_style=work_style,
            self_learning=self_learning,
            extra_courses=extra_courses,
            senior_inputs=senior_inputs,
            team_work=team_work,
            introvert=introvert,
            books=books,
        )

        # Run all 6 models
        results = {}
        for name, mdl in models.items():
            pred, conf, proba = predict_single(mdl, name, inputs)
            results[name] = {"pred": pred, "conf": conf, "proba": proba, "role": JOB_ROLES[pred]}

        # Ensemble = average probabilities across all models
        all_probas = np.array([results[n]["proba"] for n in models])
        avg_proba  = all_probas.mean(axis=0)
        ens_pred   = int(np.argmax(avg_proba))
        ens_conf   = float(avg_proba[ens_pred])
        ens_role   = JOB_ROLES[ens_pred]

        # Choose final result
        if model_choice == "Ensemble (All Models)":
            final_role  = ens_role
            final_conf  = ens_conf
            model_label = "Ensemble (6 Models)"
        else:
            r           = results[model_choice]
            final_role  = r["role"]
            final_conf  = r["conf"]
            model_label = model_choice

        icon = ROLE_ICONS.get(final_role, "🎯")

        st.markdown("<hr class='custom'>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">🏆 Prediction Result</p>', unsafe_allow_html=True)

        res_col, meta_col = st.columns([3, 2], gap="large")

        with res_col:
            st.markdown(
                f"""
                <div class="result-box">
                    <span style="font-size:3rem;">{icon}</span>
                    <div class="result-role">{final_role}</div>
                    <div class="result-badge">Predicted by {model_label}</div>
                    <div style="margin-top:10px;">
                        <span style="color:rgba(255,255,255,0.5); font-size:0.85rem;">Confidence</span>
                        <div style="font-size:1.5rem; font-weight:800; color:#a78bfa;">{final_conf*100:.1f}%</div>
                        <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{final_conf*100:.1f}%;"></div></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with meta_col:
            st.markdown('<p class="section-label">📊 All Models Breakdown</p>', unsafe_allow_html=True)

            model_display_order = [
                ("Random Forest", "RF"),
                ("XGBoost",       "XGB"),
                ("LightGBM",      "LGBM"),
                ("CatBoost",      "CatBoost"),
                ("SVM (RBF Kernel)",    "SVM-RBF"),
                ("SVM (Linear Kernel)", "SVM-Lin"),
            ]

            col_a, col_b = st.columns(2)
            for i, (name, short) in enumerate(model_display_order):
                r = results[name]
                note = "~ approx" if "SVM" in name else ""
                target_col = col_a if i % 2 == 0 else col_b
                with target_col:
                    st.markdown(
                        f'<div class="metric-pill">'
                        f'<div class="metric-val">{r["conf"]*100:.0f}%</div>'
                        f'<div class="metric-lbl">{short} {note}</div>'
                        f'<div style="font-size:0.72rem; color:#c4b5fd; margin-top:4px;">'
                        f'{ROLE_ICONS.get(r["role"], "")} {r["role"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Top-3 from ensemble
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-label">🔝 Top Alternatives (Ensemble)</p>', unsafe_allow_html=True)
            top3_idx = np.argsort(avg_proba)[::-1][:3]
            for rank, idx in enumerate(top3_idx, 1):
                role  = JOB_ROLES[idx]
                prob  = avg_proba[idx]
                bar_w = int(prob * 100)
                st.markdown(
                    f'<div style="margin-bottom:8px;">'
                    f'<div style="display:flex;justify-content:space-between;font-size:0.8rem;color:rgba(255,255,255,0.7);">'
                    f'<span>{ROLE_ICONS.get(role, "")} {role}</span>'
                    f'<span style="color:#a78bfa;font-weight:700;">{prob*100:.1f}%</span></div>'
                    f'<div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{bar_w}%;"></div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Recommended Courses ───────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">🎓 Recommended Courses for You</p>', unsafe_allow_html=True)

        courses = COURSE_RECOMMENDATIONS.get(final_role, [])
        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        for i, (title, desc) in enumerate(courses):
            with cols[i % 3]:
                st.markdown(
                    f'<div class="course-card"><strong>📘 {title}</strong>{desc}</div>',
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
