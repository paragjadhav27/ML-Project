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


# ─── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf = joblib.load("best_random_forest_model.joblib")
    xgb = joblib.load("best_xgboost_model.joblib")
    return rf, xgb


# ─── Feature engineering (mirrors training pipeline) ─────────────────────────
def build_feature_vector(inputs: dict, feature_names: list) -> pd.DataFrame:
    """Recreate the exact 99-feature vector the models expect."""
    
    # Binary label encoding for yes/no columns
    binary_map = {"yes": 1, "no": 0}
    skill_level_map = {"poor": 0, "medium": 1, "excellent": 2}
    
    row = {}

    # Directly label-encoded columns
    row["self-learning capability?"] = binary_map[inputs["self_learning"]]
    row["Extra-courses did"]          = binary_map[inputs["extra_courses"]]
    row["reading and writing skills"] = skill_level_map[inputs["rw_skills"]]
    row["memory capability score"]    = skill_level_map[inputs["memory"]]
    row["Taken inputs from seniors or elders"] = binary_map[inputs["senior_inputs"]]
    row["worked in teams ever?"]      = binary_map[inputs["team_work"]]
    row["Introvert"]                  = binary_map[inputs["introvert"]]

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
    lq  = inputs["logical_quotient"]
    hk  = inputs["hackathons"]
    cs  = inputs["coding_skills"]
    ps  = inputs["public_speaking"]

    row["Logical quotient rating"] = lq
    row["hackathons"]              = hk
    row["coding skills rating"]    = cs
    row["public speaking points"]  = ps

    # Polynomial features (degree-2 interactions)
    row["Logical quotient rating^2"]                  = lq ** 2
    row["Logical quotient rating hackathons"]          = lq * hk
    row["Logical quotient rating coding skills rating"]= lq * cs
    row["Logical quotient rating public speaking points"] = lq * ps
    row["hackathons^2"]                               = hk ** 2
    row["hackathons coding skills rating"]            = hk * cs
    row["hackathons public speaking points"]          = hk * ps
    row["coding skills rating^2"]                     = cs ** 2
    row["coding skills rating public speaking points"]= cs * ps
    row["public speaking points^2"]                   = ps ** 2

    df = pd.DataFrame([row])
    # Reorder to match training feature order
    df = df[feature_names]
    return df


# ─── App layout ──────────────────────────────────────────────────────────────
def main():
    rf_model, xgb_model = load_models()
    feature_names = rf_model.feature_names_in_.tolist()

    # Sidebar — model selector
    with st.sidebar:
        st.markdown('<p class="section-label">⚡ Configuration</p>', unsafe_allow_html=True)
        model_choice = st.radio(
            "Choose prediction model",
            ["Random Forest", "XGBoost", "Ensemble (Both)"],
            index=2,
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
        # --- Numeric Skills ---
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
        self_learning  = st.radio("Self-Learning Capability?", ["yes", "no"], horizontal=True)
        extra_courses  = st.radio("Completed Extra Courses?",  ["yes", "no"], horizontal=True)
        senior_inputs  = st.radio("Taken Inputs from Seniors?", ["yes", "no"], horizontal=True)
        team_work      = st.radio("Worked in Teams?", ["yes", "no"], horizontal=True)
        introvert      = st.radio("Are You an Introvert?", ["yes", "no"], horizontal=True)

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

        X = build_feature_vector(inputs, feature_names)

        # Predictions
        rf_pred   = rf_model.predict(X)[0]
        rf_proba  = rf_model.predict_proba(X)[0]
        xgb_pred  = xgb_model.predict(X)[0]
        xgb_proba = xgb_model.predict_proba(X)[0]

        rf_role   = JOB_ROLES[rf_pred]
        xgb_role  = JOB_ROLES[xgb_pred]

        # Ensemble = average probabilities
        avg_proba   = (rf_proba + xgb_proba) / 2
        ens_pred    = int(np.argmax(avg_proba))
        ens_conf    = float(avg_proba[ens_pred])
        ens_role    = JOB_ROLES[ens_pred]

        # Choose final role based on model_choice
        if model_choice == "Random Forest":
            final_role = rf_role
            final_conf = float(rf_proba[rf_pred])
            model_label = "Random Forest"
        elif model_choice == "XGBoost":
            final_role = xgb_role
            final_conf = float(xgb_proba[xgb_pred])
            model_label = "XGBoost"
        else:
            final_role = ens_role
            final_conf = ens_conf
            model_label = "Ensemble (RF + XGBoost)"

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
            st.markdown('<p class="section-label">📊 Model Breakdown</p>', unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            with m1:
                rf_c = float(rf_proba[rf_pred])
                st.markdown(
                    f'<div class="metric-pill">'
                    f'<div class="metric-val">{rf_c*100:.0f}%</div>'
                    f'<div class="metric-lbl">Random Forest</div>'
                    f'<div style="font-size:0.75rem; color:#c4b5fd; margin-top:4px;">{ROLE_ICONS.get(rf_role,"")} {rf_role}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with m2:
                xgb_c = float(xgb_proba[xgb_pred])
                st.markdown(
                    f'<div class="metric-pill">'
                    f'<div class="metric-val">{xgb_c*100:.0f}%</div>'
                    f'<div class="metric-lbl">XGBoost</div>'
                    f'<div style="font-size:0.75rem; color:#c4b5fd; margin-top:4px;">{ROLE_ICONS.get(xgb_role,"")} {xgb_role}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Top-3 from ensemble
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-label">🔝 Top Alternatives</p>', unsafe_allow_html=True)
            top3_idx = np.argsort(avg_proba)[::-1][:3]
            for rank, idx in enumerate(top3_idx, 1):
                role = JOB_ROLES[idx]
                prob = avg_proba[idx]
                bar_w = int(prob * 100)
                st.markdown(
                    f'<div style="margin-bottom:8px;">'
                    f'<div style="display:flex;justify-content:space-between;font-size:0.8rem;color:rgba(255,255,255,0.7);">'
                    f'<span>{ROLE_ICONS.get(role,"")} {role}</span>'
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
