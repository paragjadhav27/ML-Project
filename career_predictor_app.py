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
    "LightGBM",
    "CatBoost",
    "Ensemble (All Models)",
]

# Models that support predict_proba (all remaining models do)
PROBA_MODELS = {"Random Forest", "XGBoost", "LightGBM", "CatBoost"}


# ─── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {
        "Random Forest": joblib.load("best_random_forest_model.joblib"),
        "XGBoost":       joblib.load("best_xgboost_model.joblib"),
        "LightGBM":      joblib.load("lightgbm_model.joblib"),
        "CatBoost":      joblib.load("catboost_model.joblib"),
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


def build_feature_vector_standard(inputs: dict, feature_names: list) -> pd.DataFrame:
    """89-feature vector for RF, XGBoost, CatBoost (space-separated names)."""
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
    """Returns (predicted_class_index, confidence_0_to_1, proba_array)."""
    if model_name == "LightGBM":
        X = build_feature_vector_lgbm(inputs, model.feature_names_in_.tolist())
    elif model_name == "CatBoost":
        X = build_feature_vector_standard(inputs, model.feature_names_)
    else:
        # Random Forest, XGBoost
        X = build_feature_vector_standard(inputs, model.feature_names_in_.tolist())

    proba = model.predict_proba(X)[0]
    pred  = int(np.argmax(proba))
    conf  = float(proba[pred])
    return pred, conf, proba


# ─── Step wizard helpers ─────────────────────────────────────────────────────

STEP_META = [
    {"title": "Skills & Ratings + Abilities",          "icon": "📊", "step": 1},
    {"title": "Learning Profile + Career Preferences", "icon": "📚", "step": 2},
    {"title": "Personal Traits + Reading Preference",  "icon": "🧩", "step": 3},
]

def render_progress(current_step: int):
    """Render a 3-step progress bar at the top of the form."""
    labels = [m["icon"] + " " + m["title"] for m in STEP_META]
    cols = st.columns(3)
    for i, (col, label) in enumerate(zip(cols, labels)):
        step_num = i + 1
        if step_num < current_step:
            colour = "#34d399"   # completed — green
            txt_col = "#34d399"
            dot = "✓"
        elif step_num == current_step:
            colour = "#a78bfa"   # active — purple
            txt_col = "#e2e8f0"
            dot = str(step_num)
        else:
            colour = "rgba(255,255,255,0.15)"  # future — dim
            txt_col = "rgba(255,255,255,0.35)"
            dot = str(step_num)

        with col:
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;
                            border-radius:12px;border:1.5px solid {colour};
                            background:{'rgba(167,139,250,0.1)' if step_num==current_step else 'transparent'};">
                    <div style="width:28px;height:28px;border-radius:50%;background:{colour};
                                display:flex;align-items:center;justify-content:center;
                                font-size:0.8rem;font-weight:800;color:#0f0c29;flex-shrink:0;">
                        {dot}
                    </div>
                    <span style="font-size:0.78rem;font-weight:600;color:{txt_col};line-height:1.3;">
                        {label}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)


def nav_buttons(step: int, can_next: bool = True):
    """Render Back / Next (or Predict) buttons and return (back_clicked, next_clicked)."""
    total = len(STEP_META)
    bcol, _, ncol = st.columns([1, 2, 1])
    back = False
    nxt  = False
    with bcol:
        if step > 1:
            back = st.button("← Back", key=f"back_{step}")
    with ncol:
        label = "🔍  Predict My Career Path" if step == total else "Next →"
        nxt = st.button(label, key=f"next_{step}", disabled=not can_next)
    return back, nxt


def validation_error(msg: str):
    st.markdown(
        f'<div style="background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.5);'
        f'border-radius:10px;padding:12px 16px;color:#fca5a5;font-size:0.88rem;margin-top:8px;">'
        f'⚠️ {msg}</div>',
        unsafe_allow_html=True,
    )


# ─── App layout ──────────────────────────────────────────────────────────────
def main():
    models = load_models()

    # ── Session-state init ────────────────────────────────────────────────────
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 1
    if "answers" not in st.session_state:
        st.session_state.answers = {}

    # Sidebar — model selector
    with st.sidebar:
        st.markdown('<p class="section-label">⚡ Configuration</p>', unsafe_allow_html=True)
        model_choice = st.radio(
            "Choose prediction model",
            MODEL_OPTIONS,
            index=4,  # Default: Ensemble (All Models)
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
            "Random Forest · XGBoost · LightGBM · CatBoost"
            "</p>",
            unsafe_allow_html=True,
        )

    # Hero
    st.markdown(
        '<div class="hero-title">Career Path Predictor 🎯</div>'
        '<p class="hero-sub">Answer a few questions and discover which tech role suits you best.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='custom'>", unsafe_allow_html=True)

    step = st.session_state.wizard_step

    # ── Completed — show results ──────────────────────────────────────────────
    if step == 4:
        inputs = st.session_state.answers

        results = {}
        for name, mdl in models.items():
            pred, conf, proba = predict_single(mdl, name, inputs)
            results[name] = {"pred": pred, "conf": conf, "proba": proba, "role": JOB_ROLES[pred]}

        all_probas = np.array([results[n]["proba"] for n in models])
        avg_proba  = all_probas.mean(axis=0)
        ens_pred   = int(np.argmax(avg_proba))
        ens_conf   = float(avg_proba[ens_pred])
        ens_role   = JOB_ROLES[ens_pred]

        # For ensemble: also find which single model has the highest confidence
        best_model_name = max(results, key=lambda n: results[n]["conf"])
        best_model_r    = results[best_model_name]

        if model_choice == "Ensemble (All Models)":
            final_role  = ens_role
            final_conf  = ens_conf
            model_label = "Ensemble (4 Models)"
            show_best_model_callout = True
        else:
            final_role  = results[model_choice]["role"]
            final_conf  = results[model_choice]["conf"]
            model_label = model_choice
            show_best_model_callout = False

        icon = ROLE_ICONS.get(final_role, "🎯")

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

            # Best single model callout — only shown for Ensemble mode
            if show_best_model_callout:
                bm_icon = ROLE_ICONS.get(best_model_r["role"], "🎯")
                st.markdown(
                    f"""
                    <div style="margin-top:14px;background:rgba(52,211,153,0.1);
                                border:1px solid rgba(52,211,153,0.35);border-radius:14px;
                                padding:16px 20px;">
                        <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;
                                    text-transform:uppercase;color:#34d399;margin-bottom:6px;">
                            ⭐ Best Single Model
                        </div>
                        <div style="display:flex;align-items:center;gap:10px;">
                            <span style="font-size:1.6rem;">{bm_icon}</span>
                            <div>
                                <div style="font-size:1rem;font-weight:700;color:#e2e8f0;">
                                    {best_model_r["role"]}
                                </div>
                                <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);">
                                    {best_model_name} &nbsp;·&nbsp;
                                    <span style="color:#6ee7b7;font-weight:700;">
                                        {best_model_r["conf"]*100:.1f}% confidence
                                    </span>
                                </div>
                            </div>
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
                ("CatBoost",      "CB"),
            ]
            col_a, col_b = st.columns(2)
            for i, (name, short) in enumerate(model_display_order):
                r = results[name]
                is_best = (name == best_model_name)
                border  = "rgba(52,211,153,0.6)" if is_best else "rgba(255,255,255,0.1)"
                target_col = col_a if i % 2 == 0 else col_b
                with target_col:
                    st.markdown(
                        f'<div class="metric-pill" style="border-color:{border};">'
                        f'{"<div style=\'font-size:0.6rem;color:#34d399;font-weight:700;margin-bottom:2px;\'>⭐ BEST</div>" if is_best else ""}'
                        f'<div class="metric-val">{r["conf"]*100:.0f}%</div>'
                        f'<div class="metric-lbl">{short}</div>'
                        f'<div style="font-size:0.72rem; color:#c4b5fd; margin-top:4px;">'
                        f'{ROLE_ICONS.get(r["role"], "")} {r["role"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

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

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">🎓 Recommended Courses for You</p>', unsafe_allow_html=True)
        courses = COURSE_RECOMMENDATIONS.get(final_role, [])
        c1, c2, c3 = st.columns(3)
        cols_c = [c1, c2, c3]
        for i, (title, desc) in enumerate(courses):
            with cols_c[i % 3]:
                st.markdown(
                    f'<div class="course-card"><strong>📘 {title}</strong>{desc}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄  Start Over"):
            st.session_state.wizard_step = 1
            st.session_state.answers = {}
            st.rerun()
        return

    # ── Progress bar ─────────────────────────────────────────────────────────
    render_progress(step)

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 1 — Skills & Ratings  +  Abilities
    # ═════════════════════════════════════════════════════════════════════════
    if step == 1:
        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            st.markdown('<p class="section-label">📊 Skills & Ratings</p>', unsafe_allow_html=True)
            logical_quotient = st.slider(
                "Logical Quotient Rating", 0, 9,
                st.session_state.answers.get("logical_quotient", 0)
            )
            hackathons = st.slider(
                "Hackathons Participated", 0, 6,
                st.session_state.answers.get("hackathons", 0)
            )
            coding_skills = st.slider(
                "Coding Skills Rating", 0, 9,
                st.session_state.answers.get("coding_skills", 0)
            )
            public_speaking = st.slider(
                "Public Speaking Points", 0, 9,
                st.session_state.answers.get("public_speaking", 0)
            )

        with col_r:
            st.markdown('<p class="section-label">🧠 Abilities</p>', unsafe_allow_html=True)
            rw_skills = st.select_slider(
                "Reading & Writing Skills",
                options=["poor", "medium", "excellent"],
                value=st.session_state.answers.get("rw_skills", "poor"),
            )
            memory = st.select_slider(
                "Memory Capability Score",
                options=["poor", "medium", "excellent"],
                value=st.session_state.answers.get("memory", "poor"),
            )

        st.markdown("<br>", unsafe_allow_html=True)
        back, nxt = nav_buttons(step=1, can_next=True)

        if nxt:
            st.session_state.answers.update(
                logical_quotient=logical_quotient,
                hackathons=hackathons,
                coding_skills=coding_skills,
                public_speaking=public_speaking,
                rw_skills=rw_skills,
                memory=memory,
            )
            st.session_state.wizard_step = 2
            st.rerun()

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2 — Learning Profile  +  Career Preferences
    # ═════════════════════════════════════════════════════════════════════════
    elif step == 2:
        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            st.markdown('<p class="section-label">📚 Learning Profile</p>', unsafe_allow_html=True)

            cert_opts = [
                'app development', 'distro making', 'full stack', 'hadoop',
                'information security', 'machine learning', 'python',
                'r programming', 'shell programming'
            ]
            certifications = st.selectbox(
                "Certification Area",
                options=["— select —"] + cert_opts,
                index=(["— select —"] + cert_opts).index(
                    st.session_state.answers.get("certifications", "— select —")
                ),
            )

            workshop_opts = [
                'cloud computing', 'data science', 'database security',
                'game development', 'hacking', 'system designing',
                'testing', 'web technologies'
            ]
            workshops = st.selectbox(
                "Workshop Attended",
                options=["— select —"] + workshop_opts,
                index=(["— select —"] + workshop_opts).index(
                    st.session_state.answers.get("workshops", "— select —")
                ),
            )

            subject_opts = [
                'Computer Architecture', 'IOT', 'Management', 'Software Engineering',
                'cloud computing', 'data engineering', 'hacking', 'networks',
                'parallel computing', 'programming'
            ]
            interested_subjects = st.selectbox(
                "Interested Subject",
                options=["— select —"] + subject_opts,
                index=(["— select —"] + subject_opts).index(
                    st.session_state.answers.get("interested_subjects", "— select —")
                ),
            )

        with col_r:
            st.markdown('<p class="section-label">🏢 Career Preferences</p>', unsafe_allow_html=True)

            career_opts = [
                'Business process analyst', 'cloud computing', 'developer',
                'security', 'system developer', 'testing'
            ]
            career_area = st.selectbox(
                "Interested Career Area",
                options=["— select —"] + career_opts,
                index=(["— select —"] + career_opts).index(
                    st.session_state.answers.get("career_area", "— select —")
                ),
            )

            company_opts = [
                'BPA', 'Cloud Services', 'Finance', 'Product based', 'SAaS services',
                'Sales and Marketing', 'Service Based',
                'Testing and Maintainance Services', 'Web Services', 'product development'
            ]
            company_type = st.selectbox(
                "Preferred Company Type",
                options=["— select —"] + company_opts,
                index=(["— select —"] + company_opts).index(
                    st.session_state.answers.get("company_type", "— select —")
                ),
            )

            mgmt_tech = st.radio(
                "Orientation",
                options=["Management", "Technical"],
                index=(["Management", "Technical"].index(st.session_state.answers["mgmt_tech"])
                       if st.session_state.answers.get("mgmt_tech") in ["Management", "Technical"]
                       else None),
                horizontal=True,
                key="w_mgmt_tech",
            )

            work_style = st.radio(
                "Work Style",
                options=["hard worker", "smart worker"],
                index=(["hard worker", "smart worker"].index(st.session_state.answers["work_style"])
                       if st.session_state.answers.get("work_style") in ["hard worker", "smart worker"]
                       else None),
                horizontal=True,
                key="w_work_style",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Validate
        errors_2 = []
        if certifications      == "— select —": errors_2.append("Certification Area")
        if workshops           == "— select —": errors_2.append("Workshop Attended")
        if interested_subjects == "— select —": errors_2.append("Interested Subject")
        if career_area         == "— select —": errors_2.append("Interested Career Area")
        if company_type        == "— select —": errors_2.append("Preferred Company Type")
        if mgmt_tech  is None:                  errors_2.append("Orientation")
        if work_style is None:                  errors_2.append("Work Style")

        back, nxt = nav_buttons(step=2, can_next=True)

        if back:
            st.session_state.wizard_step = 1
            st.rerun()

        if nxt:
            if errors_2:
                validation_error("Please answer all questions before continuing: " + ", ".join(errors_2))
            else:
                st.session_state.answers.update(
                    certifications=certifications,
                    workshops=workshops,
                    interested_subjects=interested_subjects,
                    career_area=career_area,
                    company_type=company_type,
                    mgmt_tech=mgmt_tech,
                    work_style=work_style,
                )
                st.session_state.wizard_step = 3
                st.rerun()

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 3 — Personal Traits  +  Reading Preference
    # ═════════════════════════════════════════════════════════════════════════
    elif step == 3:
        col_l, col_r = st.columns(2, gap="large")

        yn_opts = ["yes", "no"]

        def yn_index(key):
            val = st.session_state.answers.get(key)
            return yn_opts.index(val) if val in yn_opts else None

        with col_l:
            st.markdown('<p class="section-label">🧩 Personal Traits</p>', unsafe_allow_html=True)

            self_learning = st.radio(
                "Self-Learning Capability?",
                options=yn_opts,
                index=yn_index("self_learning"),
                horizontal=True,
                key="w_self_learning",
            )
            extra_courses = st.radio(
                "Completed Extra Courses?",
                options=yn_opts,
                index=yn_index("extra_courses"),
                horizontal=True,
                key="w_extra_courses",
            )
            senior_inputs = st.radio(
                "Taken Inputs from Seniors?",
                options=yn_opts,
                index=yn_index("senior_inputs"),
                horizontal=True,
                key="w_senior_inputs",
            )
            team_work = st.radio(
                "Worked in Teams?",
                options=yn_opts,
                index=yn_index("team_work"),
                horizontal=True,
                key="w_team_work",
            )
            introvert = st.radio(
                "Are You an Introvert?",
                options=yn_opts,
                index=yn_index("introvert"),
                horizontal=True,
                key="w_introvert",
            )

        with col_r:
            st.markdown('<p class="section-label">📖 Reading Preference</p>', unsafe_allow_html=True)

            book_opts = [
                'Action and Adventure', 'Anthology', 'Art', 'Autobiographies', 'Biographies',
                'Childrens', 'Comics', 'Cookbooks', 'Diaries', 'Dictionaries', 'Drama',
                'Encyclopedias', 'Fantasy', 'Guide', 'Health', 'History', 'Horror',
                'Journals', 'Math', 'Mystery', 'Poetry', 'Prayer books',
                'Religion-Spirituality', 'Romance', 'Satire', 'Science',
                'Science fiction', 'Self help', 'Series', 'Travel', 'Trilogy'
            ]
            books = st.selectbox(
                "Favourite Type of Books",
                options=["— select —"] + book_opts,
                index=(["— select —"] + book_opts).index(
                    st.session_state.answers.get("books", "— select —")
                ),
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Validate
        errors_3 = []
        if self_learning is None:            errors_3.append("Self-Learning Capability")
        if extra_courses is None:            errors_3.append("Completed Extra Courses")
        if senior_inputs is None:            errors_3.append("Taken Inputs from Seniors")
        if team_work     is None:            errors_3.append("Worked in Teams")
        if introvert     is None:            errors_3.append("Are You an Introvert")
        if books         == "— select —":    errors_3.append("Favourite Type of Books")

        back, nxt = nav_buttons(step=3, can_next=True)

        if back:
            st.session_state.wizard_step = 2
            st.rerun()

        if nxt:
            if errors_3:
                validation_error("Please answer all questions before predicting: " + ", ".join(errors_3))
            else:
                st.session_state.answers.update(
                    self_learning=self_learning,
                    extra_courses=extra_courses,
                    senior_inputs=senior_inputs,
                    team_work=team_work,
                    introvert=introvert,
                    books=books,
                )
                st.session_state.wizard_step = 4
                st.rerun()


if __name__ == "__main__":
    main()
