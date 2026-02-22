import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SteamLens · AI Game Discovery",
    layout="wide",
    page_icon="🎮",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg-base:        #0a0e1a;
    --bg-surface:     #111827;
    --bg-card:        rgba(255,255,255,0.04);
    --bg-card-hover:  rgba(255,255,255,0.07);
    --border:         rgba(255,255,255,0.08);
    --border-accent:  rgba(102,192,244,0.35);
    --blue:           #66c0f4;
    --blue-bright:    #89d4ff;
    --blue-glow:      rgba(102,192,244,0.15);
    --green:          #4ade80;
    --amber:          #fbbf24;
    --red:            #f87171;
    --text-primary:   #f0f4f8;
    --text-secondary: #8b99a6;
    --text-muted:     #4a5568;
    --font-display:   'Rajdhani', sans-serif;
    --font-body:      'Inter', sans-serif;
    --radius-sm:      8px;
    --radius-md:      14px;
    --radius-lg:      20px;
    --transition:     all 0.22s cubic-bezier(0.4,0,0.2,1);
}

.stApp {
    background: var(--bg-base) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 10% -10%, rgba(102,192,244,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 110%, rgba(74,222,128,0.04) 0%, transparent 60%) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1400px !important; }

h1, h2, h3 { font-family: var(--font-display) !important; }
p, label, span { font-family: var(--font-body) !important; }

/* ── HERO ── */
.hero-wrap {
    display: flex; align-items: center; gap: 18px;
    margin-bottom: 2rem; padding-bottom: 1.8rem;
    border-bottom: 1px solid var(--border);
}
.hero-icon { font-size: 2.8rem; filter: drop-shadow(0 0 16px rgba(102,192,244,0.6)); }
.hero-title {
    font-family: var(--font-display) !important;
    font-size: 2.6rem !important; font-weight: 700 !important; letter-spacing: 1px;
    background: linear-gradient(135deg, var(--blue-bright) 0%, var(--blue) 50%, #a78bfa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin: 0 !important; line-height: 1 !important;
}
.hero-sub { color: var(--text-secondary); font-size: 0.85rem; letter-spacing: 2px; text-transform: uppercase; margin-top: 4px; }
.hero-badge {
    margin-left: auto; background: var(--blue-glow); border: 1px solid var(--border-accent);
    border-radius: 100px; padding: 6px 16px; font-size: 0.75rem; color: var(--blue);
    font-family: var(--font-body); letter-spacing: 1px; font-weight: 500; white-space: nowrap;
}

/* ── SECTION LABEL ── */
.section-label {
    font-family: var(--font-display); font-size: 0.72rem; font-weight: 600;
    letter-spacing: 3px; text-transform: uppercase; color: var(--blue);
    margin-bottom: 0.75rem; display: flex; align-items: center; gap: 8px;
}
.section-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border-accent), transparent);
}

/* ── STAT CHIPS ── */
.stat-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 0.8rem; margin-bottom: 1.8rem; }
.stat-chip {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-sm);
    padding: 8px 14px; font-size: 0.75rem; color: var(--text-secondary);
    display: inline-flex; align-items: center; gap: 6px;
}
.stat-chip strong { color: var(--text-primary); font-size: 0.9rem; }

/* ── FILTER PANEL ── */
.filter-panel {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 1.4rem 1.6rem;
}

/* ── RESULT CARD ── */
.result-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-md); overflow: hidden;
    transition: var(--transition); margin-bottom: 1rem; position: relative;
}
.result-card:hover {
    background: var(--bg-card-hover); border-color: var(--border-accent);
    transform: translateY(-2px); box-shadow: 0 8px 32px rgba(102,192,244,0.1);
}
.result-card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, var(--blue), #a78bfa);
    opacity: 0; transition: var(--transition); border-radius: 3px 0 0 3px;
}
.result-card:hover::before { opacity: 1; }

.card-title {
    font-family: var(--font-display); font-size: 1.1rem; font-weight: 700;
    color: var(--text-primary); margin: 0 0 6px 0; letter-spacing: 0.3px;
}
.card-meta { font-size: 0.78rem; color: var(--text-secondary); display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }

/* ── BADGES ── */
.badge { display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: 100px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.5px; }
.badge-positive { background: rgba(74,222,128,0.1);  color: var(--green); border: 1px solid rgba(74,222,128,0.2); }
.badge-mixed    { background: rgba(251,191,36,0.1);   color: var(--amber); border: 1px solid rgba(251,191,36,0.2); }
.badge-negative { background: rgba(248,113,113,0.1);  color: var(--red);   border: 1px solid rgba(248,113,113,0.2); }
.badge-price    { background: rgba(102,192,244,0.1);  color: var(--blue);  border: 1px solid var(--border-accent); }
.badge-free     { background: rgba(74,222,128,0.1);   color: var(--green); border: 1px solid rgba(74,222,128,0.2); }

/* ── REC CARD ── */
.rec-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-md); overflow: hidden; transition: var(--transition);
}
.rec-card:hover {
    border-color: var(--border-accent); transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(102,192,244,0.12);
}
.rec-card-inner { padding: 12px; }
.rec-title {
    font-family: var(--font-display); font-size: 0.95rem; font-weight: 600;
    color: var(--text-primary); margin: 8px 0 6px 0; line-height: 1.3;
}
.rec-bar-wrap { background: rgba(255,255,255,0.06); border-radius: 100px; height: 4px; margin: 8px 0 4px 0; overflow: hidden; }
.rec-bar      { height: 100%; border-radius: 100px; background: linear-gradient(90deg, var(--blue), var(--blue-bright)); }
.rec-bar-label { font-size: 0.7rem; color: var(--text-secondary); display: flex; justify-content: space-between; margin-bottom: 8px; }

/* ── LIBRARY ── */
.library-item {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-sm);
    padding: 10px 14px; margin-bottom: 8px; display: flex; align-items: center; gap: 10px;
    font-size: 0.83rem; color: var(--text-primary); transition: var(--transition);
}
.library-item:hover { border-color: var(--border-accent); background: var(--bg-card-hover); }
.library-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--blue); box-shadow: 0 0 8px var(--blue); flex-shrink: 0; }

/* ── EMPTY STATE ── */
.empty-state { text-align: center; padding: 3rem 2rem; color: var(--text-muted); }
.empty-icon  { font-size: 2.5rem; margin-bottom: 12px; opacity: 0.35; }
.empty-text  { font-size: 0.85rem; line-height: 1.7; }

/* ── SPINNER OVERRIDE ── */
.stSpinner > div { border-top-color: var(--blue) !important; }

/* ── DIVIDER ── */
.custom-divider { height: 1px; background: linear-gradient(90deg, transparent, var(--border), transparent); margin: 2rem 0; }

/* ── STREAMLIT WIDGETS ── */
.stTextInput > div > div > input {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; color: var(--text-primary) !important;
    font-family: var(--font-body) !important; font-size: 0.9rem !important;
    padding: 0.6rem 1rem !important; transition: var(--transition) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--blue) !important; box-shadow: 0 0 0 3px var(--blue-glow) !important;
}
.stTextInput > div > div > input::placeholder { color: var(--text-muted) !important; }

.stSelectbox > div > div {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; color: var(--text-primary) !important;
}
.stSlider > div > div > div > div { background: var(--blue) !important; }
.stSlider > div > div > div       { background: rgba(255,255,255,0.1) !important; }

.stButton > button {
    background: linear-gradient(135deg, rgba(102,192,244,0.15), rgba(167,139,250,0.1)) !important;
    border: 1px solid var(--border-accent) !important; border-radius: var(--radius-sm) !important;
    color: var(--blue) !important; font-family: var(--font-display) !important;
    font-size: 0.85rem !important; font-weight: 600 !important; letter-spacing: 1px !important;
    padding: 0.45rem 1.2rem !important; transition: var(--transition) !important; width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(102,192,244,0.25), rgba(167,139,250,0.2)) !important;
    border-color: var(--blue) !important; box-shadow: 0 0 20px var(--blue-glow) !important;
    color: var(--blue-bright) !important; transform: translateY(-1px) !important;
}
.clear-btn > button {
    background: rgba(248,113,113,0.08) !important; border-color: rgba(248,113,113,0.3) !important; color: var(--red) !important;
}
.clear-btn > button:hover {
    background: rgba(248,113,113,0.15) !important; border-color: var(--red) !important;
    box-shadow: 0 0 20px rgba(248,113,113,0.15) !important; color: var(--red) !important;
}

label, .stSelectbox label, .stSlider label, .stTextInput label {
    color: var(--text-secondary) !important; font-size: 0.78rem !important;
    font-family: var(--font-body) !important; letter-spacing: 0.5px !important; text-transform: uppercase !important;
}

.stImage img { border-radius: var(--radius-sm) !important; width: 100% !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #111827; }
::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--blue); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS  (cached — loads once per session)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    games        = joblib.load('games_data.pkl')
    tfidf_matrix = joblib.load('tfidf_matrix.pkl')
    return games, tfidf_matrix

with st.spinner("🎮  Loading AI engine…"):
    games, tfidf_matrix = load_models()

if 'library' not in st.session_state:
    st.session_state.library = []


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def rating_badge(rating: str) -> str:
    r = str(rating).lower()
    if 'overwhelmingly positive' in r or 'very positive' in r or 'mostly positive' in r:
        cls = 'badge-positive'
    elif 'mixed' in r or 'unknown' in r:
        cls = 'badge-mixed'
    else:
        cls = 'badge-negative'
    short = (str(rating)
             .replace('Overwhelmingly ', 'Ovwhlm. ')
             .replace(' Positive', ' +').replace(' Negative', ' −'))
    return f'<span class="badge {cls}">{short}</span>'

def price_badge(price: float) -> str:
    if price == 0:
        return '<span class="badge badge-free">Free</span>'
    return f'<span class="badge badge-price">${price:.2f}</span>'

def steam_img(app_id) -> str:
    return f"https://cdn.akamai.steamstatic.com/steam/apps/{int(app_id)}/header.jpg"

@st.cache_data(show_spinner=False, max_entries=200)
def get_recommendations(_tfidf_matrix, library_titles: tuple,
                         platform: str, budget: float, min_ratio: float,
                         n: int = 6):
    """
    On-the-fly cosine similarity for the library centroid.
    Cached by (library, filters) — instant on repeated queries.
    """
    # Build centroid of all library game vectors
    idxs = []
    for title in library_titles:
        match = games[games['title'] == title]
        if not match.empty:
            idxs.append(match.index[0])

    if not idxs:
        return []

    # Centroid vector across library
    lib_matrix = _tfidf_matrix[idxs]           # shape (K, F)
    centroid   = np.asarray(lib_matrix.mean(axis=0))  # shape (1, F) — plain array, not np.matrix

    # Cosine similarity: one query against all 71K games
    scores = cosine_similarity(centroid, _tfidf_matrix).flatten()  # shape (N,)

    # Zero out library games themselves
    for i in idxs:
        scores[i] = -1

    ranked  = scores.argsort()[::-1]
    results = []

    for idx in ranked:
        g = games.iloc[idx]
        plat_ok  = bool(g.get(platform, True))
        price_ok = float(g['price_final']) <= budget
        ratio_ok = float(g.get('positive_ratio', 0)) >= min_ratio
        not_lib  = g['title'] not in library_titles

        if plat_ok and price_ok and ratio_ok and not_lib:
            results.append({
                'idx':      idx,
                'score':    float(scores[idx]),
                'title':    g['title'],
                'rating':   g['rating'],
                'ratio':    float(g.get('positive_ratio', 0)),
                'price':    float(g['price_final']),
                'app_id':   g.get('app_id', 0),
            })

        if len(results) == n:
            break

    return results


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
free_pct  = int((games['price_final'] == 0).mean() * 100)
avg_score = int(games['positive_ratio'].mean())

st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-icon">🎮</div>
    <div>
        <div class="hero-title">SteamLens</div>
        <div class="hero-sub">AI-Powered Game Discovery Engine</div>
    </div>
    <div class="hero-badge">✦ {len(games):,} Games Indexed</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="stat-row">
    <div class="stat-chip">🗄️ Dataset <strong>{len(games):,} games</strong></div>
    <div class="stat-chip">🆓 Free titles <strong>{free_pct}%</strong></div>
    <div class="stat-chip">⭐ Avg approval <strong>{avg_score}%</strong></div>
    <div class="stat-chip">🤖 Algorithm <strong>TF-IDF · On-the-Fly Cosine Similarity</strong></div>
    <div class="stat-chip">⚡ Query time <strong>&lt; 1 sec</strong></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SEARCH + FILTERS
# ─────────────────────────────────────────────
col_search, col_gap, col_filters = st.columns([3, 0.15, 1.3])

with col_filters:
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">⚙ Filters</div>', unsafe_allow_html=True)

    platform_map   = {"Windows": "win", "macOS": "mac", "Linux": "linux"}
    platform_label = st.selectbox("Platform", list(platform_map.keys()))
    platform       = platform_map[platform_label]

    budget    = st.slider("Max Price (USD)", 0, 100, 60, step=5)
    min_ratio = st.slider("Min Approval %",  0, 100, 50, step=5)

    st.markdown('</div>', unsafe_allow_html=True)

with col_search:
    st.markdown('<div class="section-label">🔍 Search Library</div>', unsafe_allow_html=True)
    query = st.text_input(
        label="search_query",
        placeholder="Search any of the 71,000+ games — e.g. Elden Ring, Portal, Dota…",
        label_visibility="collapsed"
    )

    if query:
        results = games[games['title'].str.contains(query, case=False, na=False)].head(5)

        if results.empty:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🔭</div>
                <div class="empty-text">No games found.<br>Try a different search term.</div>
            </div>""", unsafe_allow_html=True)
        else:
            for _, row in results.iterrows():
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                c_img, c_info, c_btn = st.columns([1.4, 3, 1])

                with c_img:
                    try:
                        st.image(steam_img(row['app_id']), use_container_width=True)
                    except:
                        st.markdown('<div style="font-size:2rem;text-align:center;padding:1rem">🎮</div>', unsafe_allow_html=True)

                with c_info:
                    st.markdown(f"""
                    <div class="card-title">{row['title']}</div>
                    <div class="card-meta">
                        {rating_badge(row['rating'])}
                        {price_badge(row['price_final'])}
                        <span style="color:var(--text-muted)">👍 {int(row.get('positive_ratio',0))}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                with c_btn:
                    already = row['title'] in st.session_state.library
                    if already:
                        st.markdown('<div style="color:var(--green);font-size:0.8rem;text-align:center;padding-top:14px;font-weight:600">✓ Added</div>', unsafe_allow_html=True)
                    else:
                        if st.button("＋ Add", key=f"add_{row['app_id']}"):
                            st.session_state.library.append(row['title'])
                            st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DIVIDER
# ─────────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LIBRARY + RECOMMENDATIONS
# ─────────────────────────────────────────────
col_lib, col_gap2, col_rec = st.columns([1, 0.1, 3])

# ── LIBRARY ──────────────────────────────────
with col_lib:
    st.markdown('<div class="section-label">📚 Your Library</div>', unsafe_allow_html=True)

    if st.session_state.library:
        for title in st.session_state.library:
            st.markdown(f"""
            <div class="library-item">
                <div class="library-dot"></div>
                <span>{title}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("🗑  Clear Library"):
            st.session_state.library = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📭</div>
            <div class="empty-text">Your library is empty.<br>Search and add games<br>to get recommendations.</div>
        </div>""", unsafe_allow_html=True)


# ── RECOMMENDATIONS ──────────────────────────
with col_rec:
    st.markdown('<div class="section-label">✦ Recommended For You</div>', unsafe_allow_html=True)

    if st.session_state.library:
        with st.spinner("⚡ Computing recommendations across 71,000 games…"):
            recs = get_recommendations(
                tfidf_matrix,
                tuple(st.session_state.library),   # hashable for cache
                platform, budget, min_ratio, n=6
            )

        if not recs:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🔧</div>
                <div class="empty-text">No matches with current filters.<br>Try relaxing Price or Approval % sliders.</div>
            </div>""", unsafe_allow_html=True)
        else:
            grid = st.columns(3)
            for i, rec in enumerate(recs):
                with grid[i % 3]:
                    bar_w = max(4, int(rec['score'] * 100))

                    st.markdown('<div class="rec-card">', unsafe_allow_html=True)
                    try:
                        st.image(steam_img(rec['app_id']), use_container_width=True)
                    except:
                        pass

                    st.markdown(f"""
                    <div class="rec-card-inner">
                        <div class="rec-title">{rec['title']}</div>
                        <div class="rec-bar-wrap">
                            <div class="rec-bar" style="width:{bar_w}%"></div>
                        </div>
                        <div class="rec-bar-label">
                            <span>Match</span>
                            <span style="color:var(--blue);font-weight:600">{rec['score']:.1%}</span>
                        </div>
                        <div class="card-meta">
                            {rating_badge(rec['rating'])}
                            {price_badge(rec['price'])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state" style="padding:5rem 2rem">
            <div class="empty-icon" style="font-size:3.5rem">🎯</div>
            <div class="empty-text" style="font-size:0.9rem">
                Add at least one game to your library<br>to activate the AI recommendation engine.
            </div>
        </div>""", unsafe_allow_html=True)