"""The look of the interface, in one place.

Streamlit ships a competent default that every Streamlit app shares, which is
exactly the problem: the tool looks like a script someone wrapped a form around.
The rules here are deliberately narrow, because most of what makes a generated
interface look generic is having no rules at all.

  One accent, well under full saturation. A second accent is how a page starts
  looking like a dashboard template.
  No pure black and no pure white. Both read as unfinished on a real display.
  One radius scale, one border colour, one shadow. Consistency is most of what
  separates designed from assembled.
  Contrast at or above 4.5:1 for anything that carries meaning.
"""

from __future__ import annotations

import streamlit as st

# Cool neutral base with a single warm accent. Warm on cool keeps the accent
# legible at small sizes without resorting to a saturated blue or the violet
# that every generated interface reaches for.
TOKENS = {
    "bg": "#0E1114",
    "surface": "#161A1F",
    "surface_high": "#1D222A",
    "border": "rgba(255,255,255,0.09)",
    "border_strong": "rgba(255,255,255,0.16)",
    "text": "#E7EAEF",
    "muted": "#98A1AE",
    "faint": "#6B7482",
    "accent": "#D8A24A",
    "accent_dim": "rgba(216,162,74,0.14)",
    "ok": "#63B98D",
    "warn": "#DFB44A",
    "bad": "#D8736A",
    "info": "#6FA8DC",
    "r_sm": "8px",
    "r_md": "12px",
    "r_lg": "16px",
}

CSS = """
<style>
  :root {{
    --bg:{bg}; --surface:{surface}; --surface-high:{surface_high};
    --border:{border}; --border-strong:{border_strong};
    --text:{text}; --muted:{muted}; --faint:{faint};
    --accent:{accent}; --accent-dim:{accent_dim};
    --ok:{ok}; --warn:{warn}; --bad:{bad}; --info:{info};
    --r-sm:{r_sm}; --r-md:{r_md}; --r-lg:{r_lg};
  }}

  .stApp {{ background:var(--bg); color:var(--text); }}
  section[data-testid="stSidebar"] {{
    background:var(--surface); border-right:1px solid var(--border);
  }}
  .block-container {{ padding-top:2.4rem; max-width:1180px; }}

  h1, h2, h3 {{ letter-spacing:-0.02em; color:var(--text); }}
  h1 {{ font-size:1.75rem; font-weight:680; }}
  h2 {{ font-size:1.2rem; font-weight:640; margin-top:0.4rem; }}
  h3 {{ font-size:1rem; font-weight:620; }}
  p, li, label, .stMarkdown {{ color:var(--text); }}

  /* A page opener: title plus one line saying what the page is for. */
  .oc-head {{ margin-bottom:1.4rem; }}
  .oc-head .t {{ font-size:1.75rem; font-weight:680; letter-spacing:-0.025em; }}
  .oc-head .s {{ color:var(--muted); font-size:0.9rem; margin-top:0.25rem;
                 max-width:65ch; line-height:1.55; }}

  .oc-card {{
    background:var(--surface); border:1px solid var(--border);
    border-radius:var(--r-md); padding:1rem 1.1rem; margin-bottom:0.7rem;
  }}
  .oc-card.accent {{ border-left:2px solid var(--accent); }}

  .oc-row {{ display:flex; align-items:center; gap:0.75rem; }}
  .oc-grow {{ flex:1; min-width:0; }}
  .oc-title {{ font-weight:620; font-size:0.98rem; letter-spacing:-0.01em; }}
  .oc-sub {{ color:var(--muted); font-size:0.82rem; margin-top:0.15rem;
             white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}

  .oc-pill {{
    display:inline-flex; align-items:center; gap:0.35rem;
    font-size:0.72rem; font-weight:640; letter-spacing:0.02em;
    padding:0.2rem 0.55rem; border-radius:999px;
    border:1px solid var(--border-strong); color:var(--muted);
    background:rgba(255,255,255,0.03); white-space:nowrap;
  }}
  .oc-pill.run  {{ color:#0E1114; background:var(--accent); border-color:var(--accent); }}
  .oc-pill.ok   {{ color:var(--ok);   border-color:rgba(99,185,141,0.4); }}
  .oc-pill.bad  {{ color:var(--bad);  border-color:rgba(216,115,106,0.4); }}
  .oc-pill.warn {{ color:var(--warn); border-color:rgba(223,180,74,0.4); }}
  .oc-pill.gen  {{ color:var(--accent); border-color:rgba(216,162,74,0.4); }}
  .oc-pill.stock{{ color:var(--info); border-color:rgba(111,168,220,0.4); }}

  /* Weighted stage progress. Streamlit's own bar cannot show which stage. */
  .oc-track {{
    display:flex; gap:2px; height:6px; border-radius:999px;
    overflow:hidden; background:rgba(255,255,255,0.06); margin:0.55rem 0 0.4rem;
  }}
  .oc-seg {{ background:rgba(255,255,255,0.10); }}
  .oc-seg.done {{ background:var(--accent); }}
  .oc-seg.now  {{ background:var(--accent); opacity:0.55; }}
  .oc-stages {{ display:flex; justify-content:space-between;
                font-size:0.66rem; color:var(--faint); letter-spacing:0.04em;
                text-transform:uppercase; }}

  .oc-metrics {{ display:flex; gap:1.6rem; flex-wrap:wrap; margin:0.2rem 0 0.1rem; }}
  .oc-metric .v {{ font-size:1.3rem; font-weight:680; letter-spacing:-0.02em; }}
  .oc-metric .k {{ font-size:0.68rem; color:var(--faint); letter-spacing:0.07em;
                   text-transform:uppercase; margin-top:0.1rem; }}

  /* Live output. Monospace because it is a log, and logs are read by shape. */
  .oc-term {{
    background:#0A0D10; border:1px solid var(--border); border-radius:var(--r-sm);
    padding:0.8rem 0.9rem; font-family:"Cascadia Mono",Consolas,monospace;
    font-size:0.76rem; line-height:1.65; color:#B9C2CE;
    max-height:340px; overflow-y:auto; white-space:pre-wrap; word-break:break-word;
  }}
  .oc-term .hl {{ color:var(--accent); }}

  .oc-note {{
    border:1px solid var(--border); border-left:2px solid var(--info);
    background:rgba(111,168,220,0.07); border-radius:var(--r-sm);
    padding:0.7rem 0.9rem; font-size:0.84rem; color:var(--text);
  }}
  .oc-note.warn {{ border-left-color:var(--warn); background:rgba(223,180,74,0.07); }}
  .oc-note.bad  {{ border-left-color:var(--bad);  background:rgba(216,115,106,0.07); }}

  .oc-empty {{
    border:1px dashed var(--border-strong); border-radius:var(--r-md);
    padding:2.2rem 1.2rem; text-align:center; color:var(--muted);
    font-size:0.88rem;
  }}

  /* Contact Sheet Cinema Styling */
  .oc-contact-sheet {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
    gap: 1rem;
    padding: 0.8rem;
    background: #080A0D;
    border: 1px solid var(--border-strong);
    border-radius: var(--r-md);
    margin: 0.8rem 0 1.2rem;
  }}
  .oc-contact-card {{
    background: #12161D;
    border: 1px solid var(--border);
    border-radius: var(--r-sm);
    overflow: hidden;
    position: relative;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    transition: transform 0.15s ease, border-color 0.15s ease;
  }}
  .oc-contact-card:hover {{
    transform: translateY(-2px);
    border-color: var(--accent);
  }}
  .oc-contact-img {{
    width: 100%;
    aspect-ratio: 9 / 16;
    object-fit: cover;
    display: block;
    background: #0D1015;
  }}
  .oc-contact-meta {{
    padding: 0.5rem 0.65rem;
    background: #151A22;
    border-top: 1px solid var(--border);
  }}
  .oc-contact-num {{
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }}
  .oc-contact-prompt {{
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.2rem;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    line-height: 1.35;
  }}

  /* Live Pipeline Stepper */
  .oc-stepper {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #0A0D12;
    border: 1px solid var(--border);
    border-radius: var(--r-md);
    padding: 0.8rem 1.1rem;
    margin: 0.8rem 0;
    overflow-x: auto;
  }}
  .oc-step-node {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.25rem;
    position: relative;
    z-index: 2;
  }}
  .oc-step-circle {{
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    border: 2px solid var(--border-strong);
    background: #141820;
    color: var(--muted);
  }}
  .oc-step-node.done .oc-step-circle {{
    background: var(--ok);
    border-color: var(--ok);
    color: #0E1114;
    box-shadow: 0 0 10px rgba(99,185,141,0.4);
  }}
  .oc-step-node.active .oc-step-circle {{
    background: var(--accent);
    border-color: var(--accent);
    color: #0E1114;
    box-shadow: 0 0 12px rgba(216,162,74,0.6);
    animation: oc-pulse 2s infinite;
  }}
  @keyframes oc-pulse {{
    0% {{ transform: scale(1); }}
    50% {{ transform: scale(1.08); }}
    100% {{ transform: scale(1); }}
  }}
  .oc-step-label {{
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  .oc-step-node.active .oc-step-label {{
    color: var(--accent);
  }}
  .oc-step-node.done .oc-step-label {{
    color: var(--text);
  }}

  /* Telemetry & Key Ring Card */
  .oc-telemetry {{
    background: linear-gradient(145deg, #13171F, #0E1218);
    border: 1px solid var(--border-strong);
    border-radius: var(--r-md);
    padding: 0.8rem 1rem;
    margin-bottom: 0.8rem;
  }}

  /* QA Scorecard */
  .oc-qa-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.1rem;
    font-weight: 700;
    padding: 0.4rem 0.9rem;
    border-radius: var(--r-sm);
    background: rgba(99,185,141,0.12);
    border: 1px solid rgba(99,185,141,0.3);
    color: var(--ok);
  }}
  .oc-qa-badge.warn {{
    background: rgba(223,180,74,0.12);
    border-color: rgba(223,180,74,0.3);
    color: var(--warn);
  }}

  /* Mode selection cards */
  .oc-mode-banner {{
    display: flex;
    gap: 1rem;
    margin: 0.5rem 0 1rem;
  }}
  .oc-mode-option {{
    flex: 1;
    background: #141820;
    border: 2px solid var(--border);
    border-radius: var(--r-md);
    padding: 0.8rem 1rem;
    cursor: pointer;
  }}
  .oc-mode-option.selected {{
    border-color: var(--accent);
    background: rgba(216,162,74,0.06);
  }}

  .stButton > button {{
    border-radius:var(--r-sm); border:1px solid var(--border-strong);
    background:var(--surface-high); color:var(--text);
    font-weight:560; font-size:0.85rem; transition:border-color .15s, background .15s;
  }}
  .stButton > button:hover {{ border-color:var(--accent); background:#232932; }}
  .stButton > button[kind="primary"] {{
    background:var(--accent); color:#0E1114; border-color:var(--accent); font-weight:660;
  }}
  .stButton > button[kind="primary"]:hover {{ background:#E3B061; border-color:#E3B061; }}

  div[data-testid="stTextInput"] input,
  div[data-testid="stTextArea"] textarea,
  div[data-testid="stNumberInput"] input {{
    background:var(--surface-high); border-radius:var(--r-sm);
    border:1px solid var(--border); color:var(--text);
  }}
  div[data-baseweb="select"] > div {{
    background:var(--surface-high); border-radius:var(--r-sm); border-color:var(--border);
  }}
  div[data-testid="stExpander"] {{
    border:1px solid var(--border); border-radius:var(--r-md); background:var(--surface);
  }}
  .stTabs [data-baseweb="tab-list"] {{ gap:0.3rem; border-bottom:1px solid var(--border); }}
  .stTabs [data-baseweb="tab"] {{
    background:transparent; color:var(--muted); border-radius:var(--r-sm) var(--r-sm) 0 0;
    padding:0.5rem 0.9rem; font-size:0.86rem; font-weight:560;
  }}
  .stTabs [aria-selected="true"] {{ color:var(--text); background:var(--surface); }}
  div[data-testid="stDataFrame"] {{ border-radius:var(--r-sm); overflow:hidden; }}
  hr {{ border-color:var(--border); }}
  #MainMenu, footer {{ visibility:hidden; }}

  /* Sidebar futuristic styling */
  .oc-sidebar-brand {{
    background: linear-gradient(135deg, rgba(216,162,74,0.14), rgba(99,185,141,0.08));
    border: 1px solid rgba(216,162,74,0.35);
    border-radius: var(--r-md);
    padding: 0.9rem 1rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    position: relative;
    overflow: hidden;
  }}
  .oc-brand-badge {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }}
  .oc-brand-icon {{
    font-size: 1.5rem;
    background: #141820;
    border: 1px solid var(--border-strong);
    border-radius: 10px;
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 15px rgba(216,162,74,0.25);
  }}
  .oc-brand-title {{
    font-weight: 800;
    font-size: 1.05rem;
    letter-spacing: 0.05em;
    color: var(--text);
  }}
  .oc-brand-ver {{
    font-size: 0.65rem;
    background: var(--accent);
    color: #0E1114;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    font-weight: 700;
    vertical-align: middle;
  }}
  .oc-brand-subtitle {{
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.15rem;
  }}
  .oc-brand-status {{
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin-top: 0.7rem;
    padding-top: 0.55rem;
    border-top: 1px solid rgba(255,255,255,0.06);
  }}
  .oc-led-pulse {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #00FF88;
    box-shadow: 0 0 8px #00FF88, 0 0 16px #00FF88;
    animation: oc-led-glow 1.8s infinite ease-in-out;
  }}
  @keyframes oc-led-glow {{
    0% {{ opacity: 0.4; transform: scale(0.9); }}
    50% {{ opacity: 1; transform: scale(1.2); box-shadow: 0 0 12px #00FF88, 0 0 20px #00FF88; }}
    100% {{ opacity: 0.4; transform: scale(0.9); }}
  }}
  .oc-status-text {{
    font-size: 0.68rem;
    font-weight: 700;
    color: #00FF88;
    letter-spacing: 0.06em;
  }}

  /* Sidebar Navigation Radio Styling */
  section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {{
    gap: 0.35rem;
  }}
  section[data-testid="stSidebar"] div[data-testid="stRadio"] label {{
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 9px;
    padding: 0.55rem 0.85rem;
    margin: 0.1rem 0;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    cursor: pointer;
    display: flex;
    align-items: center;
    width: 100%;
  }}
  section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {{
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(216, 162, 74, 0.4);
    transform: translateX(3px);
  }}
  section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {{
    background: linear-gradient(90deg, rgba(216, 162, 74, 0.2), rgba(216, 162, 74, 0.04));
    border-color: rgba(216, 162, 74, 0.7);
    box-shadow: 0 0 14px rgba(216, 162, 74, 0.15);
  }}
  section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p {{
    color: #FFFFFF !important;
    font-weight: 600 !important;
  }}

  /* Clean Service Lights Card */
  .oc-service-card {{
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border);
    border-radius: var(--r-md);
    padding: 0.85rem 0.95rem;
    margin-top: 1.1rem;
  }}
  .oc-service-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.65rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
  }}
  .oc-service-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    font-size: 0.8rem;
  }}
  .oc-service-row:last-child {{
    border-bottom: none;
    padding-bottom: 0;
  }}
  .oc-service-left {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    color: var(--text);
    font-weight: 500;
    font-size: 0.8rem;
  }}
  .oc-service-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #22C55E;
    box-shadow: 0 0 6px #22C55E, 0 0 12px rgba(34, 197, 94, 0.4);
    flex-shrink: 0;
    display: inline-block;
  }}
  .oc-service-status {{
    color: #63B98D;
    font-size: 0.72rem;
    font-weight: 600;
  }}
</style>
"""


def apply() -> None:
    """Install the stylesheet into modern Streamlit st.html and legacy st.markdown."""
    css_content = CSS.format(**TOKENS)
    try:
        st.html(css_content)
    except Exception:
        pass
    try:
        st.markdown(css_content, unsafe_allow_html=True)
    except Exception:
        pass


def head(title: str, subtitle: str = "") -> None:
    """A page opener: what this page is, and what it is for."""
    import html

    block = f'<div class="oc-head"><div class="t">{html.escape(title)}</div>'
    if subtitle:
        block += f'<div class="s">{html.escape(subtitle)}</div>'
    st.markdown(block + "</div>", unsafe_allow_html=True)


def pill(text: str, kind: str = "") -> str:
    import html

    return f'<span class="oc-pill {kind}">{html.escape(str(text))}</span>'


def note(text: str, kind: str = "") -> None:
    import html

    st.markdown(f'<div class="oc-note {kind}">{html.escape(text)}</div>',
                unsafe_allow_html=True)


def empty(text: str) -> None:
    import html

    st.markdown(f'<div class="oc-empty">{html.escape(text)}</div>',
                unsafe_allow_html=True)
