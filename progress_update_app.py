import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Progress Update Repurchase Analyzer", page_icon="📬", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── Light mode tokens ── */
:root {
    --gold:           #4a7c59;
    --gold-light:     #6aaa82;
    --bg-card:        #f4f7f5;
    --bg-insight:     #f9fdfb;
    --bg-table-head:  #edf2ef;
    --bg-table-row:   #ffffff;
    --border-card:    #c8ddd0;
    --border-row:     #e4ede8;
    --text-primary:   #1a1f1c;
    --text-secondary: #4a5750;
    --text-muted:     #7a8c83;
    --val-pos:        #1a5c35;
    --val-neg:        #555555;
    --tag-green-bg:   #d4ede0;
    --tag-green-fg:   #145c35;
    --tag-red-bg:     #f8d7da;
    --tag-red-fg:     #721c24;
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg-card:        #1e2a23;
        --bg-insight:     #192219;
        --bg-table-head:  #243029;
        --bg-table-row:   #151d17;
        --border-card:    #3a5042;
        --border-row:     #2a3c30;
        --text-primary:   #e8f0eb;
        --text-secondary: #9ab5a2;
        --text-muted:     #6a8572;
        --val-pos:        #6fcf97;
        --val-neg:        #b0aa9f;
        --tag-green-bg:   #1a4a2e;
        --tag-green-fg:   #6fcf97;
        --tag-red-bg:     #4a1a1a;
        --tag-red-fg:     #f1948a;
    }
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif; }

.metric-card {
    background: var(--bg-card);
    border-left: 4px solid var(--gold);
    border-radius: 6px; padding: 1rem 1.25rem; margin-bottom: 0.5rem;
}
.metric-card .label {
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.09em; color: var(--text-muted); margin-bottom: 0.2rem;
}
.metric-card .value { font-size: 1.75rem; font-weight: 600; color: var(--text-primary); }
.metric-card .sub   { font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.1rem; }

.section-header {
    font-family: 'Playfair Display', serif; font-size: 1.25rem;
    color: var(--text-primary);
    border-bottom: 2px solid var(--gold);
    padding-bottom: 0.4rem; margin: 1.5rem 0 1rem 0;
}

.insight-box {
    background: var(--bg-insight); border: 1px solid var(--border-card);
    border-radius: 8px; padding: 1.2rem 1.5rem;
    font-size: 0.9rem; line-height: 1.7; color: var(--text-primary);
}

.tab-label {
    font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.07em; color: var(--text-muted);
    margin-bottom: 0.5rem;
}

.tag { display: inline-block; padding: 0.2rem 0.65rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600; margin: 0.2rem; }
.tag-green { background: var(--tag-green-bg); color: var(--tag-green-fg); }
.tag-red   { background: var(--tag-red-bg);   color: var(--tag-red-fg); }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
EXCEL_PATH = "ProgressUpdates&Repurchase.xlsx"

# ── Text feature extraction ───────────────────────────────────────────────────
STOP_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "he","she","they","we","it","is","was","are","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should",
    "may","might","shall","can","this","that","these","those","i","you",
    "your","our","their","his","her","its","my","me","us","him","just",
    "student","students","session","sessions","week","time","also","very",
    "well","s","t","re","ll","ve","d","m","great","good","really",
    "like","during","today","class","tutor","worked","work","please",
    "know","let","any","all","been","more","some","about","what","from",
    "best","thank","thanks","dear","hello","hi","hope","regards",
}

def extract_features(text: str) -> dict:
    empty = {k: 0 for k in [
        "char_count","word_count","sentence_count","avg_sentence_len",
        "number_count","has_score_mention","has_next_steps","has_specific_skills",
        "has_positive_framing","has_improvement_language","has_goals",
        "has_homework_mention","has_parent_action","has_urgency",
        "has_recommendation","has_hour_mention","has_scheduling",
        "exclamation_count","question_count","unique_word_ratio","paragraph_count",
    ]}
    if not text or not str(text).strip():
        return empty
    text = str(text)
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return {
        "char_count":             len(text),
        "word_count":             len(words),
        "sentence_count":         len(sentences),
        "avg_sentence_len":       np.mean([len(s.split()) for s in sentences]) if sentences else 0,
        "number_count":           len(re.findall(r"\d+", text)),
        "paragraph_count":        len(paragraphs),
        "has_score_mention":      int(bool(re.search(r"\b(\d{3,4}|\d+\s*points?|score|scaled|composite|section)\b", text, re.I))),
        "has_next_steps":         int(bool(re.search(r"\b(next|plan|recommend|suggest|going forward|moving forward|upcoming|continue|our goal)\b", text, re.I))),
        "has_specific_skills":    int(bool(re.search(r"\b(algebra|geometry|calculus|essay|grammar|reading|writing|comprehension|vocabulary|math|science|history|biology|chemistry|physics|english|sat|act|punctuation|trig|statistics)\b", text, re.I))),
        "has_positive_framing":   int(bool(re.search(r"\b(excelled|impressed|strong|excellent|outstanding|exceptional|fantastic|amazing|wonderful|confident|enthusiasm|motivated|engaged|proud|pleasure|delight|love working)\b", text, re.I))),
        "has_improvement_language": int(bool(re.search(r"\b(improve|improving|progress|growth|developing|building|strengthen|working on|increased|higher|better|gains?|jumped?|boost)\b", text, re.I))),
        "has_goals":              int(bool(re.search(r"\b(goal|target|aim|objective|milestone|achieve|score|deadline|by the test|before the exam)\b", text, re.I))),
        "has_homework_mention":   int(bool(re.search(r"\b(homework|practice|review|assignment|study|worksheet|problems|exercises|mock|practice test|full.?length)\b", text, re.I))),
        "has_parent_action":      int(bool(re.search(r"\b(encourage|remind|support|reinforce|discuss|consider|reach out|schedule a call|give me a call|chat)\b", text, re.I))),
        "has_urgency":            int(bool(re.search(r"\b(running low|only \d+ hours?|limited|soon|deadline|before the|final|last few|few hours|before.*exam|crunch)\b", text, re.I))),
        "has_recommendation":     int(bool(re.search(r"\b(recommend|suggest|i would|i think|i believe|strongly|advise|best course|ideal)\b", text, re.I))),
        "has_hour_mention":       int(bool(re.search(r"\b(\d+\s*hours?|add hours?|more hours?|additional hours?|hours? remaining|hours? left)\b", text, re.I))),
        "has_scheduling":         int(bool(re.search(r"\b(schedule|calendly|calendar|availability|slot|time slot|book|session time|meet|meeting)\b", text, re.I))),
        "exclamation_count":      text.count("!"),
        "question_count":         text.count("?"),
        "unique_word_ratio":      len(set(words)) / len(words) if words else 0,
    }


# Closing salutations that only count when on a SHORT line (real closings are brief)
_CLOSING_WORDS = re.compile(
    r"^(?:best|regards|sincerely|warmly|cheers|thanks|thank you|many thanks|"
    r"kind regards|warm regards|all best|take care|talk soon|"
    r"have a (?:great|good|nice|wonderful)\s*\w*|"
    r"[-—]{2,})[\s,!.]*$",
    re.I
)

def strip_signature(text: str) -> str:
    """
    Remove signature from the bottom of a message only.
    Strategy: scan lines from the bottom up; once we hit a closing salutation
    (short line matching known patterns) or a line containing an email/phone,
    cut everything from that line downward.
    Also strip encoding artifacts, phone numbers, emails, and URLs throughout.
    """
    if not text:
        return text
    # Fix common encoding artifacts from Windows-1252 / latin-1 mojibake
    # Fix Windows-1252 mojibake artifacts
    for bad, good in [("\u00e2\u0080\u0099","'"),("\u00e2\u0080\u009c",'"'),
                      ("\u00e2\u0080\u009d",'"'),("\u00e2\u0080\u0094","\u2014"),
                      ("\u00e2\u0080\u0093","\u2013"),("\u00e2\u0080\u00a6","\u2026"),
                      ("\u00e2\u0080\u0098","'"),("\u00c3\u00a9","\u00e9"),
                      ("\u00c3\u00a0","\u00e0")]:
        text = text.replace(bad, good)
    # Strip emails, phones, URLs throughout (these are always noise)
    text = re.sub(r"\+?1?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", " ", text)
    text = re.sub(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Now find the signature cutpoint by scanning from the bottom
    lines = text.split("\n")
    cut = len(lines)
    for i in range(len(lines) - 1, max(len(lines) - 15, -1), -1):
        line = lines[i].strip()
        # A signature name line: "First L." or "First Last" alone on a line
        if re.match(r"^[A-Z][a-z]+ [A-Z][a-z\.]+$", line):
            cut = i
            continue
        # Short closing salutation
        if line and len(line) < 60 and _CLOSING_WORDS.match(line):
            cut = i
            continue
        # Line with only punctuation/whitespace — blank separator
        if not line or re.match(r"^[\s\W]+$", line):
            continue
        # Hit a real content line — stop scanning
        break
    text = "\n".join(lines[:cut]).strip()
    # Clean up leftover whitespace artifacts
    text = re.sub(r"[ 	]{2,}", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)





def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["repurchased"] = df["booking_after_progress_update"].notna()
    df["hours_remaining"] = (df["purchased_hours"] - df["delivered_hours"]).clip(lower=0)
    # stripped version used for language analysis only
    df["progress_update_clean"] = df["progress_update"].astype(str).apply(strip_signature)
    # Use raw text for feature extraction (word count, tone signals, etc.)
    # Clean text is only used for TF-IDF / keyword analysis to strip signatures
    feat_df = df["progress_update"].astype(str).apply(extract_features).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), feat_df], axis=1)


# ── Tone definitions ─────────────────────────────────────────────────────────
TONES = {
    "🎯 Goal-oriented": {
        "desc": "Focused on reaching a specific target score, grade, or outcome",
        "pattern": r"\b(goal|target|aim|objective|reach|achieve|get to|score of|need to|want to|by the|before the|in order to|so that|milestone)\b",
    },
    "⚠️ Urgency / Hours pitch": {
        "desc": "Emphasizes running low on hours or need to add more",
        "pattern": r"\b(running (low|out)|only \d+ hours?|few hours?|hours? (left|remaining)|add (more )?hours?|additional hours?|need more|recommend adding|purchase|limited hours?)\b",
    },
    "🌟 Celebration / Praise": {
        "desc": "Heavy on wins, pride, and positive framing of student progress",
        "pattern": r"\b(proud|incredible|amazing|fantastic|wonderful|excelled|outstanding|great job|great work|excellent|so happy|love working|love having|pleasure|doing wonderfully|shining|nailed|knocked it|great progress|huge improvement)\b",
    },
    "📋 Clinical / Report-style": {
        "desc": "Structured summary of what was covered and what comes next",
        "pattern": r"\b(summary|covered|reviewed|went over|topics|concepts|strategies|areas|section|module|unit|the following|as follows|in session|this week we|we have been|we worked)\b",
    },
    "🤝 Relationship / Warmth": {
        "desc": "Personal connection, warmth, expressing care for the student",
        "pattern": r"\b(pleasure (to work|working)|enjoy working|love (working|having|tutoring)|great (kid|student|young|person)|such a (great|wonderful|joy|pleasure)|genuinely|care about|rooting for|means a lot|really do|honor to)\b",
    },
    "🔧 Problem / Gap focused": {
        "desc": "Highlights what is not working, areas of concern, gaps to address",
        "pattern": r"\b(struggling|concern|concerned|challenge|challenging|difficult|difficulty|gap|gaps|weak|weakness|area(s)? (to|of) (improve|work|focus|address)|not yet|needs? (work|improvement|attention)|behind|falling behind|careless errors?|mistakes?)\b",
    },
    "📅 Planning / Logistics": {
        "desc": "Focused on scheduling, next session, calendar, logistics",
        "pattern": r"\b(schedule|scheduled|scheduling|calendly|calendar|availability|next (session|week|meeting|time)|upcoming|slot|book|rescheduled|time that works|reach out to schedule|set up|coordinate)\b",
    },
}

def score_tones(text: str) -> dict:
    """Return a 0/1 score per tone for a message."""
    if not text or not str(text).strip():
        return {t: 0 for t in TONES}
    text = str(text)
    return {
        tone: int(bool(re.search(info["pattern"], text, re.I)))
        for tone, info in TONES.items()
    }


def tone_repurchase_table(df):
    """For each tone, compute repurchase rate when present vs absent."""
    rows = []
    for tone in TONES:
        if tone not in df.columns:
            continue
        present = df[df[tone] == 1]
        absent  = df[df[tone] == 0]
        if len(present) < 5:
            continue
        rate_present = present["repurchased"].mean() * 100
        rate_absent  = absent["repurchased"].mean()  * 100
        diff = rate_present - rate_absent
        rows.append({
            "Tone": tone,
            "Description": TONES[tone]["desc"],
            "Messages with tone": len(present),
            "Repurchase rate (present)": f"{rate_present:.1f}%",
            "Repurchase rate (absent)":  f"{rate_absent:.1f}%",
            "Δ": round(diff, 1),
            "_sort": diff,
        })
    df_out = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort")
    return df_out


def style_tone_delta(val):
    try:
        v = float(val)
        if v >= 10:   return "color: #27ae60; font-weight: 800"
        elif v >= 4:  return "color: #2ecc71; font-weight: 600"
        elif v >= 0:  return "color: #58d68d"
        elif v >= -4: return "color: #f1948a"
        elif v >= -10: return "color: #e67e73; font-weight: 600"
        else:          return "color: #e74c3c; font-weight: 800"
    except:
        return ""


def tone_combo_table(df, min_count=5):
    """
    Simplify each message to at most 2 dominant tones:
      - If 0 tones: 'No tones detected'
      - If 1 tone:  show that tone alone
      - If 2+ tones: show only the top 2 by a fixed priority order
    Then group and sort by repurchase rate.
    """
    tone_cols = [t for t in TONES if t in df.columns]

    # Priority order — determines which 2 tones we keep when a message has many
    TONE_PRIORITY = [
        "⚠️ Urgency / Hours pitch",
        "🎯 Goal-oriented",
        "🌟 Celebration / Praise",
        "🔧 Problem / Gap focused",
        "📋 Clinical / Report-style",
        "🤝 Relationship / Warmth",
        "📅 Planning / Logistics",
    ]

    def dominant_label(row):
        present = [t for t in TONE_PRIORITY if t in tone_cols and row.get(t, 0) == 1]
        if not present:
            return "No tones detected"
        if len(present) == 1:
            return present[0]
        return present[0] + " + " + present[1]

    df2 = df.copy()
    df2["tone_label"] = df2.apply(dominant_label, axis=1)

    all_combos = (
        df2.groupby("tone_label")
        .agg(Count=("repurchased","count"), Repurchases=("repurchased","sum"))
        .assign(**{"Repurchase Rate %": lambda x: (x["Repurchases"]/x["Count"]*100).round(1)})
        .reset_index()
        .rename(columns={"tone_label": "Dominant Tone(s)"})
    )

    # Urgency rows always shown; others need min_count
    urgency_mask = all_combos["Dominant Tone(s)"].str.contains("Urgency", na=False)
    included = all_combos[(all_combos["Count"] >= min_count) | urgency_mask]
    return included.sort_values("Repurchase Rate %", ascending=False)


def enrich_tones(df: pd.DataFrame) -> pd.DataFrame:
    """Add tone columns to the dataframe.
    Score tones on raw text — the signature stripper is too aggressive and
    cuts message bodies on phrases like 'Have a great day', losing tone signals.
    TF-IDF distinctive language analysis still uses the clean version.
    """
    tone_df = df["progress_update"].astype(str).apply(score_tones).apply(pd.Series)
    return pd.concat([df, tone_df], axis=1)


def top_keywords(text_series, n=25):
    words = []
    for t in text_series.dropna():
        words += re.findall(r"\b[a-z]{4,}\b", str(t).lower())
    return Counter([w for w in words if w not in STOP_WORDS]).most_common(n)


def build_name_blocklist(df: pd.DataFrame) -> set:
    names = set()
    if "tutor_id" in df.columns:
        # No name col in this dataset — just block very short tokens
        pass
    # Also scrape names from signatures (Last line patterns like "Name L.\n")
    for txt in df["progress_update"].dropna().astype(str):
        # Grab email local parts and signature names
        emails = re.findall(r"([a-z]+)\.[a-z]+@revolutionprep", txt.lower())
        names.update(emails)
        sig_names = re.findall(r"\n([A-Z][a-z]+)\s+[A-Z][\.,]", txt)
        names.update(n.lower() for n in sig_names)
    return names


# ── Operational boilerplate to suppress ──────────────────────────────────────
# ANY phrase containing one of these tokens is dropped — not just phrases
# made entirely of them. This kills boilerplate fragments like
# "forward continuing support", "believe covers wanted", etc.
CONTENT_STOPWORDS = {
    # generic filler & sign-off words
    "forward","continuing","continued","support","journey","regards","highlight",
    "covers","concerns","believe","seen","improvements","current","short","term",
    "overall","hope","having","remains","looking","ahead","wanted","comments",
    "share","updates","academic","functioning","wishes","sincerely","warmly",
    "cheers","hello","dear","hey","hope","wonderful","thank","thanks","please",
    "know","let","also","very","well","make","sure","going","come","back","just",
    "like","really","update","progress","call","contact","reach","touch","feel",
    "free","question","anything","everything","something","spoke","speaking",
    "chat","want","provide","provided","summary","conversation","discussed",
    "discuss","mention","attached","below","above","following","regarding",
    "concerning","revolution","prep","revolutionprep","com","www","http","https",
    # tutoring ops boilerplate
    "hours","hour","session","sessions","work","working","student","tutor",
    "tutoring","class","today","during","week","weeks","month","months","time",
    # common tutor first names (scraped from data)
    "hailey","david","susan","jessica","lisa","amy","diana","karen","nisan",
    "vahagn","hannah","christina","shawn","sean","sydney","nandita","olivia",
    "eddie","kate","camila","melanie","dina","sarah","emily","michael",
}

THEMES = {
    "📈 Score & Progress":      r"\b(score|point|points|percent|increase|improved|improvement|higher|lower|gained|gains|result|results|test|exam|sat|act|practice)\b",
    "🗓 Planning & Next Steps":  r"\b(plan|next|upcoming|schedule|schedul|recommend|suggest|going forward|moving forward|continue|meet|meeting|calendly|calendar)\b",
    "⚠️ Urgency & Hours":       r"\b(remaining|left|low|running|add|adding|additional|more hours|limited|soon|deadline|before|crunch|last)\b",
    "💡 Skill & Subject":       r"\b(math|english|reading|writing|science|history|algebra|geometry|calculus|grammar|vocabulary|comprehension|essay|punctuation|trig|biology|chemistry|physics)\b",
    "🤝 Relationship & Tone":   r"\b(love|proud|pleasure|enjoy|wonderful|amazing|fantastic|motivated|confident|engaged|hard work|dedicated|great student|pleasure working)\b",
    "📋 Homework & Practice":   r"\b(homework|practice|assignment|review|worksheet|mock|full.length|problem|exercise|quiz|drill|complete|finish)\b",
}

def assign_theme(phrase):
    for theme, pattern in THEMES.items():
        if re.search(pattern, phrase, re.I):
            return theme
    return "🔍 Other"

def tfidf_distinctive(rep_texts, norep_texts, name_blocklist, n=20):
    if not rep_texts or not norep_texts:
        return {}, {}
    labels_arr = np.array([1]*len(rep_texts) + [0]*len(norep_texts))
    texts = rep_texts + norep_texts
    n_rep = int((labels_arr == 1).sum())
    n_nor = int((labels_arr == 0).sum())
    min_rep_count = max(3, int(n_rep * 0.04))
    min_nor_count = max(3, int(n_nor * 0.04))
    try:
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(2, 3),
            max_features=3000,
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
        )
        X = vec.fit_transform(texts).toarray()
        feat_names = vec.get_feature_names_out()

        def is_noise(phrase):
            tokens = phrase.lower().split()
            # Drop if ANY token is boilerplate — kills mixed fragments
            if any(tok in CONTENT_STOPWORDS for tok in tokens):
                return True
            if any(tok in name_blocklist for tok in tokens):
                return True
            if any(re.fullmatch(r"[\d\s\+\-\(\)\.]{4,}", tok) for tok in tokens):
                return True
            if any(re.fullmatch(r"\d+", tok) for tok in tokens):
                return True
            if any(tok in {"com","org","net","edu","co","io","gov","www"} for tok in tokens):
                return True
            if any(re.search(r"[-âÃ]", tok) for tok in tokens):
                return True
            if any(tok in {"iâ","ve","â","ã","don","ll","re","didn","isn","wasn","couldn","wouldn","haven"} for tok in tokens):
                return True
            if all(len(tok) <= 3 for tok in tokens):
                return True
            return False

        valid_idx = [i for i,f in enumerate(feat_names) if not is_noise(f)]
        X = X[:, valid_idx]
        feat_names = feat_names[valid_idx]

        eps = 1e-9
        rep_results, nor_results = [], []

        for i, phrase in enumerate(feat_names):
            col = X[:, i]
            present = (col > 0).astype(float)
            rep_present = present[labels_arr == 1]
            nor_present = present[labels_arr == 0]
            p_rep = rep_present.mean() + eps
            p_nor = nor_present.mean() + eps
            lr = np.log(p_rep / p_nor)
            rep_count = rep_present.sum()
            nor_count = nor_present.sum()

            if lr > 0.35 and rep_count >= min_rep_count:
                rep_results.append({"phrase": phrase, "lr": lr,
                                    "p_rep": round(p_rep*100,1), "p_nor": round(p_nor*100,1),
                                    "theme": assign_theme(phrase)})
            elif lr < -0.35 and nor_count >= min_nor_count:
                nor_results.append({"phrase": phrase, "lr": abs(lr),
                                    "p_rep": round(p_rep*100,1), "p_nor": round(p_nor*100,1),
                                    "theme": assign_theme(phrase)})

        rep_results = sorted(rep_results, key=lambda x: x["lr"], reverse=True)[:n]
        nor_results = sorted(nor_results, key=lambda x: x["lr"], reverse=True)[:n]

        def group_by_theme(results):
            grouped = {}
            for r in results:
                grouped.setdefault(r["theme"], []).append(r)
            return dict(sorted(grouped.items(), key=lambda x: max(r["lr"] for r in x[1]), reverse=True))

        return group_by_theme(rep_results), group_by_theme(nor_results)

    except Exception:
        return {}, {}



# ── Actionable message signals ───────────────────────────────────────────────
ACTIONABLE_SIGNALS = {
    "mentions_specific_score":    (r"\b\d{3,4}\b",                                                                                           "Mentions a specific score (e.g. 1250, 28)"),
    "mentions_score_gap":         (r"\b(from|went from|started at|increased from|up from|improved from)\b.{0,40}\b\d{3,4}\b",               "Mentions score improvement (from X to Y)"),
    "mentions_test_date":         (r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b.{0,20}\b(sat|act|test|exam)\b", "Mentions a specific upcoming test date"),
    "asks_to_add_hours":          (r"\b(add|adding|purchase|recommend adding|suggest adding)\b.{0,20}\bhours?\b",                            "Explicitly asks parent to add hours"),
    "proposes_session_frequency": (r"\b(\d+\s*(times?|sessions?|hours?)\s*(per|a|each)\s*(week|month)|once a week|twice a week|weekly|biweekly)\b", "Proposes a specific session frequency"),
    "requests_phone_call":        (r"\b(phone call|give me a call|schedule a call|hop on a call|quick call|talk on the phone)\b",            "Asks for a phone call with the parent"),
    "mentions_student_by_name":   (r"\b[A-Z][a-z]{2,}\b.{0,100}\b(improved|struggled|worked|scored|showed|completed|finished|doing)\b",    "Refers to student by name in context"),
    "gives_specific_homework":    (r"\b(assigned|assign|complete|finish|work on|practice)\b.{0,30}\b(problems?|questions?|sections?|pages?|tests?|sets?)\b", "Gives specific homework assignment"),
    "mentions_college_deadline":  (r"\b(college|application|deadline|common app|early decision|early action|admission)\b",                   "Mentions college applications or deadlines"),
    "mentions_specific_weakness": (r"\b(struggling|weak|weakness|needs work|area of improvement|working on|focus on)\b.{0,30}\b(math|reading|writing|science|english|grammar|punctuation|algebra|geometry|vocabulary)\b", "Identifies a specific subject weakness"),
    "has_concrete_plan":          (r"\b(plan is|our plan|the plan|going to|we will|we are going to|next steps are|here is what)\b",          "States a concrete plan going forward"),
    "mentions_practice_test":     (r"\b(practice test|mock test|full.?length|timed test|diagnostic|practice exam)\b",                        "Mentions a practice test"),
}

def score_actionable(text: str) -> dict:
    text = str(text)
    return {k: int(bool(re.search(pat, text, re.I | re.DOTALL))) for k, (pat, _) in ACTIONABLE_SIGNALS.items()}

def enrich_actionable(df: pd.DataFrame) -> pd.DataFrame:
    sig_df = df["progress_update"].astype(str).apply(score_actionable).apply(pd.Series)
    df = df.copy()
    for col in sig_df.columns:
        df[col] = sig_df[col].values
    return df

def load_data():
    df6  = pd.read_excel(EXCEL_PATH, sheet_name="0 to 6 hours remaining")
    df10 = pd.read_excel(EXCEL_PATH, sheet_name="0 to 10 hours remaining")
    return df6, df10


def actionable_table(rep, norep):
    rows = []
    for key, (_, label) in ACTIONABLE_SIGNALS.items():
        c_val  = rep[key].mean()   if key in rep.columns   else 0
        nc_val = norep[key].mean() if key in norep.columns else 0
        diff   = (c_val - nc_val) * 100
        rows.append({
            "Signal": label,
            "Repurchased": f"{c_val*100:.0f}%",
            "Not Repurchased": f"{nc_val*100:.0f}%",
            "Δ (pp)": round(diff, 1),
        })
    return pd.DataFrame(rows).sort_values("Δ (pp)", ascending=False)

def style_actionable_delta(val):
    try:
        v = float(val)
        if v >= 10:    return "color: #27ae60; font-weight: 800"
        elif v >= 4:   return "color: #2ecc71; font-weight: 600"
        elif v >= 0:   return "color: #58d68d"
        elif v >= -4:  return "color: #f1948a"
        elif v >= -10: return "color: #e67e73; font-weight: 600"
        else:          return "color: #e74c3c; font-weight: 800"
    except: return ""


FEATURE_LABELS = {
    "hours_remaining":          "Hours remaining at time of update",
    "char_count":               "Total characters written",
    "word_count":               "Total words written",
    "sentence_count":           "Total sentences",
    "avg_sentence_len":         "Avg sentence length (words)",
    "number_count":             "Numbers / figures mentioned",
    "paragraph_count":          "Number of paragraphs",
    "has_score_mention":        "Mentions a score or points",
    "has_next_steps":           "Includes next steps / forward-looking language",
    "has_specific_skills":      "Mentions a specific subject or skill",
    "has_positive_framing":     "Uses positive / enthusiastic language",
    "has_improvement_language": "Mentions improvement or progress",
    "has_goals":                "Mentions goals or targets",
    "has_homework_mention":     "Mentions homework or practice tests",
    "has_parent_action":        "Includes action item or call-to-action for parent",
    "has_urgency":              "Conveys urgency (low hours, deadline)",
    "has_recommendation":       "Contains explicit recommendation language",
    "has_hour_mention":         "Mentions adding / remaining hours",
    "has_scheduling":           "Mentions scheduling future sessions",
    "exclamation_count":        "Exclamation points used",
    "question_count":           "Questions asked",
    "unique_word_ratio":        "Vocabulary richness (unique word ratio)",
}


def magnitude_label(c_val, nc_val, feat):
    diff = c_val - nc_val
    is_binary = feat.startswith("has_")
    if is_binary:
        abs_pp = abs(diff) * 100
        if abs_pp < 2:
            return "≈ Similar", 0
        direction = "▲" if diff > 0 else "▼"
        word = "higher" if diff > 0 else "lower"
        sign = "+" if diff > 0 else "-"
        if abs_pp >= 25:
            return f"{direction}{direction}{direction} {sign}{abs_pp:.0f}pp — much {word} in repurchased", abs_pp
        elif abs_pp >= 12:
            return f"{direction}{direction} {sign}{abs_pp:.0f}pp — notably {word} in repurchased", abs_pp
        else:
            return f"{direction} {sign}{abs_pp:.0f}pp — {word} in repurchased", abs_pp
    else:
        if nc_val == 0:
            return "—", 0
        pct = (c_val - nc_val) / nc_val * 100
        abs_pct = abs(pct)
        if abs_pct < 3:
            return "≈ Similar", 0
        direction = "▲" if pct > 0 else "▼"
        word = "higher" if pct > 0 else "lower"
        sign = "+" if pct > 0 else "-"
        if abs_pct >= 50:
            return f"{direction}{direction}{direction} {sign}{abs_pct:.0f}% — dramatically {word} in repurchased", abs_pct
        elif abs_pct >= 20:
            return f"{direction}{direction} {sign}{abs_pct:.0f}% — notably {word} in repurchased", abs_pct
        else:
            return f"{direction} {sign}{abs_pct:.0f}% — {word} in repurchased", abs_pct


def style_difference(val):
    v = str(val)
    if v.startswith("▲▲▲") or v.startswith("▼▼▼"):
        return "color: #27ae60; font-weight: 800" if "▲" in v else "color: #e74c3c; font-weight: 800"
    elif v.startswith("▲▲") or v.startswith("▼▼"):
        return "color: #2ecc71; font-weight: 700" if "▲" in v else "color: #e67e73; font-weight: 700"
    elif v.startswith("▲"):
        return "color: #58d68d; font-weight: 500"
    elif v.startswith("▼"):
        return "color: #f1948a; font-weight: 500"
    return "color: gray; font-style: italic"


def logistic_importance(df):
    feature_cols = list(FEATURE_LABELS.keys())
    data = df[feature_cols + ["repurchased"]].dropna()
    if data["repurchased"].nunique() < 2 or len(data) < 20:
        return pd.DataFrame()
    X = data[feature_cols].values
    y = data["repurchased"].astype(int).values
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    try:
        clf = LogisticRegression(max_iter=500, C=0.5)
        clf.fit(X_norm, y)
        rows = []
        for feat, coef in zip(feature_cols, clf.coef_[0]):
            word = "more" if coef > 0 else "less"
            direction = "✅" if coef > 0 else "❌"
            rows.append({
                "Factor": FEATURE_LABELS[feat],
                "Strength": round(abs(coef), 3),
                "What this means": f"{direction} {word.capitalize()} '{FEATURE_LABELS[feat].lower()}' → {'more' if coef > 0 else 'less'} likely to repurchase",
            })
        return pd.DataFrame(rows).sort_values("Strength", ascending=False)
    except Exception:
        return pd.DataFrame()


def tutor_analysis(df):
    """Roll up to tutor level with repurchase rate and message quality signals."""
    feat_cols = list(FEATURE_LABELS.keys())
    agg_dict = {
        "repurchased":   ["count", "sum", "mean"],
        "char_count":    "mean",
        "word_count":    "mean",
        "has_next_steps": "mean",
        "has_urgency":   "mean",
        "has_recommendation": "mean",
        "has_hour_mention": "mean",
        "has_score_mention": "mean",
        "has_positive_framing": "mean",
        "has_improvement_language": "mean",
    }
    # Only include feat cols that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    tdf = df.groupby("tutor_id").agg(agg_dict)
    tdf.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in tdf.columns]
    tdf = tdf.rename(columns={
        "repurchased_count": "Updates Sent",
        "repurchased_sum":   "Repurchases",
        "repurchased_mean":  "Repurchase Rate",
        "char_count_mean":   "Avg Msg Length (chars)",
        "word_count_mean":   "Avg Words",
        "has_next_steps_mean":   "% Next Steps",
        "has_urgency_mean":      "% Urgency",
        "has_recommendation_mean": "% Recommendation",
        "has_hour_mention_mean": "% Hour Mention",
        "has_score_mention_mean": "% Score Mention",
        "has_positive_framing_mean": "% Positive Framing",
        "has_improvement_language_mean": "% Improvement Language",
    })
    tdf["Repurchase Rate"] = (tdf["Repurchase Rate"] * 100).round(1)
    for col in ["% Next Steps","% Urgency","% Recommendation","% Hour Mention",
                "% Score Mention","% Positive Framing","% Improvement Language"]:
        if col in tdf.columns:
            tdf[col] = (tdf[col] * 100).round(1)
    for col in ["Avg Msg Length (chars)","Avg Words"]:
        if col in tdf.columns:
            tdf[col] = tdf[col].round(0).astype(int)
    return tdf.reset_index().sort_values("Repurchase Rate", ascending=False)


def metric_card(col, label, value, sub=""):
    col.markdown(
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def render_analysis(df, label):
    """Render the full analysis for one dataset."""
    rep    = df[df["repurchased"]]
    norep  = df[~df["repurchased"]]
    total  = len(df)
    rate   = len(rep) / total * 100 if total else 0

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">Overview — {label}</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    metric_card(c1, "Total Updates", total)
    metric_card(c2, "Repurchased", len(rep), f"{rate:.1f}% rate")
    metric_card(c3, "Did Not Repurchase", len(norep))
    metric_card(c4, "Avg Words (Repurchased)",
        f"{rep['word_count'].mean():.0f}" if len(rep) else "—",
        f"vs {norep['word_count'].mean():.0f} no repurchase" if len(norep) else "")
    metric_card(c5, "Avg Hours Remaining",
        f"{rep['hours_remaining'].mean():.1f} repurch." if len(rep) else "—",
        f"vs {norep['hours_remaining'].mean():.1f} no repurch." if len(norep) else "")

    # ── Message quality ────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">Message Quality: Repurchased vs Not</div>', unsafe_allow_html=True)
    st.info(
        "💡 **These comparisons are unadjusted** — each feature is compared in isolation. "
        "A feature may look predictive here simply because it correlates with something else "
        "(e.g. longer messages naturally contain more of everything). "
        "See the **Logistic Regression** section below for which factors independently predict "
        "repurchase when controlling for all others."
    )

    rows_df = []
    for feat, lbl in FEATURE_LABELS.items():
        if feat not in df.columns:
            continue
        c_val  = rep[feat].mean()   if len(rep)   else 0
        nc_val = norep[feat].mean() if len(norep) else 0
        diff_label, sort_key = magnitude_label(c_val, nc_val, feat)
        if feat.startswith("has_"):
            c_disp, nc_disp = f"{c_val*100:.0f}%", f"{nc_val*100:.0f}%"
        else:
            c_disp, nc_disp = f"{c_val:.2f}", f"{nc_val:.2f}"
        rows_df.append({"_sort": sort_key, "Feature": lbl,
                        "Repurchased": c_disp, "No Repurchase": nc_disp,
                        "Difference": diff_label})

    quality_df = (pd.DataFrame(rows_df)
                  .sort_values("_sort", ascending=False)
                  .drop(columns="_sort").reset_index(drop=True))
    st.dataframe(
        quality_df.style.map(style_difference, subset=["Difference"]),
        use_container_width=True, hide_index=True,
    )
    st.caption("Sorted by magnitude. ▲▲▲/▼▼▼ = dramatic; ▲▲/▼▼ = notable; ▲/▼ = moderate.")

    # ── Message length sweet spot ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Message Length Sweet Spot</div>', unsafe_allow_html=True)
    st.caption(
        "Does longer always = better? This shows repurchase rate at each word-count bucket "
        "to reveal whether very long messages start to hurt."
    )
    df_len = df.copy()
    max_words = int(df_len["word_count"].quantile(0.97))  # cap outliers
    bin_edges = list(range(0, max_words + 50, 50))
    bin_labels = [f"{b}–{b+49}" for b in bin_edges[:-1]]
    df_len["word_bucket"] = pd.cut(
        df_len["word_count"].clip(upper=max_words),
        bins=bin_edges, labels=bin_labels, right=False
    )
    bucket_tbl = (
        df_len.groupby("word_bucket", observed=True)
        .agg(Messages=("repurchased","count"), Repurchases=("repurchased","sum"))
        .assign(**{"Repurchase Rate %": lambda x: (x["Repurchases"]/x["Messages"]*100).round(1)})
        .reset_index()
        .rename(columns={"word_bucket": "Word Count Range"})
    )
    # Only show buckets with at least 5 messages
    bucket_tbl = bucket_tbl[bucket_tbl["Messages"] >= 5]

    def style_rate(val):
        try:
            v = float(val)
            overall = df["repurchased"].mean() * 100
            if v >= overall + 10: return "color: #27ae60; font-weight: 800"
            elif v >= overall + 4: return "color: #2ecc71; font-weight: 600"
            elif v <= overall - 10: return "color: #e74c3c; font-weight: 800"
            elif v <= overall - 4: return "color: #e67e73; font-weight: 600"
            return ""
        except: return ""

    st.dataframe(
        bucket_tbl.style.map(style_rate, subset=["Repurchase Rate %"]),
        use_container_width=True, hide_index=True,
    )
    overall_rate = df["repurchased"].mean() * 100
    st.caption(f"Overall repurchase rate for this dataset: {overall_rate:.1f}%. Green = above average, red = below average.")

    # ── Distinctive Language ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Distinctive Language Analysis</div>', unsafe_allow_html=True)
    st.caption(
        "Phrases (2-3 words) statistically distinctive to each outcome, scored by likelihood ratio "
        "and requiring at least 5% presence in the group. Grouped by theme. Generic single words filtered out."
    )

    name_bl = build_name_blocklist(df)
    rep_texts   = rep["progress_update"].dropna().astype(str).tolist()
    norep_texts = norep["progress_update"].dropna().astype(str).tolist()
    grouped_rep, grouped_nor = tfidf_distinctive(rep_texts, norep_texts, name_bl)

    def render_grouped_language(grouped, tag_class, pct_col, other_col):
        if not grouped:
            st.info("Not enough distinctive phrases found.")
            return
        for theme, phrases in grouped.items():
            st.markdown(f"**{theme}**")
            tags_html = "".join(f'<span class="tag {tag_class}">{p["phrase"]}</span>' for p in phrases)
            st.markdown(f'<div class="insight-box" style="margin-bottom:0.4rem">{tags_html}</div>',
                        unsafe_allow_html=True)
            rows = [{
                "Phrase": p["phrase"],
                "In this group": f"{p[pct_col]:.0f}%",
                "In other group": f"{p[other_col]:.0f}%",
                "Strength": round(p["lr"], 2),
            } for p in phrases]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    tr, tnr = st.columns(2)
    with tr:
        st.markdown("**🟢 Language more common when families REPURCHASE**")
        render_grouped_language(grouped_rep, "tag-green", "p_rep", "p_nor")
    with tnr:
        st.markdown("**🔴 Language more common when families do NOT repurchase**")
        render_grouped_language(grouped_nor, "tag-red", "p_nor", "p_rep")

    # ── Actionable signals ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Actionable Message Signals</div>', unsafe_allow_html=True)
    st.caption(
        "Each message is scored on 12 concrete, hand-crafted signals that represent "
        "specific choices a tutor makes when writing. Sorted by Δ (percentage point difference "
        "between repurchased and not). These are the most directly actionable findings."
    )

    act_tbl = actionable_table(rep, norep)
    st.dataframe(
        act_tbl.style.map(style_actionable_delta, subset=["Δ (pp)"]),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Green = more common in repurchased families. Red = more common in non-repurchased. "
        "Note: these are unadjusted — see Logistic Regression for independent effects."
    )

    # ── Logistic regression───────────────────────
    st.markdown(f'<div class="section-header">What Predicts Repurchase? (Logistic Regression)</div>', unsafe_allow_html=True)
    st.caption("Ranked by strength. 'What this means' tells you the practical takeaway in plain English.")

    coef_df = logistic_importance(df)
    if not coef_df.empty:
        def hl(val):
            if "✅" in str(val): return "color: #2d8a4e; font-weight: 600"
            elif "❌" in str(val): return "color: #c0392b; font-weight: 600"
            return ""
        st.dataframe(coef_df.style.map(hl, subset=["What this means"]),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Not enough data to fit model.")

    # ── Tone analysis ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Message Tone Analysis</div>', unsafe_allow_html=True)
    st.caption(
        "Each message is scored on 7 tone dimensions. "
        "Tones are not mutually exclusive — a message can be both Celebratory and Goal-oriented. "
        "Δ = difference in repurchase rate when this tone is present vs absent."
    )

    tone_tbl = tone_repurchase_table(df)
    if not tone_tbl.empty:
        st.dataframe(
            tone_tbl.style.map(style_tone_delta, subset=["Δ"]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Not enough data for tone analysis.")

    st.markdown("**Tone combinations — sorted by repurchase rate**")
    overall_pct = df["repurchased"].mean() * 100
    st.caption(
        f"Sorted by repurchase rate (high → low). Minimum 5 messages per combo to appear. "
        f"Urgency combinations always shown regardless of frequency. "
        f"Overall rate for this dataset: **{overall_pct:.1f}%**."
    )
    combo_tbl = tone_combo_table(df)
    if not combo_tbl.empty:
        def style_combo_rate(val):
            try:
                v = float(val)
                overall = df["repurchased"].mean() * 100
                if v >= overall + 10: return "color: #27ae60; font-weight: 800"
                elif v >= overall + 4: return "color: #2ecc71; font-weight: 600"
                elif v <= overall - 10: return "color: #e74c3c; font-weight: 800"
                elif v <= overall - 4: return "color: #e67e73; font-weight: 600"
                return ""
            except: return ""
        st.dataframe(
            combo_tbl.style.map(style_combo_rate, subset=["Repurchase Rate %"]),
            use_container_width=True, hide_index=True,
        )

    # ── Tutor analysis ────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">Tutor-Level Analysis</div>', unsafe_allow_html=True)
    st.caption("Each row is one tutor. Sorted by repurchase rate descending. Min 3 updates to appear.")

    tdf = tutor_analysis(df)
    tdf = tdf[tdf["Updates Sent"] >= 3]

    # Sub-tabs: leaderboard + scatter
    t1, t2 = st.tabs(["📋 Tutor Leaderboard", "📊 Repurchase Rate vs Message Length"])

    with t1:
        def hl_rate(val):
            try:
                v = float(val)
                if v >= 60: return "color: #27ae60; font-weight: 700"
                elif v >= 40: return "color: #f39c12; font-weight: 600"
                else: return "color: #e74c3c"
            except: return ""
        st.dataframe(
            tdf.style.map(hl_rate, subset=["Repurchase Rate"]),
            use_container_width=True, hide_index=True,
        )

    with t2:
        if len(tdf) >= 5:
            import math
            # Simple text scatter using st.write
            st.caption("Each point represents a tutor. X = avg message length (words), Y = repurchase rate %")
            chart_df = tdf[["tutor_id","Avg Words","Repurchase Rate","Updates Sent"]].dropna()
            chart_df["tutor_id"] = chart_df["tutor_id"].astype(str)
            st.scatter_chart(chart_df.rename(columns={
                "Avg Words": "Avg Message Length (Words)",
                "Repurchase Rate": "Repurchase Rate (%)",
            }), x="Avg Message Length (Words)", y="Repurchase Rate (%)",
               size="Updates Sent", color="tutor_id")
        else:
            st.info("Need at least 5 tutors with 3+ updates to display chart.")

    # ── Raw keyword counts ─────────────────────────────────────────────────────
    with st.expander("📝 Raw Keyword Frequency"):
        kc, knc = st.columns(2)
        with kc:
            st.markdown("**Repurchased**")
            st.dataframe(pd.DataFrame(top_keywords(rep["progress_update_clean"]), columns=["Word","Count"]),
                         use_container_width=True, hide_index=True)
        with knc:
            st.markdown("**No Repurchase**")
            st.dataframe(pd.DataFrame(top_keywords(norep["progress_update_clean"]), columns=["Word","Count"]),
                         use_container_width=True, hide_index=True)

    # ── Raw data ──────────────────────────────────────────────────────────────
    with st.expander("📋 Raw Data"):
        display_cols = ["student_id","tutor_id","course_id","progress_update_sent_at",
                        "purchased_hours","delivered_hours","hours_remaining",
                        "repurchased","booking_after_progress_update",
                        "word_count","char_count","has_next_steps","has_urgency",
                        "has_recommendation","has_hour_mention","has_score_mention"]
        st.dataframe(df[[c for c in display_cols if c in df.columns]]
                     .sort_values("repurchased", ascending=False),
                     use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# Progress Update Repurchase Analyzer")
st.caption("App version: 2.0 — if you see this, new code is running")
st.markdown("*What makes families buy more hours after a progress update?*")
st.markdown("---")

try:
    raw6, raw10 = load_data()
except FileNotFoundError:
    st.error("❌ Could not find `ProgressUpdates_Repurchase.xlsx`. Place it in the same folder as this script.")
    st.stop()

# enrichment is now done inside load_data()
df6  = enrich_actionable(enrich_tones(enrich(raw6)))
df10 = enrich_actionable(enrich_tones(enrich(raw10)))

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    st.caption("Applied to both datasets simultaneously.")

    all_tutors_6  = sorted(df6["tutor_id"].unique())
    all_tutors_10 = sorted(df10["tutor_id"].unique())
    all_tutors    = sorted(set(all_tutors_6) | set(all_tutors_10))
    selected_tutors = st.multiselect("Tutor ID (optional)", all_tutors, default=[])

    date_min = min(df6["progress_update_sent_at"].min(), df10["progress_update_sent_at"].min())
    date_max = max(df6["progress_update_sent_at"].max(), df10["progress_update_sent_at"].max())
    date_range = st.date_input("Progress Update Date Range",
        value=(date_min.date(), date_max.date()),
        min_value=date_min.date(), max_value=date_max.date())

    min_words = st.slider("Min message length (words)", 0, 500, 0, 10)

def apply_filters(df):
    out = df.copy()
    if selected_tutors:
        out = out[out["tutor_id"].isin(selected_tutors)]
    if len(date_range) == 2:
        out = out[
            (pd.to_datetime(out["progress_update_sent_at"]).dt.date >= date_range[0]) &
            (pd.to_datetime(out["progress_update_sent_at"]).dt.date <= date_range[1])
        ]
    if min_words > 0:
        out = out[out["word_count"] >= min_words]
    return out

f6  = apply_filters(df6)
f10 = apply_filters(df10)

# ── Side-by-side tabs ─────────────────────────────────────────────────────────
tab6, tab10 = st.tabs(["⏱ 0 to 6 Hours Remaining", "⏱ 0 to 10 Hours Remaining"])

with tab6:
    render_analysis(f6, "0–6 Hours Remaining")

with tab10:
    render_analysis(f10, "0–10 Hours Remaining")