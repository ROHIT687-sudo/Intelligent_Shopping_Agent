"""
Intelligent Shopping Agent
IBM watsonx.ai · Streamlit UI
"""

import os
import re
import math
from collections import Counter

import streamlit as st
from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

load_dotenv()

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Shopping Agent",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global style ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Tighten metric card padding and add high-end drop shadow */
    [data-testid="metric-container"] {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
    }
    
    /* Premium Glowing Recommendation Box */
    .rec-box {
        background: linear-gradient(145deg, #f0f7ff, #e6f0fa);
        border-left: 5px solid #2563eb;
        border-radius: 4px 12px 12px 4px;
        padding: 22px 26px;
        font-size: 15.5px;
        line-height: 1.8;
        color: #1e293b;
        margin-top: 12px;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.1);
    }
    
    /* Responsive High-End Product Photo Showcase Frame */
    .product-showcase-frame {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        margin: 18px 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .product-showcase-frame:hover {
        box-shadow: 0 25px 30px -5px rgba(0, 0, 0, 0.15);
    }

    /* Section divider label */
    .section-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: #57606a;
        margin-bottom: 6px;
    }
    /* Vision badge */
    .badge-vision {
        display:inline-block;
        background:#ede9fe;color:#5b21b6;
        font-size:11px;font-weight:700;
        padding:2px 9px;border-radius:10px;
    }
    .badge-text {
        display:inline-block;
        background:#dbeafe;color:#1e40af;
        font-size:11px;font-weight:700;
        padding:2px 9px;border-radius:10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1 · DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════

def get_product_data() -> dict:
    """Mock product catalogue with price matrix, reviews, and materials."""
    return {
        "wireless_headphones": {
            "name": "SoundCore Pro Wireless Headphones",
            "prices": {"Amazon": 79.99, "eBay": 72.50, "Walmart": 84.99},
            "rating": 4.2,
            "reviews": [
                "Great sound quality and very comfortable for long sessions.",
                "Battery life is amazing, lasts all day without issues.",
                "Great sound quality and very comfortable for long sessions.",
                "The noise cancellation is decent but not class-leading.",
                "Good value for the price, works well with my phone.",
                "Great sound quality and very comfortable for long sessions.",
                "Solid build, the ear cushions feel premium.",
                "Occasional Bluetooth dropout but otherwise solid.",
                "Great sound quality and very comfortable for long sessions.",
                "Would recommend to anyone looking for budget cans.",
            ],
            "materials": ["recycled plastic", "vegan leather", "rechargeable battery"],
        },
        "running_shoes": {
            "name": "SwiftStride Running Shoes",
            "prices": {"Amazon": 110.00, "eBay": 95.00, "Walmart": 108.50},
            "rating": 4.5,
            "reviews": [
                "Very lightweight and responsive on the road.",
                "Best shoes I have owned in years, zero break-in time.",
                "True to size and ships fast.",
                "The cushioning is superb for long runs.",
                "Best shoes I have owned in years, zero break-in time.",
                "Durable outsole, still going strong after 200 miles.",
                "Colours are vibrant and exactly as pictured.",
                "Best shoes I have owned in years, zero break-in time.",
                "Slight smell out of the box but fades quickly.",
                "Great grip on wet surfaces.",
            ],
            "materials": ["organic cotton lining", "natural rubber sole", "recycled mesh upper"],
        },
        "coffee_maker": {
            "name": "BrewMaster 12-Cup Coffee Maker",
            "prices": {"Amazon": 54.99, "eBay": 60.00, "Walmart": 49.95},
            "rating": 3.8,
            "reviews": [
                "Makes a decent cup but takes longer than expected.",
                "Easy to clean and the carafe keeps coffee warm.",
                "Makes a decent cup but takes longer than expected.",
                "Stopped working after three months, disappointing.",
                "Makes a decent cup but takes longer than expected.",
                "Perfect size for a small household.",
                "The timer function is a great touch.",
                "Makes a decent cup but takes longer than expected.",
                "Carafe lid leaks occasionally.",
                "Good for the price, nothing fancy.",
            ],
            "materials": ["BPA-free plastic", "stainless steel carafe", "glass"],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2 · ANALYTICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def calculate_review_integrity(reviews: list[str]) -> str:
    """
    Bot-anomaly detection via repeated 4-word (4-gram) phrases.
    bot_ratio = duplicate_4grams / total_4grams
      < 0.15  → High  |  < 0.35 → Medium  |  ≥ 0.35 → Low
    """
    all_ngrams: list[tuple] = []
    for review in reviews:
        words = re.findall(r"\b\w+\b", review.lower())
        all_ngrams.extend(zip(words, words[1:], words[2:], words[3:]))

    if not all_ngrams:
        return "High"

    freq = Counter(all_ngrams)
    bot_ratio = sum(1 for c in freq.values() if c > 1) / len(all_ngrams)

    if bot_ratio < 0.15:
        return "High"
    if bot_ratio < 0.35:
        return "Medium"
    return "Low"


def forecast_purchase_timing(prices: dict[str, float]) -> str:
    """
    Coefficient of Variation (std_dev / mean) across merchant prices.
    CV < 0.05 → Buy Now (tight cluster)  |  CV ≥ 0.05 → Wait (price spike).
    """
    vals = list(prices.values())
    if len(vals) < 2:
        return "Buy Now"
    mean = sum(vals) / len(vals)
    std_dev = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    cv = std_dev / mean if mean else 0
    return "Buy Now" if cv < 0.05 else "Wait"


def calculate_sustainability_score(materials: list[str]) -> int:
    """
    Keyword-point table summed and clamped to [1, 10].
    recycled/organic=3 · natural/vegan=2 · bpa-free/rechargeable/steel/glass=1
    """
    eco = {
        "recycled": 3, "organic": 3,
        "natural": 2,  "vegan": 2,
        "bpa-free": 1, "rechargeable": 1, "stainless steel": 1, "glass": 1,
    }
    combined = " ".join(materials).lower()
    return max(1, min(10, sum(pts for kw, pts in eco.items() if kw in combined)))


# ═══════════════════════════════════════════════════════════════════════════
# 3 · WATSONX LAYER
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _init_client() -> APIClient:
    """Cached watsonx API client — created once per Streamlit session."""
    return APIClient(
        Credentials(
            url=os.environ["WATSONX_URL"],
            api_key=os.environ["WATSONX_API_KEY"],
        )
    )


def _describe_image(filename: str, text_query: str) -> str:
    """
    Builds a text descriptor from the uploaded file name and the user's query.
    Passed directly to granite-3-1-8b-base — no separate vision model call.
    """
    # Derive a human-readable label from the filename (strip extension, normalise separators)
    stem = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
    base = f'Product image: "{stem}"'
    if text_query.strip():
        return f"{base}. Additional context: {text_query.strip()}"
    return base


def _build_prompt(query: str, product: dict, integrity: str, timing: str, eco: int) -> str:
    prices = product["prices"]
    best = min(prices, key=prices.get)
    price_str = " · ".join(f"{m} ${p:.2f}" for m, p in prices.items())
    return (
        f"You are a concise, data-driven shopping advisor.\n\n"
        f'Product search: "{query}"\n'
        f"Best match: {product['name']}\n"
        f"Rating: {product['rating']}/5\n"
        f"Prices: {price_str}\n"
        f"Cheapest: {best} at ${prices[best]:.2f}\n"
        f"Review integrity: {integrity}\n"
        f"Purchase timing: {timing}\n"
        f"Sustainability: {eco}/10\n\n"
        "Write exactly 3 sentences: (1) best deal and where to buy, "
        "(2) review trustworthiness and price timing, "
        "(3) eco-friendliness verdict. Be direct and specific."
    )


def get_recommendation(
    query: str,
    product: dict,
    integrity: str,
    timing: str,
    eco: int,
    project_id: str,
) -> str:
    """meta-llama/llama-3-3-70b-instruct: final buyer recommendation."""
    prompt = _build_prompt(query, product, integrity, timing, eco)
    model = ModelInference(
        model_id="meta-llama/llama-3-3-70b-instruct",
        api_client=_init_client(),
        project_id=project_id,
        params={
            GenParams.MAX_NEW_TOKENS: 300,
            GenParams.TEMPERATURE: 0.3,
            GenParams.STOP_SEQUENCES: ["\n\n"],
        },
    )
    resp = model.generate(prompt=prompt)
    return resp["results"][0]["generated_text"].strip()


# ═══════════════════════════════════════════════════════════════════════════
# 4 · UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════

_INTEGRITY_META = {
    "High":   ("✅ High",   "Organic reviews",       "#16a34a"),
    "Medium": ("⚠️ Medium", "Some repetition",        "#d97706"),
    "Low":    ("🚨 Low",    "Bot patterns detected",  "#dc2626"),
}

_TIMING_META = {
    "Buy Now": ("🟢 Buy Now", "Prices are stable"),
    "Wait":    ("🔴 Wait",    "Price spike detected"),
}

def _render_price_table(prices: dict[str, float]) -> None:
    best = min(prices, key=prices.get)
    rows = "".join(
        f"| {'**' + m + '**' if m == best else m} "
        f"| {'**' + f'${p:.2f}' + ' ✓ Best' + '**' if m == best else f'${p:.2f}'} |\n"
        for m, p in prices.items()
    )
    st.markdown(
        f"| Merchant | Price |\n|---|---|\n{rows}",
        unsafe_allow_html=False,
    )


def _eco_bar(score: int) -> str:
    filled = "█" * score
    empty  = "░" * (10 - score)
    colour = "#16a34a" if score >= 7 else "#d97706" if score >= 4 else "#dc2626"
    return (
        f"<span style='font-family:monospace;color:{colour};font-size:18px;'>"
        f"{filled}{empty}</span> "
        f"<span style='font-size:13px;color:#57606a;'>{score}/10</span>"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5 · SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🛍️ Shopping Agent")
    st.caption("Powered by IBM watsonx.ai")
    st.divider()

    catalogue = get_product_data()
    product_labels = {k: v["name"] for k, v in catalogue.items()}
    selected_key = st.selectbox(
        "Product",
        options=list(product_labels.keys()),
        format_func=lambda k: product_labels[k],
    )
    product = catalogue[selected_key]

    st.divider()
    st.markdown('<p class="section-label">Input mode</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload product image (optional)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Uses granite-3-0-vision when an image is provided.",
    )
    text_query = st.text_input(
        "Or type a search query",
        placeholder="e.g. noise-cancelling headphones under $100",
        disabled=uploaded_file is not None,
    )

    st.divider()
    run_btn = st.button("🔍 Analyse & Recommend", use_container_width=True, type="primary")

    st.divider()
    st.markdown('<p class="section-label">Credentials (via .env)</p>', unsafe_allow_html=True)
    cred_ok = all(os.environ.get(k) for k in ("WATSONX_URL", "WATSONX_API_KEY", "WATSONX_PROJECT_ID"))
    st.markdown(
        "🟢 `.env` loaded" if cred_ok else "🔴 `.env` missing — see `.env.example`",
        unsafe_allow_html=False,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6 · MAIN PANEL
# ═══════════════════════════════════════════════════════════════════════════

st.title("🛍️ Intelligent Shopping Agent")
st.caption(
    "Select a product, enter a query or upload a photo, then hit **Analyse & Recommend**."
)
st.divider()

# ── Always-visible analytics ────────────────────────────────────────────────
integrity    = calculate_review_integrity(product["reviews"])
timing       = forecast_purchase_timing(product["prices"])
eco_score    = calculate_sustainability_score(product["materials"])

int_label, int_help, _ = _INTEGRITY_META[integrity]
tim_label, tim_help    = _TIMING_META[timing]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Product Rating", f"{product['rating']} / 5 ⭐")
col2.metric("Review Integrity", int_label, help=int_help)
col3.metric("Purchase Timing",  tim_label, help=tim_help)
col4.metric("Sustainability",   f"{eco_score} / 10")

st.divider()

# ── Two-column layout: prices left, eco + reviews right ─────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<p class="section-label">Price Comparison</p>', unsafe_allow_html=True)
    _render_price_table(product["prices"])

with right:
    st.markdown('<p class="section-label">Eco Score</p>', unsafe_allow_html=True)
    st.markdown(_eco_bar(eco_score), unsafe_allow_html=True)
    st.caption("Materials: " + " · ".join(product["materials"]))

    st.markdown("")
    st.markdown('<p class="section-label">Customer Reviews</p>', unsafe_allow_html=True)
    unique_reviews = list(dict.fromkeys(product["reviews"]))   # preserve order, dedupe
    with st.expander(f"Show {len(unique_reviews)} unique reviews ({len(product['reviews'])} total)"):
        for r in unique_reviews:
            st.markdown(f"- {r}")

st.divider()

# ── Interactive shopping list ────────────────────────────────────────────────
st.markdown('<p class="section-label">Shopping Checklist</p>', unsafe_allow_html=True)
checklist_items = [
    f"Buy from {min(product['prices'], key=product['prices'].get)} (best price)",
    "Compare return / warranty policies",
    "Check current delivery timeline",
    "Read 1-star reviews for known defects",
    "Verify seller rating before purchase",
]
checked_items = []
check_cols = st.columns(len(checklist_items))
for col, item in zip(check_cols, checklist_items):
    if col.checkbox(item, key=f"chk_{item}"):
        checked_items.append(item)

if checked_items:
    st.caption(f"✅ {len(checked_items)} of {len(checklist_items)} steps complete")

st.divider()

# ── watsonx recommendation ───────────────────────────────────────────────────
st.markdown('<p class="section-label">AI Recommendation</p>', unsafe_allow_html=True)

if run_btn:
    if not cred_ok:
        st.error("watsonx credentials are missing. Add them to your `.env` file and restart.")
    elif not uploaded_file and not text_query.strip():
        st.warning("Enter a text query or upload an image to generate a recommendation.")
    else:
        project_id = os.environ["WATSONX_PROJECT_ID"]

        # ── Dual model path ──────────────────────────────────────────────
        if uploaded_file:
            model_badge = '<span class="badge-vision">meta-llama/llama-3-3-70b-instruct (image + text)</span>'
            st.markdown(f"Using {model_badge}", unsafe_allow_html=True)
            image_bytes = uploaded_file.read()
            effective_query = _describe_image(uploaded_file.name, text_query)
            st.image(image_bytes, caption="Uploaded image", width=280)
            with st.expander("Image descriptor passed to model"):
                st.write(effective_query)
        else:
            model_badge = '<span class="badge-text">meta-llama/llama-3-3-70b-instruct</span>'
            st.markdown(f"Using {model_badge}", unsafe_allow_html=True)
            effective_query = text_query.strip()

        # ── Call instruct model ──────────────────────────────────────────
        with st.spinner("Generating recommendation…"):
            recommendation = get_recommendation(
                effective_query, product, integrity, timing, eco_score, project_id
            )

        st.markdown(
            f'<div class="rec-box">{recommendation}</div>',
            unsafe_allow_html=True,
        )

        # ── Product summary card ─────────────────────────────────────────
        best_merchant = min(product["prices"], key=product["prices"].get)
        best_price    = product["prices"][best_merchant]
        materials_str = " · ".join(product["materials"])
        st.markdown(
            f"""
            <div style="
                background:#f7f8fa;
                border:1px solid #e5e7eb;
                border-radius:10px;
                padding:18px 22px;
                margin-top:16px;
            ">
                <div style="font-size:13px;font-weight:700;letter-spacing:.06em;
                            text-transform:uppercase;color:#57606a;margin-bottom:10px;">
                    Product Summary
                </div>
                <table style="width:100%;border-collapse:collapse;font-size:14px;color:#1f2328;">
                    <tr>
                        <td style="padding:5px 0;color:#57606a;width:38%;">Product</td>
                        <td style="padding:5px 0;font-weight:600;">{product['name']}</td>
                    </tr>
                    <tr>
                        <td style="padding:5px 0;color:#57606a;">Best price</td>
                        <td style="padding:5px 0;font-weight:600;">
                            ${best_price:.2f} at {best_merchant}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding:5px 0;color:#57606a;">Query</td>
                        <td style="padding:5px 0;">{effective_query}</td>
                    </tr>
                    <tr>
                        <td style="padding:5px 0;color:#57606a;">Materials</td>
                        <td style="padding:5px 0;">{materials_str}</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


else:
    st.info("Configure inputs in the sidebar and click **Analyse & Recommend**.")
