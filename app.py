import json
import re
import time
from pathlib import Path
from urllib import request

import streamlit as st
import torch
import torch.nn.functional as F
import open_clip


# ============================================================
# CONFIG
# ============================================================

INDEX_DIR = Path("indexes")

INDEX_OPTIONS = {
    "500": INDEX_DIR / "deepfashion_index_500.pt",
    "5,000": INDEX_DIR / "deepfashion_index_5000.pt",
    "50,000": INDEX_DIR / "deepfashion_index_50000.pt",
    "200,000": INDEX_DIR / "deepfashion_index_200000.pt",
    "289,222": INDEX_DIR / "deepfashion_index_289222.pt",
}

DEFAULT_INDEX_KEY = "289,222"

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

OLLAMA_URL = "http://localhost:11434/api/generate"

OLLAMA_MODELS = [
    "qwen3.5:4b",
    "llama3.2:3b",
]

DISPLAY_TOP_K = 5
CANDIDATE_TOP_K = 20

DEFAULT_MIN_SCORE = 0.20


# ============================================================
# FASHION ROUTING VOCABULARY
# ============================================================

FASHION_TERMS = {
    "dress",
    "shirt",
    "t-shirt",
    "tshirt",
    "tee",
    "jacket",
    "coat",
    "cardigan",
    "sweater",
    "hoodie",
    "pants",
    "trousers",
    "jeans",
    "shorts",
    "skirt",
    "blouse",
    "tank",
    "tank top",
    "boxer",
    "boxers",
    "underwear",
    "polo",
    "vest",
    "suit",
    "blazer",
    "kimono",
    "top",
    "garment",
    "jumper",
    "sweatshirt",
    "parka",
    "trench",
    "leggings",
    "chinos",
    "slacks",
}


# These words are suspicious ONLY when we do not already
# have clear fashion / garment context.
OUT_OF_SCOPE_MARKERS = {
    "write",
    "code",
    "program",
    "programming",
    "c++",
    "python",
    "java",
    "javascript",
    "explain",
    "what is",
    "who is",
    "history",
    "news",
    "recipe",
    "calculate",
    "equation",
}


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Miranda",
    page_icon="👠",
    layout="wide",
)

st.title("👠 Miranda")

st.caption(
    "Private multilingual fashion retrieval — "
    "Local LLM + OpenCLIP + PyTorch GEMM"
)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

elif torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

else:

    DEVICE = torch.device("cpu")


def sync_device():

    if DEVICE.type == "mps":

        torch.mps.synchronize()

    elif DEVICE.type == "cuda":

        torch.cuda.synchronize()


# ============================================================
# LOAD CLIP
# ============================================================

@st.cache_resource
def load_clip():

    model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL,
        pretrained=CLIP_PRETRAINED
    )

    model = model.to(DEVICE)
    model.eval()

    tokenizer = open_clip.get_tokenizer(
        CLIP_MODEL
    )

    return model, tokenizer


clip_model, tokenizer = load_clip()


# ============================================================
# LOAD INDEX
# ============================================================

@st.cache_resource
def load_index(
    index_path_str: str
):

    index_path = Path(
        index_path_str
    )

    data = torch.load(
        index_path,
        map_location="cpu"
    )

    embeddings = data[
        "embeddings"
    ].to(DEVICE)

    paths = data[
        "paths"
    ]

    catalog_size = data.get(
        "catalog_size",
        embeddings.shape[0]
    )

    dimensions = embeddings.shape[1]

    file_size_mib = (
        index_path.stat().st_size
        / 1024
        / 1024
    )

    return (
        embeddings,
        paths,
        catalog_size,
        dimensions,
        file_size_mib,
    )


# ============================================================
# ROUTING HELPERS
# ============================================================

def contains_non_ascii(
    text: str
) -> bool:

    return not text.isascii()


def contains_fashion_term(
    text: str
) -> bool:
    """
    Safer than `term in text`.

    Avoids accidental matches such as:
        "top" inside "laptop"

    Multi-word phrases such as "tank top"
    are also supported.
    """

    text = text.lower()

    for term in FASHION_TERMS:

        escaped = re.escape(
            term.lower()
        )

        pattern = (
            r"(?<![a-z0-9])"
            + escaped
            + r"(?![a-z0-9])"
        )

        if re.search(
            pattern,
            text
        ):

            return True

    return False


def contains_out_of_scope_marker(
    text: str
) -> bool:

    t = text.lower()

    return any(
        marker in t
        for marker in OUT_OF_SCOPE_MARKERS
    )


def is_obvious_english_fashion_query(
    text: str
) -> bool:
    """
    V9 routing rule:

    Explicit garment context wins.

    Examples:

        black dress
        -> direct CLIP

        T-shirt with computer code
        -> direct CLIP

        black T-shirt with programming text
        -> direct CLIP

        write C++ hello world
        -> NOT direct CLIP
        -> LLM domain gate
    """

    if contains_non_ascii(
        text
    ):

        return False


    # --------------------------------------------------------
    # IMPORTANT V9 CHANGE
    #
    # If the user explicitly mentions a garment,
    # treat the request as a fashion retrieval query.
    #
    # "code" does not make "T-shirt with code"
    # a programming request.
    # --------------------------------------------------------

    if contains_fashion_term(
        text
    ):

        return True


    # --------------------------------------------------------
    # No garment context.
    # Suspicious/general-purpose terms now matter.
    # --------------------------------------------------------

    if contains_out_of_scope_marker(
        text
    ):

        return False


    return False


# ============================================================
# OLLAMA
# ============================================================

def call_ollama_json(
    prompt: str,
    ollama_model: str
) -> dict:

    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,

        # Force machine-readable response.
        "format": "json",

        # Query normalization does not need
        # reasoning / chain-of-thought.
        "think": False,

        "options": {
            "temperature": 0
        }
    }

    req = request.Request(
        OLLAMA_URL,
        data=json.dumps(
            payload
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with request.urlopen(
        req,
        timeout=60
    ) as response:

        outer = json.loads(
            response
            .read()
            .decode("utf-8")
        )


    raw = outer.get(
        "response",
        ""
    )

    if raw is None:

        raw = ""

    raw = raw.strip()


    if not raw:

        thinking = outer.get(
            "thinking",
            ""
        )

        raise RuntimeError(
            "LLM returned an empty response. "
            f"Thinking output: "
            f"{thinking[:300]!r}"
        )


    # Defensive cleanup if model adds
    # markdown around JSON.
    if raw.startswith(
        "```"
    ):

        raw = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw,
            flags=re.IGNORECASE
        )

        raw = re.sub(
            r"\s*```$",
            "",
            raw
        )


    try:

        return json.loads(
            raw
        )

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "LLM did not return valid JSON.\n"
            f"Model: {ollama_model}\n"
            f"Raw response: {raw[:500]!r}"
        ) from exc


# ============================================================
# NON-ASCII TRANSLATION + DOMAIN CHECK
# ============================================================

def normalize_multilingual_query(
    user_message: str,
    previous_query: str | None,
    ollama_model: str
):

    previous = (
        previous_query
        or "NONE"
    )


    prompt = f"""
You are Miranda's multilingual fashion-search normalizer.

CURRENT USER MESSAGE:
{user_message}

PREVIOUS ENGLISH FASHION QUERY:
{previous}


Your task has TWO parts:

1. Decide whether the current request is about
   fashion/clothing retrieval.

2. If it is fashion-related, produce a concise
   English query for CLIP.


Fashion includes:

- garments
- clothing
- fashion graphics
- printed text on clothing
- printed computer code on clothing
- logos on clothing
- colors
- patterns
- materials
- sleeves
- collars
- length
- fit
- silhouette
- texture
- season
- formality
- refinements of an earlier fashion query


IMPORTANT DISTINCTION:

A request to WRITE programming code is not fashion.

But a request to FIND CLOTHING that contains printed
computer code, formulas, technical text or source code
IS a fashion retrieval request.


Example:

CURRENT:
C++でHello Worldを書いて

OUTPUT:
{{
    "is_fashion": false,
    "is_followup": false,
    "query_en": ""
}}


But:

CURRENT:
プログラムコードが印刷された黒いTシャツを見せて

OUTPUT:
{{
    "is_fashion": true,
    "is_followup": false,
    "query_en": "black T-shirt with programming code"
}}


Non-fashion examples:

- write programming code
- explain C++
- Python tutorials
- mathematics questions
- history
- politics
- news
- travel
- medicine
- recipes
- general knowledge


If CURRENT USER MESSAGE is a short FOLLOW-UP
to the previous fashion query, preserve previous
relevant attributes and apply the requested change.


Examples:


CURRENT:
冬に着る暖かいコートを見せて

PREVIOUS:
NONE

OUTPUT:
{{
    "is_fashion": true,
    "is_followup": false,
    "query_en": "warm winter coat"
}}


CURRENT:
もう少し軽いもの

PREVIOUS:
warm winter coat

OUTPUT:
{{
    "is_fashion": true,
    "is_followup": true,
    "query_en": "lightweight warm winter coat"
}}


CURRENT:
黒ではなくベージュで

PREVIOUS:
lightweight warm winter coat

OUTPUT:
{{
    "is_fashion": true,
    "is_followup": true,
    "query_en": "lightweight warm beige winter coat"
}}


CURRENT:
青と白のストライプシャツを見せて

PREVIOUS:
NONE

OUTPUT:
{{
    "is_fashion": true,
    "is_followup": false,
    "query_en": "blue and white striped shirt"
}}


CURRENT:
メンズのタンクトップを見せて

PREVIOUS:
NONE

OUTPUT:
{{
    "is_fashion": true,
    "is_followup": false,
    "query_en": "men's tank top"
}}


CURRENT:
C++でHello Worldを書いて

PREVIOUS:
NONE

OUTPUT:
{{
    "is_fashion": false,
    "is_followup": false,
    "query_en": ""
}}


IMPORTANT:

- Always return English in query_en.
- Preserve explicit colors literally.
- Preserve clothing type.
- Preserve requests for printed technical graphics.
- 青 = blue
- 白 = white
- 緑 = green
- 赤 = red
- 黒 = black
- ベージュ = beige
- Do not answer the user's question.
- Do not explain.
- Return JSON only.


Return:

{{
    "is_fashion": true,
    "is_followup": false,
    "query_en": "..."
}}
"""


    data = call_ollama_json(
        prompt,
        ollama_model
    )


    is_fashion = bool(
        data.get(
            "is_fashion",
            False
        )
    )

    is_followup = bool(
        data.get(
            "is_followup",
            False
        )
    )

    query = str(
        data.get(
            "query_en",
            ""
        )
    ).strip()

    query = (
        query
        .strip('"')
        .strip("'")
        .strip()
    )


    if not is_fashion:

        return {
            "domain": "out_of_scope",
            "is_followup": False,
            "query_en": "",
        }


    if not query:

        return {
            "domain": "out_of_scope",
            "is_followup": False,
            "query_en": "",
        }


    return {
        "domain": "fashion_search",
        "is_followup": is_followup,
        "query_en": query,
    }


# ============================================================
# ASCII AMBIGUOUS DOMAIN CHECK
# ============================================================

def interpret_ascii_query(
    user_message: str,
    previous_query: str | None,
    ollama_model: str
):

    previous = (
        previous_query
        or "NONE"
    )


    prompt = f"""
You are the intent classifier for Miranda,
a fashion archive search application.

CURRENT USER MESSAGE:
{user_message}

PREVIOUS FASHION QUERY:
{previous}


Miranda supports ONLY fashion/clothing retrieval.


Fashion includes:

- garments
- clothing
- visual designs
- graphics printed on clothing
- technical text printed on clothing
- source code printed on clothing
- mathematical formulas printed on clothing
- logos printed on clothing


IMPORTANT:

"write C++ hello world"

is NOT a fashion request.


But:

"T-shirt with C++ code"

"T-shirt with shell code"

"black T-shirt with programming text"

"T-shirt with physics formula"

ARE valid fashion retrieval requests.


If this is fashion-related, return a concise
English CLIP query.

If it is unrelated, return out_of_scope.

Short follow-up phrases may modify the previous
fashion query.


Examples:


"black dress"

OUTPUT:

{{
    "domain": "fashion_search",
    "is_followup": false,
    "query_en": "black dress"
}}


"make it shorter"

PREVIOUS:
black dress

OUTPUT:

{{
    "domain": "fashion_search",
    "is_followup": true,
    "query_en": "short black dress"
}}


"T-shirt with shell code"

OUTPUT:

{{
    "domain": "fashion_search",
    "is_followup": false,
    "query_en": "T-shirt with shell code"
}}


"write a C++ hello world program"

OUTPUT:

{{
    "domain": "out_of_scope",
    "is_followup": false,
    "query_en": ""
}}


Return JSON only.
"""


    data = call_ollama_json(
        prompt,
        ollama_model
    )


    domain = str(
        data.get(
            "domain",
            "out_of_scope"
        )
    ).strip()


    query = str(
        data.get(
            "query_en",
            ""
        )
    ).strip()


    is_followup = bool(
        data.get(
            "is_followup",
            False
        )
    )


    if domain != "fashion_search":

        return {
            "domain": "out_of_scope",
            "is_followup": False,
            "query_en": "",
        }


    return {
        "domain": "fashion_search",
        "is_followup": is_followup,
        "query_en": query,
    }


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "messages": [],
    "last_retrieval_query": None,
    "pending_message": None,
    "pending_model": None,
    "pending_min_score": None,
    "pending_index_key": None,
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[
            key
        ] = value


# ============================================================
# SIDEBAR
# ============================================================

processing = (
    st.session_state.pending_message
    is not None
)


with st.sidebar:

    st.subheader(
        "Miranda"
    )

    st.write(
        f"Device: `{DEVICE}`"
    )

    st.divider()


    index_keys = list(
        INDEX_OPTIONS.keys()
    )


    selected_index_key = st.selectbox(
        "Catalog size",
        index_keys,
        index=index_keys.index(
            DEFAULT_INDEX_KEY
        ),
        disabled=processing
    )


    selected_index_path = (
        INDEX_OPTIONS[
            selected_index_key
        ]
    )


    (
        catalog,
        paths,
        catalog_size,
        dimensions,
        index_file_mib,
    ) = load_index(
        str(
            selected_index_path
        )
    )


    st.write(
        f"Catalog vectors: "
        f"`{catalog_size:,}`"
    )

    st.write(
        f"Embedding dimensions: "
        f"`{dimensions}`"
    )

    st.write(
        f"Index file: "
        f"`{index_file_mib:.1f} MiB`"
    )


    raw_matrix_mib = (
        catalog_size
        * dimensions
        * 4
        / 1024
        / 1024
    )


    st.write(
        f"Raw FP32 matrix: "
        f"`{raw_matrix_mib:.1f} MiB`"
    )

    st.divider()


    ollama_model = st.selectbox(
        "Local LLM",
        OLLAMA_MODELS,
        index=0,
        disabled=processing
    )


    st.write(
        f"Selected model: "
        f"`{ollama_model}`"
    )

    st.divider()


    min_score = st.slider(
        "Minimum CLIP similarity",
        min_value=0.00,
        max_value=0.40,
        value=DEFAULT_MIN_SCORE,
        step=0.01,
        disabled=processing
    )


    st.caption(
        "Results below this threshold "
        "will not be displayed."
    )

    st.divider()


    if st.button(
        "New search",
        use_container_width=True,
        disabled=processing
    ):

        st.session_state.messages = []
        st.session_state.last_retrieval_query = None

        st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["text"]
        )


        if message.get(
            "catalog_size"
        ):

            st.caption(
                f'Catalog: '
                f'{message["catalog_size"]:,} vectors'
            )


        if message.get(
            "route"
        ):

            st.caption(
                f'Route: '
                f'{message["route"]}'
            )


        if message.get(
            "llm_model"
        ):

            st.caption(
                f'LLM: '
                f'`{message["llm_model"]}`'
            )


        if message.get(
            "retrieval_query"
        ):

            st.caption(
                f'CLIP query: '
                f'"{message["retrieval_query"]}"'
            )


        if message.get(
            "interpretation"
        ):

            st.caption(
                f'Interpretation: '
                f'{message["interpretation"]}'
            )


        if message.get(
            "timing"
        ):

            t = message[
                "timing"
            ]

            st.caption(
                f'LLM: {t["llm_ms"]:.1f} ms | '
                f'CLIP: {t["clip_ms"]:.1f} ms | '
                f'GEMM: {t["gemm_ms"]:.3f} ms | '
                f'Top-K: {t["topk_ms"]:.3f} ms'
            )


        if message.get(
            "results"
        ):

            cols = st.columns(
                len(
                    message["results"]
                )
            )


            for col, result in zip(
                cols,
                message["results"]
            ):

                with col:

                    st.image(
                        result[
                            "path"
                        ],
                        use_container_width=True
                    )


                    st.caption(
                        f'{result["score"]:.3f}\n\n'
                        f'{result["name"]}'
                    )


# ============================================================
# INPUT
# ============================================================

user_message = st.chat_input(
    "Ask Miranda's fashion archive...",
    disabled=processing
)


if user_message:

    st.session_state.messages.append(
        {
            "role": "user",
            "text": user_message,
        }
    )


    st.session_state.pending_message = (
        user_message
    )

    st.session_state.pending_model = (
        ollama_model
    )

    st.session_state.pending_min_score = (
        min_score
    )

    st.session_state.pending_index_key = (
        selected_index_key
    )


    st.rerun()


# ============================================================
# PROCESS PENDING
# ============================================================

if (
    st.session_state.pending_message
    is not None
):

    pending_message = (
        st.session_state.pending_message
    )

    pending_model = (
        st.session_state.pending_model
    )

    pending_min_score = (
        st.session_state.pending_min_score
    )

    pending_index_key = (
        st.session_state.pending_index_key
    )


    pending_index_path = (
        INDEX_OPTIONS[
            pending_index_key
        ]
    )


    (
        pending_catalog,
        pending_paths,
        pending_catalog_size,
        pending_dimensions,
        pending_index_mib,
    ) = load_index(
        str(
            pending_index_path
        )
    )


    previous_query = (
        st.session_state
        .last_retrieval_query
    )


    with st.chat_message(
        "assistant"
    ):

        status_box = st.status(
            "Miranda is searching the fashion archive...",
            expanded=True
        )


        with status_box:


            # =================================================
            # ROUTING
            # =================================================

            llm_ms = 0.0


            # -------------------------------------------------
            # ROUTE 1: NON-ASCII
            # -------------------------------------------------

            if contains_non_ascii(
                pending_message
            ):

                st.write(
                    "Multilingual query detected — "
                    "normalizing to English..."
                )


                llm_start = (
                    time.perf_counter()
                )


                try:

                    intent = normalize_multilingual_query(
                        pending_message,
                        previous_query,
                        pending_model
                    )

                    llm_ok = True


                except Exception as exc:

                    llm_ok = False

                    llm_error = str(
                        exc
                    )

                    intent = {
                        "domain": "out_of_scope",
                        "is_followup": False,
                        "query_en": "",
                    }


                llm_ms = (
                    time.perf_counter()
                    - llm_start
                ) * 1000


                route = (
                    "multilingual → LLM normalization"
                )


            # -------------------------------------------------
            # ROUTE 2: OBVIOUS ENGLISH FASHION
            # -------------------------------------------------

            elif is_obvious_english_fashion_query(
                pending_message
            ):

                intent = {
                    "domain": "fashion_search",
                    "is_followup": False,
                    "query_en": pending_message.strip(),
                }


                llm_ok = True


                route = (
                    "direct English fashion → CLIP"
                )


                st.write(
                    "Direct fashion query detected — "
                    "skipping LLM."
                )


            # -------------------------------------------------
            # ROUTE 3: AMBIGUOUS ASCII
            # -------------------------------------------------

            else:

                st.write(
                    "Interpreting request..."
                )


                llm_start = (
                    time.perf_counter()
                )


                try:

                    intent = interpret_ascii_query(
                        pending_message,
                        previous_query,
                        pending_model
                    )

                    llm_ok = True


                except Exception as exc:

                    llm_ok = False

                    llm_error = str(
                        exc
                    )

                    intent = {
                        "domain": "out_of_scope",
                        "is_followup": False,
                        "query_en": "",
                    }


                llm_ms = (
                    time.perf_counter()
                    - llm_start
                ) * 1000


                route = (
                    "ASCII → LLM interpretation"
                )


            # =================================================
            # LLM FAILURE
            # =================================================

            if not llm_ok:

                response_text = (
                    "I couldn't interpret that request. "
                    "Please try again."
                )


                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "text": response_text,
                        "catalog_size": pending_catalog_size,
                        "llm_model": pending_model,
                        "route": route,
                        "interpretation": "LLM error",
                    }
                )


                st.session_state.pending_message = None
                st.session_state.pending_model = None
                st.session_state.pending_min_score = None
                st.session_state.pending_index_key = None


                st.rerun()


            # =================================================
            # DOMAIN GATE
            # =================================================

            if (
                intent["domain"]
                != "fashion_search"
            ):

                response_text = (
                    "I can help search the fashion archive, "
                    "but I can't help with that request."
                )


                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "text": response_text,
                        "catalog_size": pending_catalog_size,
                        "llm_model": pending_model,
                        "route": route,
                        "interpretation": "out of scope",
                    }
                )


                st.session_state.pending_message = None
                st.session_state.pending_model = None
                st.session_state.pending_min_score = None
                st.session_state.pending_index_key = None


                st.rerun()


            # =================================================
            # VALID QUERY
            # =================================================

            retrieval_query = (
                intent[
                    "query_en"
                ]
            )


            is_followup = (
                intent[
                    "is_followup"
                ]
            )


            st.write(
                f'Retrieval query: '
                f'"{retrieval_query}"'
            )


            st.write(
                f"Catalog: "
                f"{pending_catalog_size:,} vectors"
            )


            # =================================================
            # CLIP
            # =================================================

            st.write(
                "Encoding fashion intent..."
            )


            tokens = tokenizer(
                [retrieval_query]
            ).to(
                DEVICE
            )


            sync_device()


            clip_start = (
                time.perf_counter()
            )


            with torch.no_grad():

                query_embedding = (
                    clip_model.encode_text(
                        tokens
                    )
                )


                query_embedding = F.normalize(
                    query_embedding,
                    dim=-1
                )


            sync_device()


            clip_ms = (
                time.perf_counter()
                - clip_start
            ) * 1000


            # =================================================
            # GEMM
            # =================================================

            st.write(
                "Searching vector space..."
            )


            sync_device()


            gemm_start = (
                time.perf_counter()
            )


            scores = (
                query_embedding
                @ pending_catalog.T
            )


            sync_device()


            gemm_ms = (
                time.perf_counter()
                - gemm_start
            ) * 1000


            # =================================================
            # TOP-K
            # =================================================

            sync_device()


            topk_start = (
                time.perf_counter()
            )


            candidate_count = min(
                CANDIDATE_TOP_K,
                pending_catalog.shape[0]
            )


            values, indices = torch.topk(
                scores,
                k=candidate_count,
                dim=-1
            )


            sync_device()


            topk_ms = (
                time.perf_counter()
                - topk_start
            ) * 1000


            # =================================================
            # CONFIDENCE GATE
            # =================================================

            st.write(
                "Applying confidence gate..."
            )


            results = []


            for score, idx in zip(
                values[0].cpu(),
                indices[0].cpu()
            ):

                score = float(
                    score
                )

                idx = int(
                    idx
                )


                if (
                    score
                    < pending_min_score
                ):

                    continue


                image_path = (
                    pending_paths[
                        idx
                    ]
                )


                product_name = (
                    Path(
                        image_path
                    )
                    .parent
                    .name
                    .replace(
                        "_",
                        " "
                    )
                )


                results.append(
                    {
                        "path": image_path,
                        "score": score,
                        "name": product_name,
                    }
                )


                if (
                    len(results)
                    >= DISPLAY_TOP_K
                ):

                    break


            # =================================================
            # RESPONSE
            # =================================================

            if results:

                response_text = (
                    f"I found {len(results)} matching "
                    f"design"
                    f"{'s' if len(results) != 1 else ''} "
                    "in the archive."
                )

            else:

                response_text = (
                    "I don't have a sufficiently strong "
                    "match in the current archive."
                )


            timing = {
                "llm_ms": llm_ms,
                "clip_ms": clip_ms,
                "gemm_ms": gemm_ms,
                "topk_ms": topk_ms,
            }


            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "text": response_text,

                    "llm_model": (
                        pending_model
                        if llm_ms > 0
                        else None
                    ),

                    "catalog_size": pending_catalog_size,

                    "retrieval_query": (
                        retrieval_query
                    ),

                    "interpretation": (
                        "follow-up"
                        if is_followup
                        else "new request"
                    ),

                    "route": route,

                    "timing": timing,

                    "results": results,
                }
            )


            st.session_state.last_retrieval_query = (
                retrieval_query
            )


            # =================================================
            # CLEAR TRANSACTION
            # =================================================

            st.session_state.pending_message = None
            st.session_state.pending_model = None
            st.session_state.pending_min_score = None
            st.session_state.pending_index_key = None


            status_box.update(
                label=(
                    f"Search complete — "
                    f"{pending_catalog_size:,} vectors"
                ),
                state="complete",
                expanded=False
            )


    time.sleep(
        0.1
    )

    st.rerun()