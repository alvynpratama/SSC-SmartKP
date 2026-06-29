import os
import re
import json
import uuid
import threading
import datetime
from functools import lru_cache
from difflib import SequenceMatcher, get_close_matches

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


load_dotenv()

# Seluruh path dibuat absolut agar sama dengan app.py dan folder project
# yang terlihat di VS Code.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_store")
DOCUMENT_REGISTRY_FILE = os.path.join(BASE_DIR, "document_registry.json")

# Tetap memakai model lama yang sudah digunakan oleh project ini.
# Tidak ada perpindahan model dan tidak ada kewajiban membuat model_cache baru.
LEGACY_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

ACTIVE_COLLECTION_FILE = os.path.join(
    VECTOR_DB_DIR,
    "active_collection.txt"
)

# Menyimpan seluruh URL yang ditemukan dari PDF aktif.
ACTIVE_DOCUMENT_LINKS_FILE = os.path.join(
    VECTOR_DB_DIR,
    "active_document_links.json"
)

RAG_LOCK = threading.RLock()


class EmbeddingUnavailableError(RuntimeError):
    """Model embedding tidak dapat dimuat saat ini."""

# Riwayat percakapan dipisahkan per tab apabila conversation_id dikirim oleh app.py.
# Untuk saat ini tetap kompatibel dengan pemanggilan lama tanpa conversation_id.
CHAT_HISTORIES = {}
MAX_HISTORY_MESSAGES = 6

URL_PATTERN = re.compile(
    r"""(?i)\b(?:https?://|www\.)[^\s<>()\[\]{}"']+"""
)

LINK_TERMS = [
    "link", "linknya", "tautan", "url", "website", "situs",
    "web", "alamat", "portal", "form", "formulir",
]

GENERIC_LINK_WORDS = {
    "link", "linknya", "tautan", "url", "website", "situs", "web",
    "alamat", "portal", "form", "formulir", "saya", "aku", "ingin",
    "mau", "minta", "tolong", "boleh", "ada", "apakah", "kah",
    "dong", "ya", "yang", "itu", "ini", "untuk", "ke", "buka",
    "akses", "download", "unduh",
}

STOPWORDS = {
    "yang", "dan", "atau", "untuk", "dengan", "tentang", "dari",
    "saya", "aku", "kamu", "anda", "kami", "minta", "ingin",
    "apakah", "adakah", "mohon", "bisa", "dapat", "tolong",
    "link", "linknya", "tautan", "url", "website", "situs", "web",
    "alamat", "portal", "form", "formulir", "bagaimana", "saja",
    "pada", "dalam", "seperti", "lebih", "juga", "itu", "ini",
    "ke", "untuknya", "nya",
}


# Fokus layanan utama SSC pada versi ini.
SSC_FOCUS_CATEGORIES = (
    "Kerja Praktik",
    "Tugas Akhir",
)

SSC_TOPIC_CONFIG = {
    "Kerja Praktik": {
        "phrases": [
            "kerja praktik",
            "kerja praktek",
            "praktik kerja",
            "praktek kerja",
            "program magang",
            "kerja magang",
        ],
        "tokens": [
            "kp",
            "kpp",
            "magang",
            "internship",
            "praktik",
            "praktek",
        ],
    },
    "Tugas Akhir": {
        "phrases": [
            "tugas akhir",
            "tugas ahir",
            "proposal tugas akhir",
            "seminar proposal",
            "seminar hasil",
            "bimbingan tugas akhir",
            "sidang tugas akhir",
        ],
        "tokens": [
            "ta",
            "taa",
            "skripsi",
            "skripsih",
            "sempro",
            "semhas",
            "sidang",
        ],
    },
}

# =========================================================
# UTILITAS DASAR
# =========================================================
@lru_cache(maxsize=1)
def get_embeddings():
    """
    Memakai model lama all-MiniLM-L6-v2 dan menyimpannya di memori selama
    Flask masih berjalan. Hal ini mencegah pemuatan ulang pada setiap chat.

    Urutan percobaan:
    1. Mode normal seperti versi lama.
    2. Bila jaringan Hugging Face sedang bermasalah, coba cache model lokal.
    """
    try:
        return HuggingFaceEmbeddings(
            model_name=LEGACY_EMBEDDING_MODEL
        )
    except Exception as online_error:
        try:
            return HuggingFaceEmbeddings(
                model_name=LEGACY_EMBEDDING_MODEL,
                model_kwargs={"local_files_only": True},
            )
        except Exception as local_error:
            raise EmbeddingUnavailableError(
                "Model embedding lama belum dapat dimuat dari koneksi maupun cache lokal."
            ) from local_error


def normalize_text(text):
    """Menormalkan teks untuk pencocokan kata dan typo ringan."""
    normalized = str(text or "").lower().strip()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def get_words(text):
    """Mengambil kata-kata penting dari sebuah teks."""
    return re.findall(r"[a-z0-9]{2,}", normalize_text(text))


def normalize_url(url):
    """Merapikan URL agar tidak memiliki tanda baca penutup."""
    cleaned = re.sub(r"\s+", "", str(url or "").strip())
    cleaned = cleaned.rstrip(".,;:!?)]}>\"'")

    if cleaned.lower().startswith("www."):
        cleaned = f"https://{cleaned}"

    return cleaned


def clean_response_text(response):
    """
    Membersihkan Markdown agar jawaban chatbot tetap rapi di UI.
    Tidak ada **, *, heading #, bullet Markdown, atau backtick.
    """
    if not response:
        return ""

    cleaned = str(response).strip()

    cleaned = re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
        r"\1: \2",
        cleaned,
    )

    cleaned = re.sub(r"(\*{1,3}|_{1,3})(.*?)\1", r"\2", cleaned)
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"(?m)^\s*#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[*•\-]\s+", "", cleaned)
    cleaned = cleaned.replace("*", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    return cleaned.strip()


# =========================================================
# METADATA DOKUMEN SSC
# =========================================================
def infer_category_from_filename(filename):
    """
    Fallback kategori bila metadata belum tersedia.
    app.py tetap menjadi sumber utama metadata kategori.
    """
    normalized_name = str(filename or "").lower()

    if any(keyword in normalized_name for keyword in [
        "kerja praktik", "kerja_praktik", "pedoman kp", "pedoman_kp",
        "_kp", "kp_",
    ]):
        return "Kerja Praktik"

    if any(keyword in normalized_name for keyword in [
        "tugas akhir", "tugas_akhir", "skripsi", "thesis", "ta_", "_ta",
    ]):
        return "Tugas Akhir"

    if "beasiswa" in normalized_name:
        return "Beasiswa"

    if any(keyword in normalized_name for keyword in [
        "akademik", "perkuliahan", "kurikulum",
    ]):
        return "Akademik"

    if any(keyword in normalized_name for keyword in [
        "administrasi", "surat", "layanan",
    ]):
        return "Administrasi"

    return "Lainnya"


def get_document_registry():
    """Membaca metadata PDF yang dibuat oleh app.py."""
    if not os.path.exists(DOCUMENT_REGISTRY_FILE):
        return {}

    try:
        with open(DOCUMENT_REGISTRY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            return {}

        return {
            str(item.get("filename")): item
            for item in data
            if isinstance(item, dict) and item.get("filename")
        }

    except (OSError, json.JSONDecodeError):
        return {}


def get_document_profile(filename, registry):
    """
    Menghasilkan profil universal untuk setiap PDF.
    Profil ini ditanam ke setiap chunk supaya retrieval lebih memahami topik.
    """
    metadata = registry.get(filename, {})

    raw_title = str(metadata.get("title") or "").strip()
    fallback_title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")

    title = raw_title or fallback_title or "Dokumen SSC"
    category = str(metadata.get("category") or "").strip()
    category = category or infer_category_from_filename(filename)

    return {
        "title": title,
        "category": category,
        "filename": filename,
    }


# =========================================================
# DETEKSI LAYANAN SSC: KERJA PRAKTIK DAN TUGAS AKHIR
# =========================================================
def compact_text(text):
    """
    Menormalkan teks lebih jauh agar typo ringan tetap mudah dikenali.
    Contoh:
    - kerjaa praktekk -> kerjaa praktekk
    - tugass akhhir -> tugass akhhir
    """
    compacted = normalize_text(text)
    compacted = re.sub(r"(.)\1{2,}", r"\1\1", compacted)
    return compacted


def compact_token(token):
    """Menormalkan satu token untuk pencocokan typo."""
    return compact_text(token).replace(" ", "")


def token_similarity(first_token, second_token):
    """Mengukur kemiripan dua kata untuk menangani typo dan transposisi."""
    first = compact_token(first_token)
    second = compact_token(second_token)

    if not first or not second:
        return 0.0

    if first == second:
        return 1.0

    return SequenceMatcher(None, first, second).ratio()


def phrase_similarity(first_phrase, second_phrase):
    """Mengukur kemiripan dua frasa dengan toleransi typo."""
    first = compact_text(first_phrase)
    second = compact_text(second_phrase)

    if not first or not second:
        return 0.0

    if first == second:
        return 1.0

    return SequenceMatcher(None, first, second).ratio()


def contains_fuzzy_phrase(question, phrase, cutoff=0.74):
    """
    Mendeteksi frasa meskipun ada salah ketik.
    Contoh yang tetap dikenali:
    - kerja praktekk
    - kerja pratik
    - tugass akir
    - semnar proposal
    """
    normalized_question = compact_text(question)
    normalized_phrase = compact_text(phrase)

    if not normalized_question or not normalized_phrase:
        return False

    if normalized_phrase in normalized_question:
        return True

    question_words = normalized_question.split()
    phrase_words = normalized_phrase.split()
    phrase_length = len(phrase_words)

    if not phrase_words or len(question_words) < phrase_length:
        return False

    for start_index in range(
        len(question_words) - phrase_length + 1
    ):
        candidate = " ".join(
            question_words[start_index:start_index + phrase_length]
        )

        if phrase_similarity(candidate, normalized_phrase) >= cutoff:
            return True

        token_scores = [
            token_similarity(candidate_word, phrase_word)
            for candidate_word, phrase_word in zip(
                candidate.split(),
                phrase_words,
            )
        ]

        if token_scores and (
            sum(token_scores) / len(token_scores)
        ) >= 0.78:
            return True

    return False


def has_fuzzy_token(question_words, expected_token):
    """
    Mendeteksi token penting dengan toleransi typo.
    Singkatan pendek seperti KP dan TA tetap menggunakan pencocokan ketat
    agar tidak mudah salah mengira kata lain sebagai layanan.
    """
    expected = compact_token(expected_token)

    for word in question_words:
        candidate = compact_token(word)

        if candidate == expected:
            return True

        if len(expected) <= 2:
            continue

        cutoff = 0.82 if len(expected) >= 6 else 0.78

        if token_similarity(candidate, expected) >= cutoff:
            return True

    return False


def score_focus_category(question, category):
    """
    Memberi skor untuk Kerja Praktik atau Tugas Akhir.
    Skor tinggi berarti topik disebut secara jelas atau melalui typo ringan.
    """
    config = SSC_TOPIC_CONFIG[category]
    question_words = compact_text(question).split()
    normalized_question = compact_text(question)
    score = 0

    for phrase in config["phrases"]:
        normalized_phrase = compact_text(phrase)

        if normalized_phrase in normalized_question:
            score += 8
        elif contains_fuzzy_phrase(question, phrase):
            score += 6

    for token in config["tokens"]:
        expected = compact_token(token)

        if expected in {"kp", "kpp", "ta", "taa"}:
            if expected in question_words:
                score += 5
            continue

        if has_fuzzy_token(question_words, token):
            score += 3

    # Petunjuk tambahan yang hanya berlaku jika konteks TA sudah kuat.
    if category == "Tugas Akhir":
        ta_context_words = {
            "proposal",
            "bimbingan",
            "sempro",
            "semhas",
            "sidang",
            "skripsi",
        }

        if any(word in question_words for word in ta_context_words):
            score += 2

    # Petunjuk tambahan untuk KP.
    if category == "Kerja Praktik":
        kp_context_words = {
            "pengajuan",
            "magang",
            "praktik",
            "praktek",
            "instansi",
        }

        if any(word in question_words for word in kp_context_words):
            score += 1

    return score


def detect_requested_focus_category(question):
    """
    Menentukan apakah pertanyaan membahas Kerja Praktik atau Tugas Akhir.

    Fungsi ini memahami typo umum seperti:
    - kerja praktekk
    - kerrja pratik
    - tugass akir
    - skripssi
    - semnar proposal
    """
    if not str(question or "").strip():
        return None

    scores = {
        category: score_focus_category(question, category)
        for category in SSC_FOCUS_CATEGORIES
    }

    highest_category = max(scores, key=scores.get)
    highest_score = scores[highest_category]

    if highest_score < 3:
        return None

    # Jika skor hampir seimbang, jangan mengunci topik secara salah.
    other_scores = [
        score
        for category, score in scores.items()
        if category != highest_category
    ]

    if other_scores and highest_score - max(other_scores) < 2:
        return None

    return highest_category


def category_matches_focus(category_value, focus_category):
    """Mencocokkan label kategori registry terhadap fokus KP atau TA."""
    normalized_category = compact_text(category_value)

    if focus_category == "Kerja Praktik":
        return (
            "kerja praktik" in normalized_category
            or "kerja praktek" in normalized_category
            or normalized_category == "kp"
        )

    if focus_category == "Tugas Akhir":
        return (
            "tugas akhir" in normalized_category
            or normalized_category == "ta"
            or "skripsi" in normalized_category
        )

    return False


def get_active_document_profiles():
    """
    Mengambil profil PDF yang benar-benar masih berada di folder uploads.
    Jadi registry yang tersisa setelah file dihapus tidak akan dianggap aktif.
    """
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    registry = get_document_registry()

    pdf_files = sorted([
        file_name
        for file_name in os.listdir(UPLOAD_FOLDER)
        if file_name.lower().endswith(".pdf")
        and os.path.isfile(os.path.join(UPLOAD_FOLDER, file_name))
    ])

    return [
        get_document_profile(file_name, registry)
        for file_name in pdf_files
    ]


def get_available_focus_categories():
    """
    Mengembalikan layanan fokus yang benar-benar tersedia:
    Kerja Praktik dan/atau Tugas Akhir.

    Selain membaca kategori yang dipilih admin, sistem juga memeriksa judul
    dan nama file. Dengan begitu, pedoman TA tetap dikenali walaupun metadata
    lama belum sepenuhnya lengkap.
    """
    available_categories = set()

    for profile in get_active_document_profiles():
        profile_text = " ".join([
            str(profile.get("category", "")),
            str(profile.get("title", "")),
            str(profile.get("filename", "")),
        ])

        for focus_category in SSC_FOCUS_CATEGORIES:
            if (
                category_matches_focus(
                    profile.get("category", ""),
                    focus_category,
                )
                or detect_requested_focus_category(profile_text) == focus_category
            ):
                available_categories.add(focus_category)

    return available_categories


def get_other_available_focus_category(missing_category):
    """Mengambil layanan lain yang masih tersedia untuk ditawarkan ke mahasiswa."""
    other_categories = sorted(
        get_available_focus_categories() - {missing_category}
    )

    return other_categories[0] if other_categories else None


def build_missing_category_response(category):
    """
    Respons saat mahasiswa sudah menyebut layanan tertentu, tetapi informasi
    layanan tersebut memang belum tersedia. Tidak menawarkan atau menyebut
    layanan lain secara otomatis.
    """
    return (
        f"Maaf ya, informasi terkait {category} saat ini belum siap tersedia di SSC.\n\n"
        f"Agar informasi yang saya berikan tetap tepat, saya belum bisa menjelaskan detail {category} terlebih dahulu. "
        "Silakan coba tanyakan lagi setelah layanan ini diperbarui ya."
    )


def get_missing_focus_category_response(question):
    """
    Mengembalikan respons khusus jika user menanyakan KP/TA yang belum aktif.
    Jika layanan yang ditanya tersedia, hasilnya None agar proses RAG lanjut.
    """
    requested_category = detect_requested_focus_category(question)

    if not requested_category:
        return None

    if requested_category not in get_available_focus_categories():
        return build_missing_category_response(requested_category)

    return None


def document_matches_focus_category(document, focus_category):
    """
    Memastikan chunk retrieval berasal dari layanan yang sedang ditanyakan.
    Ini mencegah jawaban TA mengambil konteks KP, atau sebaliknya.
    """
    metadata = document.metadata or {}

    document_text = " ".join([
        str(metadata.get("ssc_category", "")),
        str(metadata.get("ssc_title", "")),
        str(metadata.get("source_name", "")),
        str(metadata.get("source", "")),
    ])

    if category_matches_focus(
        metadata.get("ssc_category", ""),
        focus_category,
    ):
        return True

    return detect_requested_focus_category(document_text) == focus_category


# =========================================================
# COLLECTION CHROMA
# =========================================================
def get_active_collection_name():
    """Membaca nama collection Chroma aktif."""
    if not os.path.exists(ACTIVE_COLLECTION_FILE):
        return None

    try:
        with open(ACTIVE_COLLECTION_FILE, "r", encoding="utf-8") as file:
            collection_name = file.read().strip()

        return collection_name if collection_name else None

    except OSError:
        return None


def set_active_collection_name(collection_name):
    """Menyimpan collection baru sebagai memori SSC aktif."""
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    temp_file = f"{ACTIVE_COLLECTION_FILE}.tmp"

    with open(temp_file, "w", encoding="utf-8") as file:
        file.write(collection_name)

    os.replace(temp_file, ACTIVE_COLLECTION_FILE)


def clear_active_collection_name():
    """Menghapus penanda collection aktif jika tidak ada PDF."""
    if os.path.exists(ACTIVE_COLLECTION_FILE):
        os.remove(ACTIVE_COLLECTION_FILE)


def open_vector_store(embeddings, collection_name):
    """Membuka collection Chroma tertentu."""
    return Chroma(
        collection_name=collection_name,
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
    )


def collection_has_documents(embeddings, collection_name):
    """Memeriksa apakah collection aktif berisi dokumen."""
    if not collection_name:
        return False

    try:
        vector_store = open_vector_store(embeddings, collection_name)
        return vector_store._collection.count() > 0

    except Exception:
        return False


# =========================================================
# KATALOG LINK DARI PDF
# =========================================================
def get_active_document_links():
    """Membaca katalog URL yang diekstrak dari seluruh PDF aktif."""
    if not os.path.exists(ACTIVE_DOCUMENT_LINKS_FILE):
        return []

    try:
        with open(
            ACTIVE_DOCUMENT_LINKS_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        links = data.get("links", [])
        return links if isinstance(links, list) else []

    except (OSError, json.JSONDecodeError):
        return []


def set_active_document_links(links):
    """Menyimpan katalog URL PDF setelah indexing selesai."""
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    temp_file = f"{ACTIVE_DOCUMENT_LINKS_FILE}.tmp"

    payload = {
        "updated_at": datetime.datetime.now().isoformat(),
        "links": links,
    }

    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    os.replace(temp_file, ACTIVE_DOCUMENT_LINKS_FILE)


def clear_active_document_links():
    """Menghapus katalog link saat tidak ada PDF aktif."""
    if os.path.exists(ACTIVE_DOCUMENT_LINKS_FILE):
        os.remove(ACTIVE_DOCUMENT_LINKS_FILE)


def extract_urls_from_text(text):
    """Mengambil URL yang tertulis sebagai teks di dalam PDF."""
    urls = []
    seen = set()

    for raw_url in URL_PATTERN.findall(text or ""):
        url = normalize_url(raw_url)

        if url and url not in seen:
            urls.append(url)
            seen.add(url)

    return urls


def extract_annotation_links(pdf_path):
    """
    Mengambil hyperlink tersembunyi sebagai anotasi PDF.
    Contoh: URL di balik tombol, gambar, atau teks hyperlink.
    """
    if PdfReader is None:
        return {}

    page_links = {}

    try:
        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(reader.pages, start=1):
            urls = []
            annotations = page.get("/Annots", [])

            for annotation_ref in annotations:
                try:
                    annotation = annotation_ref.get_object()
                    action = annotation.get("/A")

                    if action:
                        try:
                            action = action.get_object()
                        except AttributeError:
                            pass

                        uri = action.get("/URI")

                        if uri:
                            normalized = normalize_url(uri)

                            if normalized and normalized not in urls:
                                urls.append(normalized)

                except Exception:
                    # Satu anotasi rusak tidak boleh membatalkan indexing PDF.
                    continue

            if urls:
                page_links[page_number] = urls

    except Exception:
        return {}

    return page_links


def make_link_record(url, profile, page_number, context_text):
    """Membuat satu record URL untuk katalog link PDF."""
    return {
        "url": normalize_url(url),
        "source": profile["filename"],
        "title": profile["title"],
        "category": profile["category"],
        "page": int(page_number),
        "context": re.sub(r"\s+", " ", context_text or "").strip()[:700],
    }


# =========================================================
# BUILD MEMORI SSC UNIVERSAL
# =========================================================
def build_vector_store():
    """
    Membaca seluruh PDF dari folder uploads dan membangun satu memori SSC.

    Setiap chunk diberi metadata:
    - Judul dokumen
    - Kategori dokumen
    - Nama file
    - Nomor halaman

    Dengan cara ini, satu SSC Smart Assistant dapat menangani dokumen
    Kerja Praktik, Tugas Akhir, Akademik, Beasiswa, dan topik lain tanpa
    mengunci jawaban pada satu jenis layanan.
    """
    with RAG_LOCK:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        pdf_files = sorted([
            file_name
            for file_name in os.listdir(UPLOAD_FOLDER)
            if file_name.lower().endswith(".pdf")
            and os.path.isfile(os.path.join(UPLOAD_FOLDER, file_name))
        ])

        if not pdf_files:
            clear_active_collection_name()
            clear_active_document_links()
            return None

        registry = get_document_registry()
        all_documents = []
        document_link_records = []
        seen_link_records = set()

        for file_name in pdf_files:
            file_path = os.path.join(UPLOAD_FOLDER, file_name)
            profile = get_document_profile(file_name, registry)

            annotation_links_by_page = extract_annotation_links(file_path)

            loader = PyPDFLoader(file_path)
            documents = loader.load()

            for document in documents:
                page_number = int(document.metadata.get("page", 0)) + 1
                page_text = document.page_content or ""

                # Metadata universal ikut disimpan di Chroma.
                document.metadata["source_name"] = file_name
                document.metadata["ssc_title"] = profile["title"]
                document.metadata["ssc_category"] = profile["category"]
                document.metadata["page_number"] = page_number

                # Header disisipkan agar model memahami sumber topiknya.
                document.page_content = (
                    "Informasi Dokumen SSC\n"
                    f"Judul: {profile['title']}\n"
                    f"Kategori: {profile['category']}\n"
                    f"Nama file: {file_name}\n"
                    f"Halaman: {page_number}\n\n"
                    f"{page_text}"
                )

                page_urls = extract_urls_from_text(page_text)
                page_urls.extend(
                    annotation_links_by_page.get(page_number, [])
                )

                for url in page_urls:
                    normalized_url = normalize_url(url)

                    record_key = (
                        normalized_url,
                        file_name,
                        page_number,
                    )

                    if normalized_url and record_key not in seen_link_records:
                        document_link_records.append(
                            make_link_record(
                                normalized_url,
                                profile,
                                page_number,
                                page_text,
                            )
                        )
                        seen_link_records.add(record_key)

            all_documents.extend(documents)

        if not all_documents:
            raise RuntimeError(
                "Tidak ada teks yang berhasil dibaca dari file PDF."
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=180,
        )

        chunks = text_splitter.split_documents(all_documents)

        if not chunks:
            raise RuntimeError(
                "Dokumen PDF tidak memiliki isi teks yang dapat diproses."
            )

        embeddings = get_embeddings()

        collection_name = (
            f"ssc_universal_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_"
            f"{uuid.uuid4().hex[:8]}"
        )

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=VECTOR_DB_DIR,
        )

        # Penanda baru hanya dibuat setelah indexing berhasil.
        set_active_collection_name(collection_name)
        set_active_document_links(document_link_records)

        return vector_store


# =========================================================
# RIWAYAT PERCAKAPAN
# =========================================================
def normalize_conversation_id(conversation_id=None):
    """Membuat key aman untuk memisahkan history antar tab chat."""
    value = str(conversation_id or "").strip()

    if not value:
        return "default"

    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)[:120] or "default"


def get_chat_history(conversation_id=None):
    """Mengambil history sesuai tab chat."""
    conversation_key = normalize_conversation_id(conversation_id)

    if conversation_key not in CHAT_HISTORIES:
        CHAT_HISTORIES[conversation_key] = []

    return CHAT_HISTORIES[conversation_key]


def save_chat_history(user_question, response, conversation_id=None):
    """Menyimpan maksimal tiga pasang percakapan untuk satu tab."""
    history = get_chat_history(conversation_id)

    history.append(HumanMessage(content=user_question))
    history.append(AIMessage(content=response))

    if len(history) > MAX_HISTORY_MESSAGES:
        del history[:-MAX_HISTORY_MESSAGES]


def get_previous_user_question(conversation_id=None):
    """Mengambil pertanyaan pengguna sebelumnya untuk follow-up seperti 'linknya?'."""
    history = get_chat_history(conversation_id)

    for message in reversed(history):
        if isinstance(message, HumanMessage):
            content = str(message.content or "").strip()

            if content:
                return content

    return ""


# =========================================================
# RESPONS NETRAL DAN KONTEKS PERCAKAPAN
# =========================================================
def is_neutral_greeting_or_help_request(question):
    """
    Mengenali sapaan atau permintaan bantuan yang masih umum.
    Respons untuk kondisi ini tidak boleh langsung menyebut topik tertentu.
    """
    normalized = normalize_text(question)

    if not normalized:
        return False

    neutral_phrases = {
        "halo",
        "hai",
        "hi",
        "selamat pagi",
        "selamat siang",
        "selamat sore",
        "selamat malam",
        "pagi",
        "siang",
        "sore",
        "malam",
        "butuh bantuan",
        "saya butuh bantuan",
        "mau tanya",
        "ingin bertanya",
        "apa yang bisa dibantu",
        "bisa bantu saya",
    }

    if normalized in neutral_phrases:
        return True

    greeting_starters = (
        "halo ",
        "hai ",
        "hi ",
        "selamat pagi",
        "selamat siang",
        "selamat sore",
        "selamat malam",
    )

    return (
        normalized.startswith(greeting_starters)
        and len(normalized.split()) <= 7
        and detect_requested_focus_category(normalized) is None
    )


def build_neutral_welcome_response():
    """Sapaan pembuka universal tanpa mengarahkan pengguna ke topik tertentu."""
    return "Halo! Ada yang bisa saya bantu hari ini?"


def get_previous_focus_category(conversation_id=None):
    """
    Mengambil kategori yang disebut secara eksplisit pada pesan pengguna
    sebelumnya. Berguna untuk pertanyaan lanjutan seperti 'syaratnya apa?'.
    """
    history = get_chat_history(conversation_id)

    for message in reversed(history):
        if not isinstance(message, HumanMessage):
            continue

        detected_category = detect_requested_focus_category(
            str(message.content or "")
        )

        if detected_category:
            return detected_category

    return None


def get_previous_focus_question(conversation_id=None):
    """Mengambil pertanyaan sebelumnya yang sudah menyebut topik secara jelas."""
    history = get_chat_history(conversation_id)

    for message in reversed(history):
        if not isinstance(message, HumanMessage):
            continue

        content = str(message.content or "").strip()

        if content and detect_requested_focus_category(content):
            return content

    return ""


# =========================================================
# PEMAHAMAN PESAN DAN KONTEKS PERCAKAPAN
# =========================================================
def is_neutral_greeting_or_help_request(question):
    """
    Sapaan atau permintaan bantuan umum. Responsnya tidak boleh langsung
    menyebut topik layanan tertentu.
    """
    normalized = normalize_text(question)

    if not normalized:
        return False

    simple_messages = {
        "halo", "hai", "hi", "pagi", "siang", "sore", "malam",
        "selamat pagi", "selamat siang", "selamat sore", "selamat malam",
        "butuh bantuan", "saya butuh bantuan", "mau tanya",
        "ingin bertanya", "bisa bantu saya", "bisa bantu",
        "apa yang bisa dibantu", "minta bantuan",
    }

    if normalized in simple_messages:
        return True

    greeting_starters = (
        "halo ", "hai ", "hi ", "selamat pagi",
        "selamat siang", "selamat sore", "selamat malam",
    )

    return (
        normalized.startswith(greeting_starters)
        and len(normalized.split()) <= 8
        and detect_requested_focus_category(normalized) is None
    )


def is_gratitude_message(question):
    """Mengenali ucapan terima kasih agar jawaban terasa manusiawi."""
    normalized = normalize_text(question)

    return normalized in {
        "terima kasih", "makasih", "makasi", "thanks", "thank you",
        "oke makasih", "ok makasih", "sip makasih", "terimakasih",
    }


def build_neutral_welcome_response():
    return "Halo! Ada yang bisa saya bantu hari ini?"


def build_gratitude_response():
    return (
        "Sama-sama. Semoga membantu ya. "
        "Kalau ada hal lain yang ingin ditanyakan, silakan tulis saja."
    )


def get_previous_focus_category(conversation_id=None):
    """
    Mengambil kategori dari pesan pengguna sebelumnya untuk memahami
    follow-up seperti "syaratnya apa?" atau "jelaskan lagi".
    """
    history = get_chat_history(conversation_id)

    for message in reversed(history):
        if not isinstance(message, HumanMessage):
            continue

        category = detect_requested_focus_category(
            str(message.content or "")
        )

        if category:
            return category

    return None


def get_previous_focus_question(conversation_id=None):
    """Mengambil pertanyaan sebelumnya yang memiliki topik jelas."""
    history = get_chat_history(conversation_id)

    for message in reversed(history):
        if not isinstance(message, HumanMessage):
            continue

        text = str(message.content or "").strip()

        if text and detect_requested_focus_category(text):
            return text

    return ""


def resolve_context_category(question, conversation_id=None):
    """
    Memakai kategori yang disebut pada pesan sekarang. Bila pesan merupakan
    follow-up singkat, sistem memakai konteks kategori percakapan sebelumnya.
    """
    explicit_category = detect_requested_focus_category(question)

    if explicit_category:
        return explicit_category

    if is_neutral_greeting_or_help_request(question) or is_gratitude_message(question):
        return None

    return get_previous_focus_category(conversation_id)


def is_topic_introduction_message(question, category):
    """
    Mendeteksi pengguna yang baru menyebut topik tetapi belum mengajukan
    pertanyaan spesifik, misalnya:
    - saya ingin bertanya terkait tugas akhir
    - mau tanya KP
    """
    if not category:
        return False

    normalized = normalize_text(question)

    fillers = {
        "saya", "aku", "ingin", "mau", "minta", "tolong", "boleh",
        "apakah", "bisa", "dapat", "bertanya", "tanya", "terkait",
        "tentang", "seputar", "mengenai", "info", "informasi",
        "dong", "ya", "nih", "kak",
        "kerja", "praktik", "praktek", "kp", "kpp", "magang",
        "tugas", "akhir", "ahir", "ta", "taa", "skripsi",
    }

    meaningful_words = [
        word
        for word in get_words(normalized)
        if word not in fillers
    ]

    return len(meaningful_words) == 0


def build_topic_introduction_response(category):
    """
    Respons natural setelah pengguna menyebut topik, tetapi belum menyampaikan
    kebutuhan atau pertanyaan detail.
    """
    return (
        f"Boleh, saya siap membantu terkait {category}. "
        "Bagian apa yang ingin kamu tanyakan?"
    )


def detect_message_intent(question):
    """Klasifikasi ringan untuk mengarahkan alur percakapan."""
    if is_neutral_greeting_or_help_request(question):
        return "sapaan"

    if is_gratitude_message(question):
        return "terima_kasih"

    if has_link_request(question):
        return "permintaan_tautan"

    if detect_requested_focus_category(question):
        return "pertanyaan_topik"

    return "pertanyaan_umum"


# =========================================================
# RETRIEVAL DAN LINK UNIVERSAL
# =========================================================
def format_docs(docs):
    """Menggabungkan chunk dokumen menjadi konteks untuk LLM."""
    if not docs:
        return ""

    return "\n\n".join(
        document.page_content
        for document in docs
    )


def retrieve_relevant_documents(
    vector_store,
    query,
    limit=5,
    preferred_category=None,
):
    """
    Mengambil konteks relevan dan mengurangi chunk yang terlalu berulang
    dari halaman maupun file yang sama.
    """
    search_limit = max(limit * 6, 18)

    try:
        results = vector_store.similarity_search_with_relevance_scores(
            query,
            k=search_limit,
        )

        candidates = [
            (document, score)
            for document, score in results
            if score >= 0.10
        ]
    except Exception:
        try:
            candidates = [
                (document, 1.0)
                for document in vector_store.similarity_search(
                    query,
                    k=search_limit,
                )
            ]
        except Exception:
            candidates = []

    if preferred_category:
        candidates = [
            (document, score)
            for document, score in candidates
            if document_matches_focus_category(
                document,
                preferred_category,
            )
        ]

    selected = []
    seen_pages = set()
    per_source_count = {}

    for document, _score in candidates:
        metadata = document.metadata or {}
        source = (
            metadata.get("source_name")
            or os.path.basename(str(metadata.get("source", "")))
        )
        page = metadata.get(
            "page_number",
            metadata.get("page", 0),
        )
        page_key = (source, page)

        if page_key in seen_pages:
            continue

        if per_source_count.get(source, 0) >= 3:
            continue

        selected.append(document)
        seen_pages.add(page_key)
        per_source_count[source] = per_source_count.get(source, 0) + 1

        if len(selected) >= limit:
            break

    return selected


def has_link_request(question):
    """
    Mendeteksi permintaan link dengan typo ringan seperti:
    linlk, lnik, tautna, weebsite.
    """
    normalized = normalize_text(question)
    words = get_words(question)

    for term in LINK_TERMS:
        if term in normalized:
            return True

        match = get_close_matches(
            term,
            words,
            n=1,
            cutoff=0.78,
        )

        if match:
            return True

    return False


def get_meaningful_words(question):
    """Mengambil kata topik yang bisa dipakai mencari link relevan."""
    return [
        word
        for word in get_words(question)
        if word not in STOPWORDS
        and word not in GENERIC_LINK_WORDS
    ]


def is_generic_link_request(question):
    """
    True untuk pertanyaan seperti 'saya ingin link' tanpa menyebut dokumen,
    layanan, atau topik apa pun.
    """
    if not has_link_request(question):
        return False

    return len(get_meaningful_words(question)) == 0


def get_link_query(question, conversation_id=None):
    """
    Memahami follow-up seperti "linknya?" menggunakan topik yang telah
    disebut sebelumnya dalam tab percakapan yang sama.
    """
    if not is_generic_link_request(question):
        return question

    previous_question = get_previous_user_question(conversation_id)

    if previous_question and detect_requested_focus_category(previous_question):
        return previous_question

    previous_focus_question = get_previous_focus_question(conversation_id)

    if previous_focus_question:
        return previous_focus_question

    return previous_question or ""


def get_retrieved_document_pages(retrieved_docs):
    """Mengambil kombinasi file dan halaman dari chunk retrieval."""
    pages = set()

    for document in retrieved_docs or []:
        source = (
            document.metadata.get("source_name")
            or os.path.basename(str(document.metadata.get("source", "")))
        )
        page = int(
            document.metadata.get(
                "page_number",
                int(document.metadata.get("page", 0)) + 1,
            )
        )

        if source:
            pages.add((source, page))

    return pages


def get_urls_visible_in_retrieved_docs(retrieved_docs):
    """Mengambil URL yang tampak dalam chunk PDF yang relevan."""
    results = []
    seen = set()

    for document in retrieved_docs or []:
        source = (
            document.metadata.get("source_name")
            or os.path.basename(str(document.metadata.get("source", "Dokumen SSC")))
        )
        title = document.metadata.get("ssc_title", source)
        category = document.metadata.get("ssc_category", "Lainnya")
        page = int(
            document.metadata.get(
                "page_number",
                int(document.metadata.get("page", 0)) + 1,
            )
        )

        for url in extract_urls_from_text(document.page_content):
            normalized_url = normalize_url(url)

            if normalized_url and normalized_url not in seen:
                results.append({
                    "url": normalized_url,
                    "source": source,
                    "title": title,
                    "category": category,
                    "page": page,
                    "score": 20,
                })
                seen.add(normalized_url)

    return results


def score_document_link(record, keywords, retrieved_pages, requested_category=None):
    """Memberi skor relevansi untuk URL katalog PDF."""
    source = str(record.get("source", "")).lower()
    title = str(record.get("title", "")).lower()
    category = str(record.get("category", "")).lower()
    context = str(record.get("context", "")).lower()
    page = int(record.get("page", 0))

    score = 0

    if (record.get("source"), page) in retrieved_pages:
        score += 18

    # Saat pengguna sudah menyebut topik, tautan dari kategori yang sama
    # diprioritaskan meskipun ia hanya mengetik singkatan seperti "link KP".
    if requested_category and category_matches_focus(
        record.get("category", ""),
        requested_category,
    ):
        score += 20

    for keyword in keywords:
        score += context.count(keyword) * 4
        score += title.count(keyword) * 4
        score += category.count(keyword) * 3
        score += source.count(keyword) * 2

    return score


def get_relevant_document_links(question, retrieved_docs, requested_category=None):
    """
    Mengambil maksimal tiga URL yang relevan. Bila topik sudah diketahui,
    tautan dari topik lain tidak akan ikut dipilih.
    """
    catalog = get_active_document_links()
    keywords = get_meaningful_words(question)
    retrieved_pages = get_retrieved_document_pages(retrieved_docs)
    requested_category = requested_category or detect_requested_focus_category(question)

    candidates = []

    for item in get_urls_visible_in_retrieved_docs(retrieved_docs):
        if requested_category and not category_matches_focus(
            item.get("category", ""),
            requested_category,
        ):
            continue

        candidates.append(item)

    for record in catalog:
        url = normalize_url(record.get("url", ""))

        if not url:
            continue

        if requested_category and not category_matches_focus(
            record.get("category", ""),
            requested_category,
        ):
            continue

        score = score_document_link(
            record,
            keywords,
            retrieved_pages,
            requested_category=requested_category,
        )

        if score >= 18 or (keywords and score >= 5):
            candidate = dict(record)
            candidate["url"] = url
            candidate["score"] = score
            candidates.append(candidate)

    candidates.sort(
        key=lambda item: item.get("score", 0),
        reverse=True,
    )

    selected = []
    seen_urls = set()

    for candidate in candidates:
        url = normalize_url(candidate.get("url", ""))

        if not url or url in seen_urls:
            continue

        selected.append(url)
        seen_urls.add(url)

        if len(selected) >= 3:
            break

    return selected


def build_link_clarification_response():
    """Respons netral saat pengguna meminta tautan tanpa konteks."""
    return (
        "Tentu. Tautan untuk kebutuhan yang mana, ya?\n\n"
        "Ceritakan sedikit topik atau proses yang kamu maksud agar saya bisa mencarikannya dengan lebih tepat."
    )


def build_document_link_response(question, vector_store, requested_category=None):
    """
    Memberikan tautan yang benar-benar ditemukan pada informasi aktif.
    """
    requested_category = (
        requested_category
        or detect_requested_focus_category(question)
    )

    retrieved_docs = retrieve_relevant_documents(
        vector_store,
        question,
        limit=5,
        preferred_category=requested_category,
    )

    links = get_relevant_document_links(
        question,
        retrieved_docs,
        requested_category=requested_category,
    )

    if links:
        return (
            "Berikut tautan yang relevan untuk kebutuhanmu:\n\n"
            + "\n".join(links)
        )

    return (
        "Maaf ya, saya belum menemukan tautan yang sesuai untuk kebutuhan itu.\n\n"
        "Coba jelaskan sedikit lagi proses atau informasi yang kamu cari agar saya bisa mencarinya dengan lebih tepat."
    )


# =========================================================
# JAWABAN SSC UNIVERSAL
# =========================================================
def get_current_daypart():
    """Menghasilkan sapaan waktu sederhana."""
    current_hour = datetime.datetime.now().hour

    if 5 <= current_hour < 11:
        return "pagi"
    if 11 <= current_hour < 15:
        return "siang"
    if 15 <= current_hour < 18:
        return "sore"
    return "malam"


def build_no_document_response(user_question):
    """
    Respons saat informasi belum tersedia. Topik hanya disebut jika pengguna
    sudah menyebutkannya terlebih dahulu.
    """
    category = detect_requested_focus_category(user_question)

    if category:
        return build_missing_category_response(category)

    if is_neutral_greeting_or_help_request(user_question):
        return build_neutral_welcome_response()

    return (
        "Maaf ya, saya belum memiliki informasi yang cukup untuk menjawab pertanyaan itu dengan tepat.\n\n"
        "Coba ceritakan kebutuhanmu sedikit lebih detail, lalu saya akan membantu mengarahkannya."
    )


def build_rag_response(user_question, vector_store, conversation_id=None):
    """
    Membuat jawaban berdasarkan konteks yang sesuai dan menjaga percakapan
    tetap manusiawi, singkat, serta tidak terdengar seperti template.
    """
    context_category = resolve_context_category(
        user_question,
        conversation_id,
    )

    if not context_category:
        return (
            "Boleh ceritakan sedikit lebih detail tentang kebutuhan atau pertanyaanmu?\n\n"
            "Saya akan membantu setelah memahami topik yang kamu maksud."
        )

    retrieved_docs = retrieve_relevant_documents(
        vector_store,
        user_question,
        limit=5,
        preferred_category=context_category,
    )

    if not retrieved_docs:
        return (
            "Maaf ya, saya belum menemukan informasi yang cukup spesifik untuk pertanyaanmu.\n\n"
            "Coba jelaskan sedikit lagi bagian proses yang ingin kamu pahami supaya saya bisa membantu dengan lebih tepat ya."
        )

    context = format_docs(retrieved_docs)
    history = get_chat_history(conversation_id)

    system_prompt = f"""
Anda adalah asisten virtual Student Service Center Telkom University Surabaya.

KONTEKS TOPIK:
Pengguna sedang membahas {context_category}. Gunakan hanya konteks topik tersebut. Jangan mencampurkan informasi dari topik lain.

ATURAN UTAMA:
1. Jawab dengan gaya ramah, hangat, dan natural seperti petugas layanan kampus yang membantu mahasiswa.
2. Gunakan kata "saya" untuk diri sendiri dan "kamu" untuk pengguna. Jangan gunakan kata "aku" atau "Rek".
3. Langsung tanggapi inti pertanyaan pengguna. Jangan membuka jawaban dengan daftar layanan atau kalimat template.
4. Jangan mengulang "Selamat pagi", "Selamat siang", atau "Selamat malam" kecuali pengguna memang menyapa terlebih dahulu.
5. Hindari pembuka yang berulang seperti "Tentu", "Baik", atau "Tenang" pada setiap jawaban.
6. Gunakan hanya fakta yang didukung konteks. Jangan membuat tanggal, syarat, prosedur, biaya, kontak, atau tautan baru.
7. Bila pertanyaan kurang jelas, ajukan satu pertanyaan klarifikasi yang singkat dan alami.
8. Jangan menyebut RAG, vector database, embedding, koleksi, PDF, atau proses teknis apa pun.
9. Jangan berkata bahwa Anda sedang membaca dokumen atau basis data.
10. Buat paragraf pendek dan mudah dipahami. Berikan detail lebih panjang hanya bila pengguna meminta penjelasan detail.

CONTOH GAYA:
Pengguna: "saya bingung mulai dari mana"
Jawaban: "Boleh, saya bantu arahkan. Ceritakan dulu bagian proses yang sedang kamu hadapi agar penjelasannya bisa lebih sesuai."

Pengguna: "masih belum paham"
Jawaban: "Tidak apa-apa. Bagian mana yang masih membingungkan? Kamu bisa sebutkan langkah atau istilah yang ingin dijelaskan lagi."

KONTEKS INFORMASI:
{{context}}
"""

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.32,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({
            "context": context,
            "input": user_question,
            "chat_history": history,
        })
    except Exception:
        return (
            "Maaf ya, saya sedang mengalami kendala sementara saat menyusun jawaban. "
            "Coba kirim pertanyaanmu sekali lagi beberapa saat nanti ya."
        )

    return clean_response_text(response)


def get_response_from_rag(user_question, conversation_id=None):
    """
    Router utama SSC Smart Assistant.

    Alur:
    - sapaan universal;
    - deteksi topik dan typo;
    - pengecekan ketersediaan topik hanya setelah pengguna menyebutnya;
    - pemahaman follow-up;
    - pencarian informasi dan tautan yang sesuai.
    """
    user_question = str(user_question or "").strip()

    if not user_question:
        return "Silakan tuliskan pertanyaanmu terlebih dahulu ya."

    intent = detect_message_intent(user_question)
    explicit_category = detect_requested_focus_category(user_question)

    if intent == "sapaan":
        response = build_neutral_welcome_response()
        save_chat_history(user_question, response, conversation_id)
        return response

    if intent == "terima_kasih":
        response = build_gratitude_response()
        save_chat_history(user_question, response, conversation_id)
        return response

    active_profiles = get_active_document_profiles()

    if not active_profiles:
        response = build_no_document_response(user_question)
        save_chat_history(user_question, response, conversation_id)
        return response

    # Ketersediaan kategori diperiksa hanya setelah mahasiswa menyebutkan
    # kategori tersebut. Hal ini menjaga pengalaman awal tetap universal.
    if explicit_category:
        missing_response = get_missing_focus_category_response(user_question)

        if missing_response:
            save_chat_history(user_question, missing_response, conversation_id)
            return missing_response

        # Untuk pesan pembuka yang baru menyebut topik, ajukan pertanyaan
        # natural tanpa memaksa LLM atau proses retrieval.
        if (
            intent != "permintaan_tautan"
            and is_topic_introduction_message(
                user_question,
                explicit_category,
            )
        ):
            response = build_topic_introduction_response(explicit_category)
            save_chat_history(user_question, response, conversation_id)
            return response

    try:
        embeddings = get_embeddings()
    except EmbeddingUnavailableError:
        category = explicit_category or get_previous_focus_category(conversation_id)

        if category:
            response = (
                f"Maaf ya, saya sedang mengalami kendala sementara saat memuat informasi terkait {category}. "
                "Coba lagi beberapa saat nanti ya."
            )
        else:
            response = (
                "Maaf ya, saya sedang mengalami kendala sementara saat memuat informasi. "
                "Coba lagi beberapa saat nanti ya."
            )

        save_chat_history(user_question, response, conversation_id)
        return response

    collection_name = get_active_collection_name()

    if not collection_has_documents(embeddings, collection_name):
        category = explicit_category or get_previous_focus_category(conversation_id)

        if category:
            response = (
                f"Maaf ya, informasi terkait {category} sedang dipersiapkan. "
                "Silakan coba lagi setelah informasi tersebut tersedia ya."
            )
        else:
            response = (
                "Maaf ya, layanan virtual SSC masih sedang dipersiapkan. "
                "Silakan coba tanyakan lagi beberapa saat nanti ya."
            )

        save_chat_history(user_question, response, conversation_id)
        return response

    vector_store = open_vector_store(
        embeddings,
        collection_name,
    )

    if intent == "permintaan_tautan":
        effective_query = get_link_query(
            user_question,
            conversation_id,
        )

        if not effective_query:
            response = build_link_clarification_response()
            save_chat_history(user_question, response, conversation_id)
            return response

        effective_category = (
            detect_requested_focus_category(effective_query)
            or get_previous_focus_category(conversation_id)
        )

        if effective_category and (
            effective_category not in get_available_focus_categories()
        ):
            response = build_missing_category_response(effective_category)
            save_chat_history(user_question, response, conversation_id)
            return response

        response = build_document_link_response(
            effective_query,
            vector_store,
            requested_category=effective_category,
        )
        save_chat_history(user_question, response, conversation_id)
        return response

    response = build_rag_response(
        user_question,
        vector_store,
        conversation_id,
    )

    save_chat_history(user_question, response, conversation_id)
    return response


# =========================================================
# LAPIS PERCAKAPAN SIAP LAUNCHING
# =========================================================
# Fokus perbaikan:
# - respons sosial yang natural;
# - perpindahan topik tanpa membingungkan;
# - toleransi typo dan bahasa santai;
# - langkah satu per satu yang konsisten;
# - tautan pedoman resmi yang diprioritaskan;
# - menghindari asumsi bahwa mahasiswa sudah menjalani KP atau TA.

MAX_HISTORY_MESSAGES = 18
GUIDED_STEP_SESSIONS = {}
GUIDED_STEP_LOCK = threading.RLock()

# Tautan resmi yang sudah diberikan dan diverifikasi oleh admin aplikasi.
# Ubah hanya jika terdapat tautan resmi terbaru.
OFFICIAL_SERVICE_LINKS = {
    "Kerja Praktik": [
        {
            "label": "Pedoman Kerja Praktik",
            "url": "https://tr.ee/ORVlH4d9xK",
            "keywords": {"pedoman", "panduan", "kerja", "praktik", "praktek", "kp"},
        },
    ],
    "Tugas Akhir": [
        {
            "label": "Pedoman Tugas Akhir",
            "url": "https://tr.ee/0sQf3sjEFr",
            "keywords": {"pedoman", "panduan", "tugas", "akhir", "ta", "skripsi"},
        },
    ],
}


def get_previous_bot_response(conversation_id=None):
    """Mengambil jawaban asisten terakhir pada tab yang sama."""
    history = get_chat_history(conversation_id)

    for message in reversed(history):
        if isinstance(message, AIMessage):
            content = str(message.content or "").strip()
            if content:
                return content

    return ""


def normalize_casual_text(text):
    """Meratakan typo ringan dan pengulangan huruf dalam percakapan santai."""
    normalized = normalize_text(text)
    normalized = re.sub(r"(.)\1{2,}", r"\1\1", normalized)
    return normalized.strip()


def get_conversation_key(conversation_id=None):
    return normalize_conversation_id(conversation_id)


def clear_guided_step_session(conversation_id=None):
    """Mengakhiri mode penjelasan bertahap pada tab tertentu."""
    with GUIDED_STEP_LOCK:
        GUIDED_STEP_SESSIONS.pop(get_conversation_key(conversation_id), None)


def get_guided_step_session(conversation_id=None):
    """Mengambil mode penjelasan bertahap yang masih aktif."""
    with GUIDED_STEP_LOCK:
        return GUIDED_STEP_SESSIONS.get(get_conversation_key(conversation_id))


def set_guided_step_session(conversation_id, category, steps, source_question):
    """Menyimpan urutan langkah yang sedang dijelaskan satu per satu."""
    with GUIDED_STEP_LOCK:
        GUIDED_STEP_SESSIONS[get_conversation_key(conversation_id)] = {
            "category": category,
            "steps": steps,
            "index": 0,
            "source_question": source_question,
            "created_at": datetime.datetime.now().isoformat(),
        }


def has_fuzzy_token_match(words, variants, cutoff=0.76):
    """Mendeteksi kata santai/typo seperti makasii, makasihh, atau yaaaw."""
    for word in words:
        compact_word = compact_token(word)

        for variant in variants:
            compact_variant = compact_token(variant)

            if compact_word == compact_variant:
                return True

            if len(compact_variant) >= 4 and token_similarity(
                compact_word,
                compact_variant,
            ) >= cutoff:
                return True

    return False


def is_neutral_greeting_or_help_request(question):
    """
    Sapaan umum harus mendapat jawaban universal tanpa menyebut KP atau TA.
    """
    normalized = normalize_casual_text(question)

    if not normalized:
        return False

    simple_messages = {
        "halo", "hai", "hi", "pagi", "siang", "sore", "malam",
        "selamat pagi", "selamat siang", "selamat sore", "selamat malam",
        "butuh bantuan", "saya butuh bantuan", "minta bantuan",
        "mau tanya", "ingin bertanya", "bisa bantu saya",
        "bisa bantu", "apa yang bisa dibantu",
    }

    if normalized in simple_messages:
        return True

    greeting_prefixes = (
        "halo ", "hai ", "hi ", "selamat pagi",
        "selamat siang", "selamat sore", "selamat malam",
    )

    return (
        normalized.startswith(greeting_prefixes)
        and len(normalized.split()) <= 9
        and detect_requested_focus_category(normalized) is None
    )


def is_returning_greeting(question):
    """Mengenali sapaan seperti 'hai saya kembali lagi'."""
    normalized = normalize_casual_text(question)

    has_greeting = any(item in normalized.split() for item in {"hai", "halo", "hi"})
    has_returning_signal = any(phrase in normalized for phrase in (
        "saya kembali", "aku kembali", "kembali lagi", "balik lagi",
        "datang lagi", "muncul lagi",
    ))

    return has_greeting and has_returning_signal


def is_general_request_to_ask(question):
    """Mengenali pembuka sebelum pengguna menyampaikan hal yang ingin ditanyakan."""
    normalized = normalize_casual_text(question)

    return normalized in {
        "saya ingin tanya",
        "saya mau tanya",
        "ingin tanya",
        "mau tanya",
        "saya ingin bertanya",
        "ingin bertanya",
        "saya mau bertanya",
        "mau bertanya",
        "ada yang ingin saya tanyakan",
        "saya ada pertanyaan",
    }


def is_gratitude_message(question):
    """
    Mengenali ucapan terima kasih termasuk variasi seperti:
    makasii yaaa, baik makasi, terimakasihh.
    """
    normalized = normalize_casual_text(question)
    words = get_words(normalized)

    direct_messages = {
        "terima kasih", "terimakasih", "makasih", "makasi",
        "thanks", "thank you", "oke makasih", "ok makasih",
        "sip makasih", "baik terimakasih", "baik makasih",
    }

    if normalized in direct_messages:
        return True

    return has_fuzzy_token_match(
        words,
        {"makasih", "makasi", "terimakasih", "thanks"},
        cutoff=0.74,
    )


def is_compliment_message(question):
    """Mengenali apresiasi seperti 'kamu baik sekali'."""
    normalized = normalize_casual_text(question)

    return any(term in normalized for term in (
        "kamu baik",
        "baik sekali",
        "baik banget",
        "membantu sekali",
        "sangat membantu",
        "helpful",
    ))


def is_acknowledgement_message(question):
    """Respons pendek pengguna yang bukan pertanyaan baru."""
    normalized = normalize_casual_text(question)

    exact_messages = {
        "iya", "iya ya", "iya begitu ya", "oh begitu ya",
        "oo begitu ya", "o begitu ya", "ohh begitu ya",
        "oh begitu", "oo begitu", "o begitu",
        "oh paham", "oo paham", "paham", "saya paham",
        "baik", "baiklah", "oke", "ok", "okeh", "sip",
        "siap", "yaudah", "ya udah", "hmm", "oh", "oo",
        "ooh", "begitu ya", "emm begitu ya", "em begitu ya",
    }

    if normalized in exact_messages:
        return True

    return (
        len(normalized.split()) <= 7
        and normalized.startswith((
            "oh begitu", "oo begitu", "o begitu",
            "oh paham", "oo paham", "iya begitu",
            "oke berarti", "sip berarti", "emm begitu",
        ))
    )


def is_completion_message(question):
    """
    Mendeteksi pengguna telah memahami penjelasan tanpa mengasumsikan
    bahwa pengguna sedang menjalankan proses tersebut.
    """
    normalized = normalize_casual_text(question)

    return any(term in normalized for term in (
        "sudah paham", "udah paham", "saya paham",
        "sudah mengerti", "udah mengerti", "saya mengerti",
        "paham terkait", "mengerti terkait",
    ))


def is_user_correction_message(question):
    """Mendeteksi ketika pengguna memperbaiki asumsi asisten."""
    normalized = normalize_casual_text(question)

    return normalized.startswith((
        "kan saya belum",
        "saya belum",
        "aku belum",
        "belum mengerjakan",
        "belum mulai",
        "saya masih persiapan",
        "saya belum sampai",
        "belum kok",
    ))


def is_cancel_message(question):
    """Mengenali pembatalan santai seperti 'tidak jadi deh'."""
    normalized = normalize_casual_text(question)

    return normalized in {
        "tidak jadi", "tidak jadi deh", "ga jadi", "gak jadi",
        "nggak jadi", "belum jadi", "nanti saja", "nanti aja",
        "sudah cukup", "cukup dulu",
    }


def is_change_mind_to_topic_message(question, category):
    """
    Mendeteksi perpindahan topik seperti:
    'oh tidak jadi, saya ingin bertanya terkait tugas akhir'.
    """
    if not category:
        return False

    normalized = normalize_casual_text(question)

    return (
        any(term in normalized for term in (
            "tidak jadi", "ga jadi", "gak jadi", "nggak jadi",
        ))
        and any(term in normalized for term in (
            "ingin tanya", "ingin bertanya", "mau tanya",
            "mau bertanya", "terkait", "tentang", "mengenai",
        ))
    )


def is_opening_prompt_message(question):
    """
    Mendeteksi pesan seperti 'tau ngga?' yang merupakan pembuka cerita,
    bukan pertanyaan yang harus dicari di pedoman.
    """
    normalized = normalize_casual_text(question)

    return normalized in {
        "tau ngga",
        "tahu ngga",
        "tau gak",
        "tahu gak",
        "tau nggak",
        "tahu nggak",
        "kamu tau ngga",
        "kamu tahu ngga",
        "kamu tau gak",
        "kamu tahu gak",
    }


def is_step_by_step_request(question):
    normalized = normalize_casual_text(question)

    return any(pattern in normalized for pattern in (
        "step by step",
        "langkah demi langkah",
        "satu satu",
        "1 1",
        "satu persatu",
        "pelan pelan",
        "jelaskan bertahap",
        "urut satu satu",
    ))


def is_continue_step_request(question):
    normalized = normalize_casual_text(question)

    return normalized in {
        "lanjut", "lanjutkan", "terus", "berikutnya",
        "next", "lanjut ya", "lanjut dong", "teruskan",
    }


def is_step_count_request(question):
    normalized = normalize_casual_text(question)

    return any(term in normalized for term in (
        "berapa langkah",
        "ada berapa langkah",
        "berapa tahap",
        "ada berapa tahap",
        "jumlah langkah",
        "jumlah tahap",
        "total langkah",
        "total tahap",
    ))


def build_neutral_welcome_response():
    return "Halo! Ada yang bisa saya bantu hari ini?"


def build_returning_greeting_response():
    return "Hai, selamat datang kembali. Ada yang ingin kamu bahas hari ini?"


def build_general_request_response():
    return "Tentu, silakan. Kamu ingin menanyakan hal apa?"


def build_gratitude_response():
    return (
        "Sama-sama. Senang bisa membantu. "
        "Kalau nanti ada hal lain yang ingin kamu pahami, silakan tanyakan saja ya."
    )


def build_compliment_response(conversation_id=None):
    category = get_previous_focus_category(conversation_id)

    if category:
        return (
            f"Terima kasih, itu baik sekali. Senang bisa membantu kamu memahami {category}. "
            "Semoga penjelasan tadi membuat persiapanmu terasa lebih ringan."
        )

    return (
        "Terima kasih, itu baik sekali. Senang bisa membantu kamu. "
        "Kalau ada yang ingin dibahas lagi, silakan tulis saja ya."
    )


def build_completion_response(conversation_id=None):
    category = get_previous_focus_category(conversation_id)

    if category:
        return (
            f"Senang penjelasan tentang {category} tadi sudah lebih jelas. "
            "Semoga ini membantu kamu mempersiapkan diri dengan lebih tenang. "
            "Kamu tidak perlu memikirkan semuanya sekaligus, cukup pahami satu tahap demi satu tahap."
        )

    return (
        "Senang penjelasannya sudah lebih jelas. "
        "Kalau nanti ada bagian lain yang ingin kamu pahami, silakan tanyakan saja ya."
    )


def build_correction_response(conversation_id=None):
    category = get_previous_focus_category(conversation_id)

    if category:
        return (
            f"Iya, maaf ya, saya tadi terlalu mengasumsikan kamu sudah menjalani {category}. "
            "Berarti kamu masih di tahap persiapan. Itu justru bagus karena kamu bisa memahami alurnya lebih awal dan menyiapkan diri dengan lebih tenang."
        )

    return (
        "Iya, maaf ya, saya tadi terlalu cepat berasumsi. "
        "Terima kasih sudah meluruskan. Saya akan menyesuaikan penjelasannya dengan kondisi kamu."
    )


def build_cancel_response():
    return (
        "Tidak apa-apa. Kita bisa membahasnya kapan saja saat kamu sudah ingin melanjutkan. "
        "Kalau ada pertanyaan lain, silakan tulis saja ya."
    )


def build_acknowledgement_response(conversation_id=None):
    category = get_previous_focus_category(conversation_id)
    previous_answer = get_previous_bot_response(conversation_id).lower()

    if category and any(word in previous_answer for word in (
        "langkah", "tahap", "proses", "alur", "persiapan",
    )):
        return (
            "Iya, kurang lebih begitu. Kamu tidak perlu memahami semuanya sekaligus. "
            "Cukup fokus pada satu tahap terlebih dahulu, lalu lanjut saat sudah lebih siap."
        )

    if category:
        return (
            "Iya, kurang lebih begitu. "
            "Kalau masih ada bagian yang membuat kamu ragu atau bingung, sebutkan saja dan saya bantu jelaskan dengan lebih sederhana."
        )

    return (
        "Iya, kurang lebih begitu. "
        "Kalau ada bagian yang masih ingin kamu tanyakan, silakan bilang saja ya."
    )


def build_opening_prompt_response():
    return (
        "Saya belum tahu maksudnya, tapi kamu boleh ceritakan dulu ya. "
        "Saya siap mendengarkan."
    )


def get_previous_focus_category(conversation_id=None):
    """Mengambil topik eksplisit dari pesan pengguna sebelumnya."""
    history = get_chat_history(conversation_id)

    for message in reversed(history):
        if not isinstance(message, HumanMessage):
            continue

        category = detect_requested_focus_category(
            str(message.content or "")
        )

        if category:
            return category

    return None


def get_previous_focus_question(conversation_id=None):
    """Mengambil pertanyaan sebelumnya yang sudah mempunyai topik jelas."""
    history = get_chat_history(conversation_id)

    for message in reversed(history):
        if not isinstance(message, HumanMessage):
            continue

        content = str(message.content or "").strip()

        if content and detect_requested_focus_category(content):
            return content

    return ""


def resolve_context_category(question, conversation_id=None):
    """
    Menggunakan kategori yang disebut saat ini. Untuk follow-up singkat,
    gunakan kategori dari riwayat percakapan.
    """
    explicit_category = detect_requested_focus_category(question)

    if explicit_category:
        return explicit_category

    social_messages = (
        is_neutral_greeting_or_help_request(question)
        or is_gratitude_message(question)
        or is_compliment_message(question)
        or is_acknowledgement_message(question)
        or is_completion_message(question)
        or is_user_correction_message(question)
        or is_cancel_message(question)
        or is_opening_prompt_message(question)
    )

    if social_messages:
        return None

    return get_previous_focus_category(conversation_id)


def is_topic_introduction_message(question, category):
    """
    Mendeteksi pembuka topik tanpa pertanyaan detail, termasuk variasi:
    'baikk, sekarang saya ingin tanya tentang tugas akhir'.
    """
    if not category:
        return False

    normalized = normalize_casual_text(question)

    has_intro_signal = any(phrase in normalized for phrase in (
        "ingin tanya",
        "ingin bertanya",
        "mau tanya",
        "mau bertanya",
        "terkait",
        "tentang",
        "seputar",
        "mengenai",
    ))

    if not has_intro_signal:
        return False

    filler_words = {
        "baik", "baikk", "oke", "ok", "okeh", "sekarang", "saya",
        "aku", "ingin", "mau", "minta", "tolong", "boleh",
        "bertanya", "tanya", "terkait", "tentang", "seputar",
        "mengenai", "info", "informasi", "dong", "ya", "nih", "kak",
        "oh", "oo", "ohh", "tidak", "jadi", "deh",
        "kerja", "praktik", "praktek", "kp", "kpp", "magang",
        "tugas", "akhir", "ahir", "ta", "taa", "skripsi",
    }

    meaningful_words = [
        word for word in get_words(normalized)
        if word not in filler_words
    ]

    return len(meaningful_words) == 0


def build_topic_introduction_response(category):
    return (
        f"Boleh, saya siap membantu terkait {category}. "
        "Bagian apa yang ingin kamu tanyakan?"
    )


def profile_matches_category(profile, category):
    """Mencocokkan kategori berdasarkan registry, judul, dan nama file."""
    profile_text = " ".join([
        str(profile.get("category", "")),
        str(profile.get("title", "")),
        str(profile.get("filename", "")),
    ])

    return (
        category_matches_focus(
            profile.get("category", ""),
            category,
        )
        or detect_requested_focus_category(profile_text) == category
    )


def is_category_available_locally(category):
    """
    Menggunakan file aktif yang benar-benar ada di uploads, bukan hanya
    metadata lama. Ini mencegah status KP/TA berubah-ubah karena registry.
    """
    profiles = get_active_document_profiles()

    return any(
        profile_matches_category(profile, category)
        for profile in profiles
    )


def build_contextual_retrieval_query(question, category, conversation_id=None):
    """
    Memperjelas pertanyaan follow-up pendek sebelum melakukan retrieval.
    """
    current_question = str(question or "").strip()
    explicit_category = detect_requested_focus_category(current_question)

    if explicit_category:
        return f"Topik: {explicit_category}. Pertanyaan: {current_question}"

    previous_question = get_previous_user_question(conversation_id)

    if category and previous_question:
        return (
            f"Topik: {category}. "
            f"Pertanyaan sebelumnya: {previous_question}. "
            f"Pertanyaan lanjutan: {current_question}"
        )

    if category:
        return f"Topik: {category}. Pertanyaan: {current_question}"

    return current_question


def normalize_step_item(item):
    if not isinstance(item, dict):
        return None

    title = str(item.get("title") or "").strip()
    explanation = str(item.get("explanation") or "").strip()

    if not title or not explanation:
        return None

    return {
        "title": re.sub(r"\s+", " ", title)[:100],
        "explanation": re.sub(r"\s+", " ", explanation)[:560],
    }


def fallback_step_plan(category):
    """
    Cadangan umum yang tidak mengklaim syarat spesifik kampus.
    Dipakai hanya ketika informasi langkah tidak bisa dibentuk dari konteks.
    """
    if category == "Kerja Praktik":
        return [
            {
                "title": "Pahami ketentuan awal",
                "explanation": (
                    "Mulailah dengan memahami alur, persyaratan, serta kebutuhan administrasi yang berlaku sebelum memulai proses pengajuan."
                ),
            },
            {
                "title": "Siapkan kebutuhan pengajuan",
                "explanation": (
                    "Setelah memahami ketentuannya, siapkan informasi dan berkas yang diperlukan agar proses awal bisa berjalan lebih rapi."
                ),
            },
            {
                "title": "Jalani kegiatan sesuai arahan",
                "explanation": (
                    "Saat kegiatan sudah dimulai, ikuti arahan dari pihak terkait dan catat hal penting yang nantinya diperlukan untuk pelaporan."
                ),
            },
            {
                "title": "Susun hasil atau laporan",
                "explanation": (
                    "Dokumentasikan proses dan hasil secara bertahap agar penyusunan laporan tidak terasa menumpuk di akhir."
                ),
            },
            {
                "title": "Selesaikan tahapan penutup",
                "explanation": (
                    "Setelah hasil atau laporan siap, selesaikan evaluasi maupun pengumpulan akhir sesuai arahan yang berlaku."
                ),
            },
        ]

    return [
        {
            "title": "Tentukan arah topik",
            "explanation": (
                "Mulailah dari bidang atau masalah yang ingin kamu pahami. Pilih arah yang cukup kamu minati agar proses persiapannya lebih nyaman."
            ),
        },
        {
            "title": "Susun rencana awal",
            "explanation": (
                "Rancang masalah, tujuan, dan gambaran pendekatan yang ingin digunakan agar kamu memiliki arah sebelum masuk ke pembahasan yang lebih detail."
            ),
        },
        {
            "title": "Kembangkan melalui bimbingan",
            "explanation": (
                "Setelah arah awal terbentuk, kembangkan pembahasan secara bertahap melalui arahan dan penyesuaian yang diperlukan."
            ),
        },
        {
            "title": "Susun hasil menjadi laporan",
            "explanation": (
                "Tuangkan proses dan hasil yang diperoleh ke dalam laporan secara bertahap agar penulisannya lebih teratur."
            ),
        },
        {
            "title": "Selesaikan evaluasi akhir",
            "explanation": (
                "Pada tahap akhir, siapkan kebutuhan evaluasi dan lakukan perbaikan sesuai arahan yang diberikan."
            ),
        },
    ]


def generate_guided_step_plan(vector_store, category, question, conversation_id=None):
    """
    Menyusun langkah dari konteks relevan. Bila format tidak dapat diproses,
    gunakan fallback umum yang aman.
    """
    retrieval_query = build_contextual_retrieval_query(
        question,
        category,
        conversation_id,
    )

    docs = retrieve_relevant_documents(
        vector_store,
        retrieval_query,
        limit=6,
        preferred_category=category,
    )

    if not docs:
        return fallback_step_plan(category)

    context = format_docs(docs)

    step_prompt = f"""
Anda menyusun panduan bertahap untuk Student Service Center.

TOPIK: {category}

Gunakan hanya informasi yang didukung oleh konteks. Buat 3 sampai 7 langkah.
Jangan membuat syarat, jadwal, atau prosedur yang tidak ada pada konteks.
Gunakan bahasa singkat dan mudah dipahami.

Balas HANYA dalam JSON valid:
{{
  "steps": [
    {{"title": "judul singkat", "explanation": "penjelasan satu langkah"}}
  ]
}}

KONTEKS:
{context}
"""

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.12,
        )
        result = llm.invoke(step_prompt)
        raw_text = str(getattr(result, "content", result)).strip()

        json_match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        payload = json.loads(json_match.group(0) if json_match else raw_text)
        raw_steps = payload.get("steps", []) if isinstance(payload, dict) else []
        steps = [normalize_step_item(item) for item in raw_steps]
        steps = [item for item in steps if item]

        if 3 <= len(steps) <= 7:
            return steps
    except Exception:
        pass

    return fallback_step_plan(category)


def render_step_response(session, index):
    """Menyajikan satu langkah saja agar pengguna tidak kewalahan."""
    steps = session["steps"]
    step = steps[index]
    total = len(steps)

    response = (
        f"Langkah {index + 1} dari {total}: {step['title']}\n\n"
        f"{step['explanation']}"
    )

    if index + 1 < total:
        response += '\n\nKalau sudah siap, tulis "lanjut" untuk ke langkah berikutnya.'
    else:
        response += "\n\nItu langkah terakhir. Kalau ada langkah yang ingin dibahas lebih dalam, sebutkan saja ya."

    return response


def build_guided_steps_overview(session):
    """Memberi jumlah serta daftar judul langkah tanpa mengulang penjelasan."""
    steps = session["steps"]
    current = session["index"] + 1
    titles = "\n".join(
        f"{position}. {step['title']}"
        for position, step in enumerate(steps, start=1)
    )

    return (
        f"Untuk penjelasan bertahap ini ada {len(steps)} langkah. "
        f"Kamu sekarang berada di langkah {current}.\n\n"
        f"{titles}\n\n"
        'Tulis "lanjut" saat kamu ingin saya jelaskan langkah berikutnya.'
    )


def handle_guided_step_message(question, vector_store, category, conversation_id=None):
    """
    Menangani permintaan step-by-step, lanjut, dan jumlah langkah.
    """
    session = get_guided_step_session(conversation_id)
    explicit_category = detect_requested_focus_category(question)

    if (
        session
        and explicit_category
        and explicit_category != session.get("category")
        and not is_continue_step_request(question)
    ):
        clear_guided_step_session(conversation_id)
        session = None

    if is_step_by_step_request(question):
        active_category = category or get_previous_focus_category(conversation_id)

        if not active_category:
            return (
                "Boleh. Sebutkan dulu topik atau proses yang ingin kamu pelajari satu per satu, "
                "supaya saya bisa menjelaskannya dengan urut."
            )

        steps = generate_guided_step_plan(
            vector_store,
            active_category,
            question,
            conversation_id,
        )

        set_guided_step_session(
            conversation_id,
            active_category,
            steps,
            question,
        )

        session = get_guided_step_session(conversation_id)

        return (
            "Baik, saya jelaskan pelan-pelan satu per satu ya.\n\n"
            + render_step_response(session, 0)
        )

    if session and is_step_count_request(question):
        return build_guided_steps_overview(session)

    if session and is_continue_step_request(question):
        next_index = session["index"] + 1

        if next_index >= len(session["steps"]):
            return (
                "Kita sudah sampai di langkah terakhir. "
                "Kalau kamu ingin, saya bisa menjelaskan salah satu langkah tadi dengan lebih detail."
            )

        session["index"] = next_index
        return render_step_response(session, next_index)

    return None


def build_missing_category_response(category):
    return (
        f"Maaf ya, informasi terkait {category} belum siap tersedia di SSC. "
        "Saya tidak ingin memberi arahan yang kurang tepat. "
        "Silakan coba tanyakan lagi setelah informasinya diperbarui ya."
    )


def get_curated_links(category, question):
    """
    Mengambil tautan resmi yang disimpan admin. Untuk permintaan pedoman,
    tautan ini diprioritaskan daripada tautan umum lain.
    """
    if not category:
        return []

    items = OFFICIAL_SERVICE_LINKS.get(category, [])
    normalized_question = normalize_casual_text(question)
    requested_keywords = set(get_words(normalized_question))

    selected = []

    for item in items:
        item_keywords = item.get("keywords", set())

        # Tautan pedoman selalu relevan untuk permintaan link umum kategori,
        # dan menjadi pilihan utama jika pengguna menyebut pedoman/panduan.
        if not requested_keywords or (
            requested_keywords & set(item_keywords)
        ):
            selected.append(item)

    return selected or items


def format_curated_links_response(category, question):
    """Menampilkan tautan admin dengan label yang jelas dan tidak bercampur."""
    items = get_curated_links(category, question)

    if not items:
        return ""

    lines = [
        f"Berikut tautan resmi yang relevan untuk {category}:",
        "",
    ]

    for item in items:
        lines.append(f"{item['label']}:")
        lines.append(item["url"])

    return "\n".join(lines)


def build_rag_response(user_question, vector_store, conversation_id=None):
    """
    Menjawab berdasarkan konteks aktif dengan gaya profesional, hangat,
    dan tidak mengasumsikan kondisi pengguna.
    """
    context_category = resolve_context_category(
        user_question,
        conversation_id,
    )

    if not context_category:
        return (
            "Boleh ceritakan sedikit lagi kebutuhanmu? "
            "Saya ingin memahami dulu hal yang sedang kamu tanyakan supaya jawabannya lebih sesuai."
        )

    retrieval_query = build_contextual_retrieval_query(
        user_question,
        context_category,
        conversation_id,
    )

    retrieved_docs = retrieve_relevant_documents(
        vector_store,
        retrieval_query,
        limit=5,
        preferred_category=context_category,
    )

    if not retrieved_docs:
        return (
            "Maaf ya, saya belum menemukan penjelasan yang cukup untuk menjawab bagian itu dengan tepat. "
            "Coba tulis sedikit lebih spesifik, misalnya kamu sedang mencari persiapan, langkah, syarat, jadwal, atau proses tertentu."
        )

    context = format_docs(retrieved_docs)
    history = get_chat_history(conversation_id)

    system_prompt = f"""
Anda adalah asisten virtual Student Service Center Telkom University Surabaya.

TOPIK AKTIF:
Pengguna sedang membahas {context_category}. Gunakan hanya informasi yang sesuai dengan topik tersebut. Jangan mencampurkan informasi dari topik lain.

KARAKTER:
Anda adalah asisten profesional yang hangat, teliti, dan mudah diajak bicara. Jawaban tidak boleh terdengar seperti buku teks, template otomatis, atau chatbot yang hanya menempelkan definisi.

ATURAN JAWABAN:
1. Gunakan "saya" untuk diri sendiri dan "kamu" untuk pengguna. Jangan gunakan "aku" atau "Rek".
2. Tanggapi inti pertanyaan terlebih dahulu. Hindari pembuka yang berulang seperti "Tentu", "Baik", "Selamat malam", atau "Singkatnya" jika tidak diperlukan.
3. Untuk pertanyaan persiapan, jangan mengasumsikan pengguna sudah menjalani prosesnya. Gunakan bahasa seperti "mempersiapkan diri", "memahami alur", atau "mulai dari tahap awal".
4. Jangan menggunakan pertanyaan retoris seperti "kan?" di akhir jawaban.
5. Bila pengguna tampak khawatir atau ragu, berikan dukungan singkat tanpa berlebihan.
6. Bila pengguna meminta langkah, gunakan urutan yang jelas. Bila tidak diminta, jangan memaksakan daftar panjang.
7. Bila pengguna mengatakan belum paham, jelaskan lebih sederhana dan tawarkan penjelasan bagian tertentu.
8. Bila pengguna mengoreksi asumsi, minta maaf singkat dan sesuaikan penjelasan.
9. Gunakan hanya fakta dari konteks. Jangan membuat syarat, jadwal, dosen, biaya, kontak, atau prosedur yang tidak tersedia.
10. Jangan memberi disclaimer umum seperti "setiap kampus berbeda" kecuali konteks memang menyatakannya.
11. Jangan menyebut PDF, RAG, embedding, vector database, koleksi, basis data, atau proses teknis.
12. Jangan berkata bahwa Anda sedang membaca dokumen.
13. Jangan mengakhiri setiap jawaban dengan pertanyaan. Ajukan pertanyaan lanjutan hanya jika benar-benar diperlukan.
14. Gunakan paragraf pendek yang enak dibaca. Detail panjang hanya jika pengguna meminta penjelasan detail.

CONTOH GAYA:
Pengguna: "apa yang perlu saya persiapkan?"
Jawaban: "Untuk tahap awal, kamu bisa mulai dari memahami alur dan menyiapkan hal-hal yang relevan dengan topikmu. Setelah itu, baru lanjut ke bagian yang lebih teknis secara bertahap."

Pengguna: "saya belum mengerjakannya"
Jawaban: "Iya, maaf ya, saya tadi terlalu mengasumsikan. Berarti kamu masih di tahap persiapan, jadi kita bisa fokus memahami alurnya lebih dulu."

Pengguna: "apakah bab 1 susah?"
Jawaban: "Bab 1 memang sering terasa menantang di awal karena kamu perlu merapikan alasan, masalah, dan tujuan yang ingin dibahas. Tapi bagian ini biasanya jauh lebih mudah setelah topikmu sudah jelas."

KONTEKS INFORMASI:
{{context}}
"""

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.30,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({
            "context": context,
            "input": user_question,
            "chat_history": history,
        })
    except Exception:
        return (
            "Maaf ya, saya sedang mengalami kendala sementara saat menyusun jawaban. "
            "Coba kirim pertanyaanmu sekali lagi beberapa saat nanti ya."
        )

    cleaned_response = clean_response_text(response)

    if not cleaned_response:
        return (
            "Maaf ya, saya belum bisa menyusun jawaban yang tepat untuk bagian itu. "
            "Coba jelaskan sedikit lagi kebutuhanmu ya."
        )

    return cleaned_response


def build_document_link_response(question, vector_store, requested_category=None, conversation_id=None):
    """
    Prioritas pertama: tautan resmi yang sudah disimpan admin.
    Jika pengguna meminta link lebih spesifik yang belum ada di konfigurasi,
    baru cari dari konteks informasi aktif.
    """
    requested_category = (
        requested_category
        or detect_requested_focus_category(question)
        or get_previous_focus_category(conversation_id)
    )

    curated_response = format_curated_links_response(
        requested_category,
        question,
    )

    normalized_question = normalize_casual_text(question)
    wants_pedoman = any(term in normalized_question for term in (
        "pedoman", "panduan",
    ))

    # Untuk permintaan kategori umum atau pedoman spesifik, tautan resmi
    # yang dikurasi admin adalah hasil paling aman dan paling relevan.
    if curated_response and (
        wants_pedoman
        or detect_requested_focus_category(question)
        or is_generic_link_request(question)
    ):
        return curated_response

    retrieval_query = build_contextual_retrieval_query(
        question,
        requested_category,
        conversation_id,
    )

    retrieved_docs = retrieve_relevant_documents(
        vector_store,
        retrieval_query,
        limit=5,
        preferred_category=requested_category,
    )

    links = get_relevant_document_links(
        retrieval_query,
        retrieved_docs,
        requested_category=requested_category,
    )

    if links:
        return (
            "Berikut tautan yang relevan untuk kebutuhanmu:\n\n"
            + "\n".join(links)
        )

    if curated_response:
        return curated_response

    return (
        "Maaf ya, saya belum menemukan tautan yang sesuai untuk kebutuhan itu. "
        "Coba sebutkan bagian proses atau informasi yang kamu cari agar saya bisa mencarinya lagi dengan lebih tepat."
    )


def get_response_from_rag(user_question, conversation_id=None):
    """
    Router percakapan siap launching:
    - menangani percakapan sosial dengan natural;
    - memahami koreksi, pembatalan, dan pergantian topik;
    - menjaga konteks follow-up;
    - menjelaskan langkah satu per satu;
    - memberi tautan resmi spesifik.
    """
    user_question = str(user_question or "").strip()

    if not user_question:
        return "Silakan tuliskan pertanyaanmu terlebih dahulu ya."

    explicit_category = detect_requested_focus_category(user_question)

    # 1. Respons sosial yang tidak membutuhkan vector store.
    if is_returning_greeting(user_question):
        clear_guided_step_session(conversation_id)
        response = build_returning_greeting_response()
        save_chat_history(user_question, response, conversation_id)
        return response

    if is_neutral_greeting_or_help_request(user_question):
        response = build_neutral_welcome_response()
        save_chat_history(user_question, response, conversation_id)
        return response

    if is_general_request_to_ask(user_question):
        response = build_general_request_response()
        save_chat_history(user_question, response, conversation_id)
        return response

    if is_opening_prompt_message(user_question):
        response = build_opening_prompt_response()
        save_chat_history(user_question, response, conversation_id)
        return response

    if is_user_correction_message(user_question):
        response = build_correction_response(conversation_id)
        save_chat_history(user_question, response, conversation_id)
        return response

    if is_completion_message(user_question):
        response = build_completion_response(conversation_id)
        save_chat_history(user_question, response, conversation_id)
        return response

    if is_compliment_message(user_question):
        response = build_compliment_response(conversation_id)
        save_chat_history(user_question, response, conversation_id)
        return response

    if is_gratitude_message(user_question):
        response = build_gratitude_response()
        save_chat_history(user_question, response, conversation_id)
        return response

    # Perpindahan topik diproses sebelum pembatalan umum.
    if explicit_category and is_change_mind_to_topic_message(
        user_question,
        explicit_category,
    ):
        clear_guided_step_session(conversation_id)

        if not is_category_available_locally(explicit_category):
            response = build_missing_category_response(explicit_category)
        else:
            response = build_topic_introduction_response(explicit_category)

        save_chat_history(user_question, response, conversation_id)
        return response

    if is_cancel_message(user_question):
        clear_guided_step_session(conversation_id)
        response = build_cancel_response()
        save_chat_history(user_question, response, conversation_id)
        return response

    if is_acknowledgement_message(user_question):
        response = build_acknowledgement_response(conversation_id)
        save_chat_history(user_question, response, conversation_id)
        return response

    active_profiles = get_active_document_profiles()

    if not active_profiles:
        response = build_no_document_response(user_question)
        save_chat_history(user_question, response, conversation_id)
        return response

    # 2. Pastikan topik yang disebut memang punya file aktif.
    if explicit_category:
        current_session = get_guided_step_session(conversation_id)

        if (
            current_session
            and current_session.get("category") != explicit_category
        ):
            clear_guided_step_session(conversation_id)

        if not is_category_available_locally(explicit_category):
            response = build_missing_category_response(explicit_category)
            save_chat_history(user_question, response, conversation_id)
            return response

        if (
            not has_link_request(user_question)
            and not is_step_by_step_request(user_question)
            and is_topic_introduction_message(
                user_question,
                explicit_category,
            )
        ):
            response = build_topic_introduction_response(explicit_category)
            save_chat_history(user_question, response, conversation_id)
            return response

    context_category = resolve_context_category(
        user_question,
        conversation_id,
    )

    try:
        embeddings = get_embeddings()
    except EmbeddingUnavailableError:
        category = explicit_category or context_category

        response = (
            f"Maaf ya, saya sedang mengalami kendala sementara saat memuat informasi terkait {category}. "
            "Coba lagi beberapa saat nanti ya."
            if category
            else
            "Maaf ya, saya sedang mengalami kendala sementara saat memuat informasi. "
            "Coba lagi beberapa saat nanti ya."
        )

        save_chat_history(user_question, response, conversation_id)
        return response

    collection_name = get_active_collection_name()

    if not collection_has_documents(embeddings, collection_name):
        category = explicit_category or context_category

        response = (
            build_missing_category_response(category)
            if category
            else
            "Maaf ya, layanan virtual SSC masih sedang dipersiapkan. Silakan coba tanyakan lagi beberapa saat nanti ya."
        )

        save_chat_history(user_question, response, conversation_id)
        return response

    vector_store = open_vector_store(
        embeddings,
        collection_name,
    )

    # 3. Mode penjelasan satu per satu.
    guided_response = handle_guided_step_message(
        user_question,
        vector_store,
        context_category,
        conversation_id,
    )

    if guided_response:
        save_chat_history(user_question, guided_response, conversation_id)
        return guided_response

    # 4. Tautan resmi atau tautan dari konteks aktif.
    if has_link_request(user_question):
        effective_query = get_link_query(
            user_question,
            conversation_id,
        )

        if not effective_query:
            response = build_link_clarification_response()
            save_chat_history(user_question, response, conversation_id)
            return response

        effective_category = (
            detect_requested_focus_category(effective_query)
            or get_previous_focus_category(conversation_id)
        )

        # Jika ada tautan resmi yang dikurasi admin, tautan itu tetap dapat
        # diberikan secara konsisten untuk layanan terkait.
        response = build_document_link_response(
            effective_query,
            vector_store,
            requested_category=effective_category,
            conversation_id=conversation_id,
        )

        save_chat_history(user_question, response, conversation_id)
        return response

    # 5. Jawaban berbasis konteks.
    response = build_rag_response(
        user_question,
        vector_store,
        conversation_id,
    )

    save_chat_history(user_question, response, conversation_id)
    return response


# =========================================================
# DETEKKSI PERMINTAAN DOKUMEN PDF
# =========================================================
def detect_document_request(question, conversation_id=None):
    """
    Mengembalikan kategori dokumen bila pengguna memang meminta file/pedoman.

    Contoh yang dikenali:
    - kirim pedoman kerja praktik
    - saya mau file KP
    - download PDF tugas akhir
    - tolong berikan panduan TA
    - saya ingin dokumennya
    """
    normalized = normalize_text(question)

    if not normalized:
        return None

    # Mendeteksi kategori dari pesan saat ini.
    category = detect_requested_focus_category(normalized)

    # Mendukung follow-up:
    # User: "Saya ingin tahu tentang TA"
    # User: "Kirim pedomannya"
    if not category and conversation_id:
        category = get_previous_focus_category(conversation_id)

    if not category:
        return None

    document_terms = (
        "dokumen",
        "file",
        "pdf",
        "pedoman",
        "panduan",
        "buku",
    )

    download_terms = (
        "download",
        "unduh",
    )

    has_document_term = any(term in normalized for term in document_terms)
    has_download_term = any(term in normalized for term in download_terms)

    if has_document_term or has_download_term:
        return category

    return None

