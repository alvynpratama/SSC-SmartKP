import datetime
import json
import os
from functools import wraps
from pathlib import Path

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from rag_logic import (
    OFFICIAL_SERVICE_LINKS,
    build_vector_store,
    get_previous_focus_category,
    get_response_from_rag,
    save_chat_history,
)


# =========================================================
# PATH PROJECT
# =========================================================
# Semua path dibuat absolut berdasarkan lokasi app.py.
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
VECTOR_DB_DIR = BASE_DIR / "vector_store"
DOCUMENT_REGISTRY_FILE = BASE_DIR / "document_registry.json"

# PDF menjadi sumber knowledge base RAG, sedangkan DOCX menjadi template unduhan.
ALLOWED_EXTENSIONS = {"pdf", "docx"}
FOCUS_CATEGORIES = ("Kerja Praktik", "Tugas Akhir")
DOCUMENT_TYPES = ("pedoman", "template")

# Ganti nilai ini sebelum aplikasi dipakai secara publik.
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin"


app = Flask(__name__)
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "supersecretkeytelkom_ganti_saat_deploy",
)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["ALLOWED_EXTENSIONS"] = ALLOWED_EXTENSIONS
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # Maksimum upload: 25 MB


# =========================================================
# UTILITAS UMUM
# =========================================================
def ensure_project_folders():
    """Memastikan folder utama aplikasi selalu tersedia."""
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


def file_extension(filename):
    """Mengambil ekstensi file dalam huruf kecil, termasuk titik."""
    return Path(str(filename or "")).suffix.lower()


def allowed_file(filename):
    """Memastikan file upload adalah PDF atau Word DOCX."""
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def normalize_document_type(filename):
    """
    Menentukan fungsi dokumen berdasarkan format file.

    PDF dipakai sebagai pedoman dan sumber knowledge base RAG.
    DOCX dipakai sebagai template unduhan dan tidak masuk ke RAG.
    """
    return "template" if file_extension(filename) == ".docx" else "pedoman"


def document_type_label(document_type):
    """Label ramah pengguna untuk tampilan workspace."""
    return "Template Word" if document_type == "template" else "Pedoman PDF"


def is_knowledge_source(filename):
    """Hanya PDF yang diproses ke vector store RAG."""
    return file_extension(filename) == ".pdf" and normalize_document_type(filename) == "pedoman"


def login_required(view_function):
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return view_function(*args, **kwargs)

    return wrapped_view


def normalize_filename_text(value):
    """Menormalkan nama file atau judul agar KP dan TA mudah dikenali."""
    return (
        str(value or "")
        .lower()
        .replace("-", " ")
        .replace("_", " ")
        .replace(".", " ")
    )


def display_title_from_filename(filename):
    """Membuat judul yang lebih rapi dari nama file."""
    title = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    return title or "Dokumen SSC"


def infer_document_category(filename, title=""):
    """
    Menentukan kategori berdasarkan nama file atau judul dokumen.
    Dokumen yang tidak dapat dikenali tetap tampil di workspace,
    tetapi tidak dipakai untuk pengiriman pedoman KP atau TA.
    """
    text = normalize_filename_text(f"{filename} {title}")

    kerja_praktik_terms = (
        "kerja praktik",
        "kerja praktek",
        "praktik kerja",
        "praktek kerja",
        "magang",
        "internship",
        "pedoman kp",
        "panduan kp",
        "template kp",
        "format kp",
        "template kerja praktik",
        "proposal kerja praktik",
    )

    tugas_akhir_terms = (
        "tugas akhir",
        "tugas ahir",
        "skripsi",
        "sempro",
        "semhas",
        "sidang",
        "proposal ta",
        "pedoman ta",
        "panduan ta",
        "template ta",
        "format ta",
        "template tugas akhir",
        "buku tugas akhir",
    )

    if any(term in text for term in kerja_praktik_terms):
        return "Kerja Praktik"

    if any(term in text for term in tugas_akhir_terms):
        return "Tugas Akhir"

    return "Belum Dikenali"


# =========================================================
# REGISTRY DOKUMEN
# =========================================================
def load_document_registry():
    """Membaca metadata dokumen dari document_registry.json."""
    if not DOCUMENT_REGISTRY_FILE.exists():
        return []

    try:
        data = json.loads(DOCUMENT_REGISTRY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_document_registry(documents):
    """Menyimpan metadata secara aman agar registry tidak mudah rusak."""
    temporary_file = DOCUMENT_REGISTRY_FILE.with_suffix(".tmp")
    temporary_file.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_file.replace(DOCUMENT_REGISTRY_FILE)


def list_upload_documents():
    """Mengambil seluruh file PDF dan DOCX yang berada di folder uploads."""
    ensure_project_folders()

    return sorted(
        [
            item.name
            for item in UPLOAD_FOLDER.iterdir()
            if item.is_file() and item.suffix.lower() in {".pdf", ".docx"}
        ],
        key=str.lower,
    )


def make_default_document_metadata(filename):
    """Membuat metadata awal untuk dokumen baru yang ditemukan."""
    title = display_title_from_filename(filename)
    now = datetime.datetime.now().isoformat()
    document_type = normalize_document_type(filename)

    return {
        "filename": filename,
        "title": title,
        "category": infer_document_category(filename, title),
        "document_type": document_type,
        "document_type_label": document_type_label(document_type),
        "uploaded_at": now,
        "updated_at": now,
        "status": "active",
    }


def sync_document_registry():
    """
    Menyinkronkan folder uploads dengan document_registry.json.

    PDF yang ditambah langsung dari file manager tetap terdeteksi saat
    Workspace dibuka. File DOCX dikenali sebagai template unduhan.
    """
    filenames = list_upload_documents()
    old_registry = load_document_registry()

    registry_by_filename = {
        str(item.get("filename")): item
        for item in old_registry
        if isinstance(item, dict) and item.get("filename")
    }

    updated_registry = []

    for filename in filenames:
        current = registry_by_filename.get(filename)
        document_type = normalize_document_type(filename)

        if not current:
            current = make_default_document_metadata(filename)
        else:
            current = dict(current)
            current["filename"] = filename

            title = str(current.get("title") or "").strip()
            if not title:
                title = display_title_from_filename(filename)

            current["title"] = title
            current["category"] = infer_document_category(filename, title)
            current["document_type"] = document_type
            current["document_type_label"] = document_type_label(document_type)
            current["status"] = "active"
            current.setdefault("uploaded_at", datetime.datetime.now().isoformat())
            current.setdefault("updated_at", datetime.datetime.now().isoformat())

        updated_registry.append(current)

    updated_registry.sort(
        key=lambda item: (
            item.get("category") == "Belum Dikenali",
            item.get("document_type") == "template",
            str(item.get("title", "")).lower(),
        )
    )

    save_document_registry(updated_registry)
    return updated_registry


def upsert_document_metadata(filename, title=""):
    """Menambah atau memperbarui metadata setelah upload berhasil."""
    documents = sync_document_registry()
    cleaned_title = str(title or "").strip()
    document_type = normalize_document_type(filename)

    for document in documents:
        if document.get("filename") == filename:
            document["title"] = (
                cleaned_title
                or document.get("title")
                or display_title_from_filename(filename)
            )
            document["category"] = infer_document_category(
                filename,
                document["title"],
            )
            document["document_type"] = document_type
            document["document_type_label"] = document_type_label(document_type)
            document["status"] = "active"
            document["updated_at"] = datetime.datetime.now().isoformat()
            save_document_registry(documents)
            return document

    metadata = make_default_document_metadata(filename)
    metadata["title"] = cleaned_title or metadata["title"]
    metadata["category"] = infer_document_category(filename, metadata["title"])
    documents.append(metadata)
    save_document_registry(documents)
    return metadata


def remove_document_metadata(filename):
    """Menghapus metadata setelah file dihapus dari Workspace."""
    current_registry = load_document_registry()
    save_document_registry(
        [item for item in current_registry if item.get("filename") != filename]
    )


# =========================================================
# DOKUMEN PEDOMAN DAN TEMPLATE UNTUK CHAT
# =========================================================
DOWNLOAD_CATEGORY_SLUGS = {
    "kerja-praktik": "Kerja Praktik",
    "tugas-akhir": "Tugas Akhir",
}

CATEGORY_DOWNLOAD_SLUGS = {
    category: slug
    for slug, category in DOWNLOAD_CATEGORY_SLUGS.items()
}


def _message_tokens(value):
    """Memecah pesan menjadi token sederhana untuk mendeteksi KP dan TA."""
    normalized = normalize_filename_text(value)
    return set(
        "".join(character if character.isalnum() else " " for character in normalized)
        .split()
    )


def _category_is_mentioned(normalized_message, category):
    """Mendeteksi kategori secara terpisah agar KP dan TA dapat terbaca bersamaan."""
    normalized = normalize_filename_text(normalized_message)
    tokens = _message_tokens(normalized)

    if category == "Kerja Praktik":
        phrases = (
            "kerja praktik",
            "kerja praktek",
            "praktik kerja",
            "praktek kerja",
            "magang",
            "internship",
            "pedoman kp",
            "panduan kp",
            "template kp",
            "proposal kp",
        )
        return "kp" in tokens or any(phrase in normalized for phrase in phrases)

    if category == "Tugas Akhir":
        phrases = (
            "tugas akhir",
            "tugas ahir",
            "skripsi",
            "sempro",
            "semhas",
            "sidang",
            "pedoman ta",
            "panduan ta",
            "template ta",
            "proposal ta",
            "buku tugas akhir",
        )
        return "ta" in tokens or any(phrase in normalized for phrase in phrases)

    return False


def is_requesting_both_categories(normalized_message):
    """
    Mendeteksi permintaan untuk kedua layanan sekaligus.

    Contoh yang didukung:
    - kedua template
    - kirim keduanya
    - semua pedoman
    - link KP dan TA
    - template Kerja Praktik dan Tugas Akhir
    """
    normalized = normalize_filename_text(normalized_message)
    tokens = _message_tokens(normalized)

    explicit_categories = [
        category
        for category in FOCUS_CATEGORIES
        if _category_is_mentioned(normalized, category)
    ]

    if len(explicit_categories) == len(FOCUS_CATEGORIES):
        return True

    both_markers = {
        "kedua",
        "keduanya",
        "semua",
        "both",
        "semuanya",
    }

    # "dua-duanya" menjadi "dua duanya" setelah normalisasi.
    has_double_marker = (
        "dua duanya" in normalized
        or "dua dua" in normalized
        or bool(tokens.intersection(both_markers))
    )

    request_context = (
        "template",
        "word",
        "docx",
        "link",
        "tautan",
        "url",
        "pedoman",
        "panduan",
        "buku",
        "dokumen",
        "file",
        "pdf",
        "berkas",
        "kirim",
        "unduh",
        "download",
    )

    return has_double_marker and any(term in normalized for term in request_context)


def get_requested_categories(message, conversation_id=None):
    """
    Mengambil satu atau dua kategori yang diminta.

    Prioritas:
    1. Bila KP dan TA disebut bersamaan, kembalikan keduanya.
    2. Bila pengguna meminta "keduanya", "kedua", atau "semua" pada
       konteks template/link/pedoman, kembalikan KP dan TA.
    3. Bila hanya satu kategori disebut, kembalikan kategori tersebut.
    4. Bila pesan adalah lanjutan, gunakan fokus obrolan sebelumnya.
    """
    normalized = normalize_filename_text(message)

    if not normalized:
        return []

    explicit_categories = [
        category
        for category in FOCUS_CATEGORIES
        if _category_is_mentioned(normalized, category)
    ]

    if len(explicit_categories) == len(FOCUS_CATEGORIES):
        return list(FOCUS_CATEGORIES)

    if is_requesting_both_categories(normalized):
        return list(FOCUS_CATEGORIES)

    if explicit_categories:
        return explicit_categories

    previous_category = get_previous_focus_category(conversation_id)

    if previous_category in FOCUS_CATEGORIES:
        return [previous_category]

    return []


def get_requested_category(message, conversation_id=None):
    """
    Pembungkus kompatibilitas untuk fungsi lama yang hanya membutuhkan satu kategori.
    """
    categories = get_requested_categories(message, conversation_id)
    return categories[0] if categories else None


def is_template_request_text(normalized_message):
    """Mengenali permintaan file Word template."""
    template_terms = (
        "template",
        "format word",
        "file word",
        "dokumen word",
        "word",
        "docx",
        "format dokumen",
    )
    return any(term in normalized_message for term in template_terms)


def detect_template_request(message, conversation_id=None):
    """
    Mendeteksi permintaan template Word.

    Mendukung satu atau dua kategori:
    - Tolong kirim template KP
    - Saya ingin template TA
    - Kirim kedua template
    """
    normalized = normalize_filename_text(message)

    if not normalized or not is_template_request_text(normalized):
        return []

    return get_requested_categories(message, conversation_id)


def is_link_request_text(normalized_message):
    """Mengenali permintaan tautan resmi."""
    link_terms = (
        "link",
        "tautan",
        "url",
        "website",
        "web resmi",
        "akses resmi",
    )
    return any(term in normalized_message for term in link_terms)


def detect_pedoman_link_request(message, conversation_id=None):
    """
    Mendeteksi permintaan tautan pedoman resmi.

    Mendukung satu atau dua kategori:
    - Berikan link pedoman KP
    - Link KP dan TA
    - Saya ingin kedua link pedoman
    """
    normalized = normalize_filename_text(message)

    if (
        not normalized
        or not is_link_request_text(normalized)
        or is_template_request_text(normalized)
    ):
        return []

    return get_requested_categories(message, conversation_id)


def get_official_pedoman_link(category):
    """
    Mengambil tautan resmi dari rag_logic.py tanpa membuat atau menebak URL baru.
    Fungsi mendukung struktur dictionary, list, maupun string agar tetap aman
    apabila format OFFICIAL_SERVICE_LINKS diperbarui.
    """
    entries = OFFICIAL_SERVICE_LINKS.get(category, [])

    if isinstance(entries, str):
        entries = [{"label": f"Pedoman {category}", "url": entries}]
    elif isinstance(entries, dict):
        entries = [entries]

    if not isinstance(entries, list):
        return None

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        url = str(entry.get("url", "")).strip()
        label = str(entry.get("label", "")).strip()

        if url:
            return {
                "label": label or f"Pedoman {category}",
                "url": url,
            }

    return None


def join_category_names(categories):
    """Membuat daftar kategori yang enak dibaca dalam respons chat."""
    names = [str(category) for category in categories if category]

    if not names:
        return ""

    if len(names) == 1:
        return names[0]

    if len(names) == 2:
        return f"{names[0]} dan {names[1]}"

    return f"{', '.join(names[:-1])}, dan {names[-1]}"


def build_pedoman_link_response(categories):
    """
    Membentuk jawaban untuk satu atau dua tautan resmi tanpa kartu PDF.
    """
    available_links = []
    unavailable_categories = []

    for category in categories:
        official_link = get_official_pedoman_link(category)

        if official_link:
            available_links.append((category, official_link))
        else:
            unavailable_categories.append(category)

    if not available_links:
        return (
            f"Maaf ya, tautan resmi pedoman {join_category_names(categories)} "
            "belum tersedia di SSC."
        )

    if len(available_links) == 1:
        category, official_link = available_links[0]
        response = (
            f"Berikut tautan resmi {official_link['label']}:\n"
            f"{official_link['url']}"
        )
    else:
        lines = ["Berikut tautan resmi pedoman yang kamu minta:"]
        for category, official_link in available_links:
            lines.extend([
                "",
                f"{category} — {official_link['label']}",
                official_link["url"],
            ])
        response = "\n".join(lines)

    if unavailable_categories:
        response += (
            f"\n\nTautan untuk {join_category_names(unavailable_categories)} "
            "belum tersedia di SSC."
        )

    return response


def detect_document_request(message, conversation_id=None):
    """
    Mendeteksi permintaan file pedoman PDF.

    Permintaan link dan template sengaja dikecualikan agar tidak salah
    mengirimkan kartu PDF.
    """
    normalized = normalize_filename_text(message)

    if (
        not normalized
        or is_template_request_text(normalized)
        or is_link_request_text(normalized)
    ):
        return []

    requested_categories = get_requested_categories(message, conversation_id)

    if not requested_categories:
        return []

    document_terms = (
        "pedoman",
        "pedomannya",
        "panduan",
        "panduannya",
        "dokumen",
        "dokumennya",
        "file",
        "filenya",
        "pdf",
        "buku",
        "bukunya",
        "berkas",
        "berkasnya",
        "download",
        "unduh",
        "kirim",
        "kirimkan",
        "berikan",
    )

    return (
        requested_categories
        if any(term in normalized for term in document_terms)
        else []
    )


def collect_documents(categories, finder):
    """
    Mengambil dokumen dari satu atau dua kategori dan memisahkan yang belum tersedia.
    """
    found_documents = []
    unavailable_categories = []

    for category in categories:
        document = finder(category)

        if document:
            found_documents.append((category, document))
        else:
            unavailable_categories.append(category)

    return found_documents, unavailable_categories


def build_attachment_response(
    document_kind,
    button_label,
    found_documents,
    unavailable_categories,
):
    """
    Membuat respons chat untuk kartu file PDF atau Word.

    document_kind: "pedoman" atau "template Word".
    """
    if not found_documents:
        requested = join_category_names(unavailable_categories)
        return f"Maaf ya, {document_kind} untuk {requested} belum tersedia untuk diunduh."

    if len(found_documents) == 1:
        _, document = found_documents[0]
        response = (
            f"Berikut {document.get('title', document_kind)} yang kamu minta. "
            f"Tekan tombol {button_label} untuk menyimpannya."
        )
    else:
        response = (
            f"Berikut kedua {document_kind} yang kamu minta. "
            f"Tekan tombol {button_label} pada setiap kartu untuk menyimpannya."
        )

    if unavailable_categories:
        response += (
            f" {document_kind.capitalize()} untuk "
            f"{join_category_names(unavailable_categories)} belum tersedia."
        )

    return response


def find_document_by_category(category, document_type):
    """Mengambil dokumen aktif sesuai kategori dan jenisnya."""
    candidates = []

    for document in sync_document_registry():
        filename = str(document.get("filename", "")).strip()
        title = str(document.get("title", "")).strip()
        file_path = UPLOAD_FOLDER / filename

        if (
            document.get("status") != "active"
            or document.get("category") != category
            or document.get("document_type") != document_type
            or not filename
            or not file_path.is_file()
        ):
            continue

        expected_extension = ".docx" if document_type == "template" else ".pdf"
        if file_extension(filename) != expected_extension:
            continue

        searchable_text = normalize_filename_text(f"{title} {filename}")
        priority_keywords = (
            ("template", "format")
            if document_type == "template"
            else ("pedoman", "panduan", "buku")
        )
        priority = 0 if any(
            keyword in searchable_text for keyword in priority_keywords
        ) else 1

        candidates.append((priority, title.lower(), document))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def find_pedoman_document(category):
    """Mengambil satu pedoman PDF aktif dari kategori yang diminta."""
    return find_document_by_category(category, "pedoman")


def find_template_document(category):
    """Mengambil satu template Word aktif dari kategori yang diminta."""
    return find_document_by_category(category, "template")


def make_document_attachment(document):
    """Menyiapkan metadata kartu download untuk chat.html."""
    category = str(document.get("category", "")).strip()
    slug = CATEGORY_DOWNLOAD_SLUGS.get(category)
    document_type = str(document.get("document_type", "pedoman")).strip()

    if not slug:
        return None

    is_template = document_type == "template"

    return {
        "title": document.get("title") or "Dokumen SSC",
        "category": category,
        "file_type": "DOCX" if is_template else "PDF",
        "attachment_type": "template" if is_template else "pedoman",
        "download_label": "Unduh Template Word" if is_template else "Unduh PDF",
        "download_url": url_for(
            "download_template" if is_template else "download_pedoman",
            category_slug=slug,
        ),
    }


@app.route("/download/pedoman/<category_slug>")
def download_pedoman(category_slug):
    """Mengirim pedoman PDF sebagai file unduhan."""
    category = DOWNLOAD_CATEGORY_SLUGS.get(category_slug)
    if not category:
        abort(404)

    document = find_pedoman_document(category)
    if not document:
        abort(404)

    filename = str(document.get("filename", "")).strip()
    file_path = UPLOAD_FOLDER / filename
    if not filename or not file_path.is_file():
        abort(404)

    return send_from_directory(
        str(UPLOAD_FOLDER),
        filename,
        as_attachment=True,
        download_name=filename,
    )


@app.route("/download/template/<category_slug>")
def download_template(category_slug):
    """Mengirim template Word DOCX sebagai file unduhan."""
    category = DOWNLOAD_CATEGORY_SLUGS.get(category_slug)
    if not category:
        abort(404)

    document = find_template_document(category)
    if not document:
        abort(404)

    filename = str(document.get("filename", "")).strip()
    file_path = UPLOAD_FOLDER / filename
    if not filename or not file_path.is_file():
        abort(404)

    return send_from_directory(
        str(UPLOAD_FOLDER),
        filename,
        as_attachment=True,
        download_name=filename,
    )


# =========================================================
# HALAMAN PUBLIK
# =========================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat")
def chat_page():
    return render_template("chat.html")


# =========================================================
# LOGIN ADMIN
# =========================================================
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if session.get("admin_logged_in"):
        return redirect(url_for("workspace_page"))

    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session.clear()
            session["admin_logged_in"] = True
            session["admin_username"] = username
            return redirect(url_for("workspace_page"))

        error = "Username atau password salah."

    return render_template("admin_login.html", error=error)


@app.route("/admin/logout", methods=["POST"])
@login_required
def admin_logout():
    session.clear()
    return redirect(url_for("index"))


# =========================================================
# WORKSPACE ADMIN
# =========================================================
@app.route("/workspace")
@login_required
def workspace_page():
    documents = sync_document_registry()

    return render_template(
        "workspace.html",
        documents=documents,
        files=[item["filename"] for item in documents],
        document_categories=FOCUS_CATEGORIES,
        document_types=DOCUMENT_TYPES,
        status=request.args.get("status", ""),
        message=request.args.get("message", ""),
        admin_username=session.get("admin_username", "admin"),
    )


@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    ensure_project_folders()

    if "dataset_file" not in request.files:
        return redirect(
            url_for(
                "workspace_page",
                status="error",
                message="Form upload tidak menemukan file yang dikirim.",
            )
        )

    uploaded_file = request.files["dataset_file"]

    if not uploaded_file or not uploaded_file.filename:
        return redirect(
            url_for(
                "workspace_page",
                status="error",
                message="Silakan pilih file PDF atau Word terlebih dahulu.",
            )
        )

    if not allowed_file(uploaded_file.filename):
        return redirect(
            url_for(
                "workspace_page",
                status="error",
                message="Format file belum sesuai. Gunakan file PDF atau Word DOCX.",
            )
        )

    filename = secure_filename(uploaded_file.filename)
    if not filename:
        return redirect(
            url_for(
                "workspace_page",
                status="error",
                message="Nama file tidak valid.",
            )
        )

    destination = UPLOAD_FOLDER / filename
    uploaded_file.save(destination)

    document_title = request.form.get("document_title", "").strip()
    document = upsert_document_metadata(filename, document_title)

    # DOCX hanya disimpan sebagai template download, sehingga tidak mengubah RAG.
    if is_knowledge_source(filename):
        try:
            build_vector_store()
        except Exception as error:
            return redirect(
                url_for(
                    "workspace_page",
                    status="error",
                    message=(
                        f"Pedoman {filename} sudah tersimpan dan muncul di Workspace, "
                        f"tetapi informasi AI belum berhasil diperbarui: {error}"
                    ),
                )
            )

        success_message = (
            f"Pedoman {filename} berhasil tersinkronisasi dan informasi SSC sudah diperbarui."
        )
    else:
        success_message = (
            f"Template Word {document.get('title', filename)} berhasil disimpan. "
            "Template siap dikirim kepada mahasiswa melalui chat."
        )

    return redirect(
        url_for(
            "workspace_page",
            status="success",
            message=success_message,
        )
    )


@app.route("/delete/<filename>")
@login_required
def delete_file(filename):
    safe_filename = os.path.basename(filename)
    file_path = UPLOAD_FOLDER / safe_filename

    if not file_path.exists():
        sync_document_registry()
        return redirect(
            url_for(
                "workspace_page",
                status="error",
                message="File tidak ditemukan. Daftar Workspace sudah disinkronkan ulang.",
            )
        )

    needs_rag_rebuild = is_knowledge_source(safe_filename)
    document_type = normalize_document_type(safe_filename)
    file_path.unlink()
    remove_document_metadata(safe_filename)

    if needs_rag_rebuild:
        try:
            build_vector_store()
        except Exception as error:
            return redirect(
                url_for(
                    "workspace_page",
                    status="error",
                    message=(
                        "Pedoman sudah dihapus dari Workspace, tetapi informasi AI "
                        f"belum berhasil diperbarui: {error}"
                    ),
                )
            )
        success_message = (
            f"Pedoman {safe_filename} berhasil dihapus dan knowledge base sudah diperbarui."
        )
    else:
        success_message = (
            f"Template Word {safe_filename} berhasil dihapus dari Workspace."
            if document_type == "template"
            else f"Dokumen {safe_filename} berhasil dihapus dari Workspace."
        )

    return redirect(
        url_for(
            "workspace_page",
            status="success",
            message=success_message,
        )
    )


# =========================================================
# API CHAT
# =========================================================
@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(silent=True) or {}

    user_message = str(payload.get("message", "")).strip()
    conversation_id = str(payload.get("conversation_id", "")).strip()

    if not user_message:
        return jsonify({
            "error": "Silakan tuliskan pertanyaan terlebih dahulu."
        }), 400

    try:
        attachments = []

        # 1. Permintaan link resmi selalu diprioritaskan.
        # Contoh:
        # - "Link pedoman KP dan TA"
        # - "Saya ingin kedua link"
        requested_link_categories = detect_pedoman_link_request(
            user_message,
            conversation_id,
        )

        if requested_link_categories:
            response_text = build_pedoman_link_response(
                requested_link_categories
            )
            save_chat_history(
                user_message,
                response_text,
                conversation_id,
            )

        # 2. Permintaan template Word.
        # Contoh:
        # - "Kirim template KP"
        # - "Saya ingin kedua template"
        else:
            requested_template_categories = detect_template_request(
                user_message,
                conversation_id,
            )

            if requested_template_categories:
                found_documents, unavailable_categories = collect_documents(
                    requested_template_categories,
                    find_template_document,
                )

                for _, document in found_documents:
                    attachment = make_document_attachment(document)
                    if attachment:
                        attachments.append(attachment)

                response_text = build_attachment_response(
                    "template Word",
                    "Unduh Template Word",
                    found_documents,
                    unavailable_categories,
                )

                save_chat_history(
                    user_message,
                    response_text,
                    conversation_id,
                )

            # 3. Permintaan file pedoman PDF.
            # Contoh:
            # - "Kirim file pedoman KP"
            # - "Saya ingin kedua pedoman"
            else:
                requested_document_categories = detect_document_request(
                    user_message,
                    conversation_id,
                )

                if requested_document_categories:
                    found_documents, unavailable_categories = collect_documents(
                        requested_document_categories,
                        find_pedoman_document,
                    )

                    for _, document in found_documents:
                        attachment = make_document_attachment(document)
                        if attachment:
                            attachments.append(attachment)

                    response_text = build_attachment_response(
                        "pedoman",
                        "Unduh PDF",
                        found_documents,
                        unavailable_categories,
                    )

                    # Permintaan file melewati RAG sehingga riwayat disimpan manual.
                    save_chat_history(
                        user_message,
                        response_text,
                        conversation_id,
                    )

                # 4. Pertanyaan biasa tetap diproses oleh RAG.
                else:
                    response_text = get_response_from_rag(
                        user_message,
                        conversation_id,
                    )

        return jsonify({
            "response": response_text,
            "attachments": attachments,
        })

    except Exception:
        app.logger.exception("Terjadi error saat memproses chat SSC.")
        return jsonify({
            "error": (
                "Terjadi kendala sementara saat memproses pesan. "
                "Silakan coba lagi beberapa saat."
            )
        }), 500


@app.errorhandler(413)
def file_too_large(error):
    """Pesan lebih jelas saat admin mengunggah PDF terlalu besar."""
    return redirect(
        url_for(
            "workspace_page",
            status="error",
            message="Ukuran dokumen terlalu besar. Maksimum ukuran file adalah 25 MB.",
        )
    )


if __name__ == "__main__":
    ensure_project_folders()
    app.run(debug=True)
