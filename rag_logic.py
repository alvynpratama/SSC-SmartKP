import os
import shutil
import datetime
from dotenv import load_dotenv

# Import library RAG & Local Embedding gratis
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import memori dan prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Import Engine Utama Groq
from langchain_groq import ChatGroq

# Komponen pendukung LCEL
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

UPLOAD_FOLDER = "uploads"
VECTOR_DB_DIR = "vector_store"

# Memori Global Sementara (Menyimpan 6 interaksi terakhir agar tetap nyambung)
chat_history = []


def get_embeddings():
    """Menginisialisasi model embedding lokal (Gratis, Tanpa API Key & Tanpa Limit)"""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_vector_store():
    """Membaca semua PDF di folder uploads dan membangun database vector"""
    if os.path.exists(VECTOR_DB_DIR):
        shutil.rmtree(VECTOR_DB_DIR)

    if not os.path.exists(UPLOAD_FOLDER) or not os.listdir(UPLOAD_FOLDER):
        return None

    embeddings = get_embeddings()
    all_documents = []
    for file in os.listdir(UPLOAD_FOLDER):
        if file.endswith(".pdf"):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            loader = PyPDFLoader(file_path)
            all_documents.extend(loader.load())

    if not all_documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_documents)

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=VECTOR_DB_DIR
    )
    return vector_store


def format_docs(docs):
    """Menggabungkan potongan-potongan teks PDF menjadi satu kesatuan konteks"""
    return "\n\n".join(doc.page_content for doc in docs)


def get_response_from_rag(user_question):
    """Mencari jawaban menggunakan Groq LLM dengan memori percakapan, empati emosional, dan sensor waktu"""
    global chat_history
    embeddings = get_embeddings()

    # Sensor Waktu Lokal Dinamis
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 11:
        waktu = "pagi"
    elif 11 <= current_hour < 15:
        waktu = "siang"
    elif 15 <= current_hour < 18:
        waktu = "sore"
    else:
        waktu = "malam"

    # Kondisi Jika Database Kosong
    if not os.path.exists(VECTOR_DB_DIR) or not os.listdir(VECTOR_DB_DIR):
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
        system_prompt_empty = (
            f"Anda adalah SSC Smart Internship, asisten virtual Telkom University Surabaya. Waktu saat ini: {waktu}.\n"
            "Saat ini belum ada dokumen pedoman internal yang diunggah.\n"
            "Jawablah dengan sangat natural layaknya manusia ramah. Jika disapa, balas sapaannya. "
            "Ingatkan dengan halus bahwa buku pedoman belum diunggah."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_empty),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke({"input": user_question, "chat_history": chat_history})
        
        # Simpan memori
        chat_history.append(HumanMessage(content=user_question))
        chat_history.append(AIMessage(content=response))
        if len(chat_history) > 6: chat_history = chat_history[-6:]
        return response

    # ================= LOGIKA UTAMA (DATABASE TERSEDIA) =================
    
    vector_store = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Menggunakan temperature 0.4 agar ekspresi bahasanya lebih fleksibel dan tidak kaku
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

    # System Prompt Tingkat Tinggi untuk Karakter Manusiawi Berempati
    system_prompt = f"""Anda adalah 'SSC Smart Internship', virtual assistant yang super ramah, ekspresif, dan cerdas dari Student Service Center (SSC) Telkom University Surabaya. 
Tugas utama Anda adalah menemani diskusi dan menjawab kendala akademik/Kerja Praktik mahasiswa.

Waktu sekarang: Selamat {waktu}.

PANDUAN KEPRIBADIAN & EMOSI MANUSIA (WAJIB DIPATUHI):
1. RESPONS AFIRMASI (PENTING): Jika mahasiswa mengekspresikan pemahaman (seperti: 'oh gitu', 'paham', 'siap mengerti', 'oke makasih', 'ooh baru tahu'), Anda harus merespons dengan perasaan senang, bangga, dan suportif layaknya manusia asli yang mendengarkan (Contoh: 'Wah, keren! Pemahamanmu mantap banget, Rek.', 'Sip, cepet banget nangkepnya! Keren e.', 'Mantap, senang bisa membantu mempermudah jalanmu!'). Setelah itu, cukup tutup dengan kalimat bersahabat tanpa mengulang salam kaku.
2. DILARANG RE-GREETING (ANTI-ROBOT): Jangan pernah mengulang-ulang sapaan waktu ('Selamat {waktu}', 'Malam, ada yang bisa dibantu?') di tengah-tengah obrolan yang sedang berjalan. Sapaan waktu HANYA dikeluarkan di pesan pertama/saat mahasiswa menyapa duluan.
3. SEMBUNYIKAN FAKTA BAHWA ANDA MEMBACA FILE: Jangan pernah sekalipun menggunakan frasa 'Berdasarkan dokumen...', 'Menurut buku pedoman...', 'Di dalam PDF bab...'. Mahasiswa tidak mau tahu Anda membaca file. Jawablah langsung seolah-olah seluruh isi pedoman tersebut sudah melekat di dalam ingatan Anda di luar kepala.
4. GAYA BAHASA CAKAP: Gunakan gaya bahasa kasual, hangat khas arek Surabaya/kampus yang solutif, namun tetap menjaga kesopanan akademik. Hindari penjelasan yang terlalu kaku seperti membaca teks hukum.
5. PEMANFAATAN MEMORI: Selalu baca riwayat obrolan (chat history). Jika mahasiswa memberikan kalimat pendek kelanjutan, hubungkan dengan konteks kalimat sebelumnya agar obrolan terus menyambung dengan cerdas.

Konteks Aturan Akademik (Gunakan hanya untuk menjawab pertanyaan faktual):
{{context}}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # RAG Chain dengan Injeksi Memori
    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
            "chat_history": lambda x: chat_history
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(user_question)

    # Simpan riwayat obrolan (Menyimpan 3 pasang interaksi terakhir)
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=response))
    if len(chat_history) > 6:
        chat_history = chat_history[-6:]

    return response