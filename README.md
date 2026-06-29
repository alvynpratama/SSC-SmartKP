# SSC Smart Assistant

SSC Smart Assistant merupakan sistem layanan informasi akademik berbasis *Artificial Intelligence* yang dikembangkan untuk mendukung layanan Student Service Center di Telkom University Surabaya.

Sistem ini membantu mahasiswa memperoleh informasi akademik melalui chatbot berbasis dokumen. Mahasiswa dapat mengajukan pertanyaan menggunakan bahasa yang lebih natural, meminta tautan resmi, mengunduh pedoman, serta memperoleh template dokumen melalui satu halaman chat.

SSC Smart Assistant dirancang secara universal. Admin SSC dapat menambahkan atau memperbarui dokumen sesuai kebutuhan layanan akademik. Pada implementasi awal, sistem menggunakan dokumen Kerja Praktik dan Tugas Akhir sebagai sampel basis pengetahuan chatbot.

## Link Chatbot

https://ssc-telu-ai.duckdns.org

## Laporan Final Project

[Lihat Laporan Final Project](LAPORAN_TUBES_AI_KELOMPOK_2.pdf)

## Dataset dan Dokumen Sampel

Dataset serta dokumen sampel tersedia pada folder [`uploads`](uploads/).

| No. | Dokumen                         | Format | Fungsi dalam Sistem        |
| --- | ------------------------------- | ------ | -------------------------- |
| 1   | Pedoman Kerja Praktik           | PDF    | Sumber informasi chatbot   |
| 2   | Pedoman Tugas Akhir             | PDF    | Sumber informasi chatbot   |
| 3   | Template Proposal Kerja Praktik | DOCX   | Template unduhan mahasiswa |
| 4   | Template Buku Tugas Akhir       | DOCX   | Template unduhan mahasiswa |

Dokumen PDF digunakan sebagai basis pengetahuan chatbot. Dokumen DOCX digunakan sebagai template yang dapat dikirimkan kepada mahasiswa melalui halaman chat.

## Fitur Utama

* Chatbot layanan informasi akademik berbasis dokumen.
* Pertanyaan dapat diajukan menggunakan bahasa yang lebih natural.
* Jawaban chatbot disusun menggunakan pendekatan *Retrieval-Augmented Generation* atau RAG.
* Pengiriman tautan resmi sesuai layanan yang diminta mahasiswa.
* Unduhan pedoman dalam format PDF.
* Unduhan template dalam format Word atau DOCX.
* Dukungan permintaan lebih dari satu dokumen sekaligus.
* Riwayat percakapan pada browser.
* Tombol bantuan lanjutan melalui WhatsApp.
* Admin Workspace untuk mengunggah, melihat, dan menghapus dokumen.

## Konsep Sistem

SSC Smart Assistant memiliki dua jenis pengguna:

1. **Mahasiswa**
   Mahasiswa dapat mengajukan pertanyaan melalui halaman chat, meminta tautan resmi, mengunduh pedoman PDF, dan memperoleh template Word.

2. **Admin SSC**
   Admin dapat mengelola dokumen melalui Admin Workspace. Dokumen PDF digunakan sebagai sumber informasi chatbot, sedangkan dokumen DOCX digunakan sebagai template unduhan.

## Teknologi yang Digunakan

| Komponen                      | Teknologi                                        |
| ----------------------------- | ------------------------------------------------ |
| Antarmuka Website             | HTML, CSS, JavaScript, Bootstrap, Jinja Template |
| Backend                       | Python dan Flask                                 |
| Pemrosesan Dokumen            | LangChain dan PyPDF                              |
| Embedding                     | HuggingFace `all-MiniLM-L6-v2`                   |
| Vector Store                  | ChromaDB                                         |
| Model Artificial Intelligence | Groq API                                         |
| Application Server            | Gunicorn                                         |
| Reverse Proxy                 | Nginx                                            |
| Infrastruktur                 | VPS Ubuntu                                       |
| Domain dan HTTPS              | DuckDNS dan Certbot                              |

## Struktur Project

```text
ssc-chatbot-rag/
│
├── LAPORAN_TUBES_AI_KELOMPOK_2.pdf
├── link website.txt
├── README.md
│
├── app.py
├── rag_logic.py
├── requirements.txt
├── document_registry.json
├── .gitignore
│
├── static/                  # Aset tampilan website
├── templates/               # Halaman HTML sistem
└── uploads/                 # Dataset, pedoman, dan template
```

## Cara Menjalankan Sistem Secara Lokal

1. Pastikan Python telah terpasang pada perangkat.

2. Buat dan aktifkan *virtual environment*.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Instal seluruh kebutuhan library.

```powershell
pip install -r requirements.txt
```

4. Buat file `.env` pada folder utama project, kemudian masukkan API key Groq.

```env
GROQ_API_KEY=masukkan_api_key_groq_di_sini
```

5. Jalankan aplikasi Flask.

```powershell
python -m flask --app app run --debug --no-reload
```

6. Buka sistem melalui browser.

```text
http://127.0.0.1:5000
```

## Catatan

* File `.env` tidak diunggah ke GitHub karena berisi API key.
* Folder `venv`, `model_cache`, `vector_store`, dan `__pycache__` tidak diunggah karena merupakan file hasil instalasi atau proses sistem.
* Dokumen pada folder `uploads` dapat diperbarui melalui Admin Workspace.
* Saat pedoman PDF diperbarui, basis pengetahuan chatbot akan menggunakan dokumen aktif terbaru.

## Anggota Kelompok

| No. | Nama                        | NIM        |
| --- | --------------------------- | ---------- |
| 1   | Alvyn Wira Pratama          | 1204230012 |
| 2   | Elvina Angelina Kurniawan   | 1204230067 |
| 3   | Ridho Nasrullah             | 1204230085 |
| 4   | Joe Petra Kusuma Adi        | 1204230041 |
| 5   | Muhammad Raihan Rizky P. H. | 1204230072 |

## Institusi

Program Studi Sistem Informasi
Fakultas Direktorat Surabaya
Telkom University Surabaya
2026
