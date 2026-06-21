from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
# Import fungsi dari file rag_logic.py yang baru kita buat
from rag_logic import build_vector_store, get_response_from_rag

app = Flask(__name__)
app.secret_key = 'supersecretkeytelkom'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/workspace')
def workspace_page():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    status = request.args.get('status', None)
    message = request.args.get('message', None)
    return render_template('workspace.html', files=pdf_files, status=status, message=message)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'dataset_file' not in request.files:
        return redirect(url_for('workspace_page', status='error', message='Form file tidak ditemukan.'))
    file = request.files['dataset_file']
    if file.filename == '':
        return redirect(url_for('workspace_page', status='error', message='Tidak ada file yang dipilih.'))
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # TRIGGER RAG ENGINE: Proses ulang dokumen menjadi otak AI secara otomatis
        build_vector_store()
        
        return redirect(url_for('workspace_page', status='success', message=f'File {filename} berhasil diproses ke dalam memori AI!'))
    else:
        return redirect(url_for('workspace_page', status='error', message='Gagal! Format file harus berupa PDF.'))

@app.route('/delete/<filename>')
def delete_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        
        # TRIGGER RAG ENGINE: Perbarui ulang memori AI setelah file dihapus
        build_vector_store()
        
        return redirect(url_for('workspace_page', status='success', message=f'Dataset {filename} berhasil dihapus.'))
    return redirect(url_for('workspace_page', status='error', message='File tidak ditemukan.'))

# Tampilan Halaman Utama Chatbot
@app.route('/chat')
def chat_page():
    return render_template('chat.html')

# API EndPoint untuk memproses pesan Chatbot (Dijalankan di background oleh JavaScript)
@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json()
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'Pesan kosong'}), 400
        
    try:
        # Panggil logika RAG untuk mendapatkan jawaban pintar Gemini
        ai_response = get_response_from_rag(user_message)
        return jsonify({'response': ai_response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)