import json
import os
from datetime import datetime, timedelta
from flask import request, jsonify, render_template, Blueprint, current_app, flash, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import or_
import numpy as np
from collections import Counter
import logging

from flask_login import login_user, logout_user, login_required, current_user
from models import db, User, Riwayat
# Import fungsi baru is_image_a_leaf
from services import get_models, preprocess_image, get_prediction_analysis, penanganan_data, is_image_a_leaf
from config import UPLOAD_FOLDER, CLEAN_CLASS_NAMES, MONTH_MAP

main_bp = Blueprint('main', __name__)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main_bp.route('/')
def index():
    return render_template('index.html')

# --- Rute Autentikasi ---
@main_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash('Password konfirmasi tidak cocok!', 'danger')
            return redirect(url_for('main.register'))

        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username sudah ada. Silakan pilih yang lain.', 'warning')
            return redirect(url_for('main.register'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Akun berhasil dibuat! Silakan login.', 'success')
        return redirect(url_for('main.login'))
    return render_template('register.html')

@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if not user or not check_password_hash(user.password, password):
            flash('Login gagal. Periksa kembali username dan password Anda.', 'danger')
            return redirect(url_for('main.login'))
        
        login_user(user, remember=True)
        return redirect(url_for('main.dashboard'))
    return render_template('login.html')

@main_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))

# --- Rute Dasbor ---
@main_bp.route('/dashboard')
@login_required
def dashboard():
    # Mengambil 5 riwayat terbaru untuk ditampilkan di dashboard
    riwayat_list = Riwayat.query.filter_by(user_id=current_user.id).order_by(Riwayat.timestamp.desc()).limit(5).all()
    
    # Lakukan analisis pada data riwayat yang akan ditampilkan
    for history in riwayat_list:
        if isinstance(history.detailed_results, str):
            details = json.loads(history.detailed_results)
        else:
            details = history.detailed_results
        
        if details:
            pred_mobilenet = np.array(details.get("MobileNetV2", [])) / 100.0
            pred_efficientnet = np.array(details.get("EfficientNetV2M", [])) / 100.0
            pred_resnet = np.array(details.get("ResNet101", [])) / 100.0
            
            if pred_mobilenet.size > 0:
                analysis = get_prediction_analysis(pred_mobilenet, pred_efficientnet, pred_resnet)
                history.feedback = get_qualitative_feedback(analysis["top_prediction"]["score"], analysis["conflict_score"])
            else:
                history.feedback = {"label": "Data Tidak Lengkap", "alert_class": "alert-secondary"}
        else:
            history.feedback = {"label": "Data Tidak Lengkap", "alert_class": "alert-secondary"}
        
        wib_timestamp = history.timestamp + timedelta(hours=7)
        history.formatted_date = wib_timestamp.strftime('%d %B %Y, %H:%M WIB').replace(wib_timestamp.strftime('%B'), MONTH_MAP[wib_timestamp.strftime('%B')])

    # Statistik keseluruhan
    all_riwayat = Riwayat.query.filter_by(user_id=current_user.id).all()
    total_uploads = len(all_riwayat)
    
    if all_riwayat:
        all_predictions = [r.prediction for r in all_riwayat]
        most_common_disease = Counter(all_predictions).most_common(1)[0][0]
    else:
        most_common_disease = None

    return render_template('dashboard.html', 
                           total_uploads=total_uploads,
                           last_disease=riwayat_list[0].prediction if riwayat_list else None,
                           most_common_disease=most_common_disease,
                           riwayat_list=riwayat_list)

# --- Rute Aplikasi Inti (Diproteksi) ---
@main_bp.route('/klasifikasi')
@login_required
def klasifikasi():
    return render_template('klasifikasi.html')

def get_qualitative_feedback(score, conflict_score):
    """Memberikan label kualitatif dan pesan peringatan berdasarkan skor dan konflik."""
    if score >= 90:
        label = "Kesesuaian Sangat Tinggi"
        alert_class = "alert-success"
    elif score >= 70:
        label = "Kesesuaian Tinggi"
        alert_class = "alert-primary"
    elif score >= 60:
        label = "Kesesuaian Sedang"
        alert_class = "alert-primary"
    else: # score < 60
        label = "Kesesuaian Rendah"
        alert_class = "alert-warning"

    message = f"<strong>{label}.</strong> "
    if conflict_score > 30: # Ambang batas konflik (bisa disesuaikan)
        message += "Namun, model kami mendeteksi beberapa kemungkinan gejala. Verifikasi manual sangat disarankan. Pastikan gambar jelas dan coba lagi jika perlu."
        alert_class = "alert-danger"
    elif score < 70:
        message += "Hasil ini mungkin kurang akurat. Pastikan gambar jelas dan coba lagi jika perlu."

    return {"label": label, "message": message, "alert_class": alert_class}

@main_bp.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'File tidak ditemukan'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Tipe file tidak valid'}), 400

    # Ambil semua 4 model
    gatekeeper_model, mobilenet_model, efficientnet_model, resnet_model = get_models()
    if not all([gatekeeper_model, mobilenet_model, efficientnet_model, resnet_model]):
        return jsonify({'error': 'Model tidak siap'}), 500

    filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        file.save(filepath)

        # --- LANGKAH 1: Pemeriksaan oleh Penjaga Gerbang ---
        if not is_image_a_leaf(filepath):
            os.remove(filepath) # Hapus file yang tidak valid
            logging.info(f"Image {filename} rejected by gatekeeper.")
            return jsonify({
                "status": "not_a_leaf",
                "message": "Objek yang terdeteksi bukan daun. Silakan unggah gambar daun tomat."
            })

        # --- LANGKAH 2: Lanjutkan ke klasifikasi penyakit jika lolos ---
        logging.info(f"Image {filename} passed gatekeeper. Proceeding with classification.")
        processed_image = preprocess_image(filepath)
        pred_mobilenet = mobilenet_model.predict(processed_image)[0]
        pred_efficientnet = efficientnet_model.predict(processed_image)[0]
        pred_resnet = resnet_model.predict(processed_image)[0]

        analysis_results = get_prediction_analysis(pred_mobilenet, pred_efficientnet, pred_resnet)
        
        top_prediction = analysis_results["top_prediction"]

        # --- Logika Ambang Batas Ketidakpastian yang Ditingkatkan ---
        # Dinyatakan tidak pasti jika skor terlalu rendah ATAU jika skor sedang namun konflik antar model tinggi.
        score = analysis_results["top_prediction"]["score"]
        conflict = analysis_results["conflict_score"]
        
        if score < 40 or (score < 65 and conflict > 20):
            # Jangan hapus file di sini, karena mungkin pengguna ingin melihatnya
            message = f"Tidak Dapat Diidentifikasi. Skor kecocokan (Score: {score:.1f}%) atau kesepakatan antar model (Conflict: {conflict:.1f}) terlalu rendah. Pastikan gambar jelas, fokus, dan diambil dalam pencahayaan yang baik."
            return jsonify({
                "status": "uncertain",
                "message": message,
                "image_path": os.path.join('static/uploads', filename).replace("/")
            })

        # --- Dapatkan Label Kualitatif ---
        feedback = get_qualitative_feedback(top_prediction["score"], analysis_results["conflict_score"])

        # --- Simpan ke Riwayat (hanya prediksi utama) ---
        image_db_path = os.path.join('static/uploads', filename).replace("\\", "/")
        detailed_results_full = {
            "MobileNetV2": [round(float(c) * 100, 2) for c in pred_mobilenet],
            "EfficientNetV2M": [round(float(c) * 100, 2) for c in pred_efficientnet],
            "ResNet101": [round(float(c) * 100, 2) for c in pred_resnet]
        }
        new_history = Riwayat(
            filename=file.filename,
            prediction=top_prediction["name"],
            confidence=top_prediction["score"],
            image_path=image_db_path,
            detailed_results=json.dumps(detailed_results_full),
            user_id=current_user.id
        )
        db.session.add(new_history)
        db.session.commit()

        penanganan_info = penanganan_data.get(top_prediction["name"], {})
        penanganan_slug = penanganan_info.get('slug', '')
        indonesian_name = penanganan_info.get('indonesian_name', '')

        return jsonify({
            "status": "success",
            "analysis": analysis_results,
            "feedback": feedback,
            "penanganan_slug": penanganan_slug,
            "indonesian_name": indonesian_name,
            "image_path": image_db_path,
            "original_filename": file.filename,
            "riwayat_id": new_history.id
        })

    except Exception as e:
        # Jika terjadi error, pastikan file yang mungkin sudah tersimpan dihapus
        if os.path.exists(filepath):
            os.remove(filepath)
        logging.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({'error': f'Terjadi kesalahan saat prediksi: {str(e)}'}), 500


@main_bp.route('/penanganan')
def penanganan_index():
    return render_template('penanganan.html', data=penanganan_data)

@main_bp.route('/riwayat')
@login_required
def riwayat():
    query = request.args.get('query', '')
    sort_by = request.args.get('sort_by', 'timestamp')
    sort_order = request.args.get('sort_order', 'desc')

    histories_query = Riwayat.query.filter_by(user_id=current_user.id)

    if query:
        search_pattern = f"%{query}%"
        histories_query = histories_query.filter(or_(
            Riwayat.filename.ilike(search_pattern),
            Riwayat.prediction.ilike(search_pattern)
        ))

    order_by_column = getattr(Riwayat, sort_by, Riwayat.timestamp)
    if sort_order == 'asc':
        histories_query = histories_query.order_by(order_by_column.asc())
    else:
        histories_query = histories_query.order_by(order_by_column.desc())

    histories_from_db = histories_query.all()
    
    for history in histories_from_db:
        # Ubah string JSON menjadi dictionary langsung pada objek
        if isinstance(history.detailed_results, str):
            history.detailed_results = json.loads(history.detailed_results)
        
        # Tambahkan analisis dan feedback untuk konsistensi
        if history.detailed_results:
            pred_mobilenet = np.array(history.detailed_results.get("MobileNetV2", [])) / 100.0
            pred_efficientnet = np.array(history.detailed_results.get("EfficientNetV2M", [])) / 100.0
            pred_resnet = np.array(history.detailed_results.get("ResNet101", [])) / 100.0
            
            if pred_mobilenet.size > 0: # Pastikan ada data untuk dianalisis
                history.analysis = get_prediction_analysis(pred_mobilenet, pred_efficientnet, pred_resnet)
                history.feedback = get_qualitative_feedback(history.analysis["top_prediction"]["score"], history.analysis["conflict_score"])
            else:
                history.analysis = None
                history.feedback = {"label": "Data Tidak Lengkap", "alert_class": "alert-secondary"}
        else:
            history.analysis = None
            history.feedback = {"label": "Data Tidak Lengkap", "alert_class": "alert-secondary"}

        # Tambahkan nama Indonesia
        history.indonesian_name = penanganan_data.get(history.prediction, {}).get('indonesian_name', '')

        wib_timestamp = history.timestamp + timedelta(hours=7)
        history.formatted_date = wib_timestamp.strftime('%d %B %Y, %H:%M WIB').replace(wib_timestamp.strftime('%B'), MONTH_MAP[wib_timestamp.strftime('%B')])

    return render_template('riwayat.html', histories=histories_from_db, query=query, sort_by=sort_by, sort_order=sort_order)

@main_bp.route('/riwayat/<int:riwayat_id>')
@login_required
def riwayat_detail(riwayat_id):
    history = Riwayat.query.filter_by(id=riwayat_id, user_id=current_user.id).first_or_404()
    
    # Ubah string JSON menjadi dictionary langsung pada objek
    if isinstance(history.detailed_results, str):
        history.detailed_results = json.loads(history.detailed_results)

    # Lakukan analisis dan feedback untuk detail view
    if history.detailed_results:
        pred_mobilenet = np.array(history.detailed_results.get("MobileNetV2", [])) / 100.0
        pred_efficientnet = np.array(history.detailed_results.get("EfficientNetV2M", [])) / 100.0
        pred_resnet = np.array(history.detailed_results.get("ResNet101", [])) / 100.0

        if pred_mobilenet.size > 0:
            history.analysis = get_prediction_analysis(pred_mobilenet, pred_efficientnet, pred_resnet)
            history.feedback = get_qualitative_feedback(history.analysis["top_prediction"]["score"], history.analysis["conflict_score"])
            history.penanganan_slug = penanganan_data.get(history.analysis["top_prediction"]["name"], {}).get('slug', '')
        else:
            history.analysis = None
            history.feedback = {"label": "Data Tidak Lengkap", "message": "Data detail prediksi tidak ditemukan.", "alert_class": "alert-secondary"}
            history.penanganan_slug = ''
    else:
        history.analysis = None
        history.feedback = {"label": "Data Tidak Lengkap", "message": "Data detail prediksi tidak ditemukan.", "alert_class": "alert-secondary"}
        history.penanganan_slug = ''

    wib_timestamp = history.timestamp + timedelta(hours=7)
    history.formatted_date = wib_timestamp.strftime('%d %B %Y, %H:%M WIB').replace(wib_timestamp.strftime('%B'), MONTH_MAP[wib_timestamp.strftime('%B')])
    
    penanganan_info = penanganan_data.get(history.prediction)

    return render_template('detail_riwayat.html', history=history, penanganan_info=penanganan_info, CLEAN_CLASS_NAMES=CLEAN_CLASS_NAMES)

@main_bp.route('/riwayat/delete/<int:riwayat_id>', methods=['POST'])
@login_required
def delete_riwayat(riwayat_id):
    history = Riwayat.query.get_or_404(riwayat_id)
    
    # Pastikan pengguna hanya bisa menghapus riwayat miliknya sendiri
    if history.author.id != current_user.id:
        return jsonify({'status': 'error', 'message': 'Akses ditolak.'}), 403
    
    try:
        # Hapus file gambar terkait jika ada
        if history.image_path:
            image_file_path = os.path.join(current_app.root_path, history.image_path)
            if os.path.exists(image_file_path):
                os.remove(image_file_path)
                
        db.session.delete(history)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Riwayat berhasil dihapus.'})
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error deleting history: {e}")
        return jsonify({'error': 'Gagal menghapus riwayat.'}), 500
