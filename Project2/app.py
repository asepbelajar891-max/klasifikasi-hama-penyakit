import os
from flask import Flask
import logging

# Import konfigurasi dari config.py
from config import UPLOAD_FOLDER, DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS, SECRET_KEY
from extensions import db, login_manager # Import db dan login_manager dari extensions.py
from models import User # Import User model

# Inisialisasi Flask App
app = Flask(__name__)

# Konfigurasi aplikasi dari config.py
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 # Batas file 10 MB


# Inisialisasi ekstensi
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'main.login' # Memberitahu LoginManager halaman login ada di blueprint 'main' dengan fungsi 'login'
login_manager.login_message = "Silakan login untuk mengakses halaman ini."
login_manager.login_message_category = "info"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Pastikan folder uploads ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import dan daftarkan blueprint dari routes.py
from routes import main_bp
app.register_blueprint(main_bp)

# Import model database agar terdaftar oleh SQLAlchemy
from models import Riwayat

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        logging.info("Database tables created/checked.")
    app.run(host='0.0.0.0', debug=True)
