import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SECRET_KEY = os.urandom(24)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads')
DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'database.db')
SQLALCHEMY_TRACK_MODIFICATIONS = False

CLASS_NAMES = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Fusarium', 'Tomato_Healthy',
    'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Mosaic_Virus',
    'Tomato_Septoria_Leaf_Spot', 'Tomato_Spider_Mites', 'Tomato_Target_Spot',
    'Tomato_Yellow_Leaf_Curl_Virus'
]
CLEAN_CLASS_NAMES = [name.replace('_', ' ') for name in CLASS_NAMES]

# Definisikan month_map di sini agar bisa diakses secara global
MONTH_MAP = {
    'January': 'Januari', 'February': 'Februari', 'March': 'Maret', 'April': 'April',
    'May': 'Mei', 'June': 'Juni', 'July': 'Juli', 'August': 'Agustus',
    'September': 'September', 'October': 'Oktober', 'November': 'November', 'December': 'Desember'
}
