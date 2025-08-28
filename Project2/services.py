import os
import numpy as np
import tensorflow as tf
from PIL import Image
from config import BASE_DIR, CLASS_NAMES, CLEAN_CLASS_NAMES, UPLOAD_FOLDER

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# ==============================================================================
# PEMUATAN MODEL
# ==============================================================================
print("="*30)
print("MEMUAT MODEL DEEP LEARNING...")
try:
    # Model "Penjaga Gerbang" untuk deteksi objek umum
    gatekeeper_model = ResNet50(weights='imagenet')
    print("MODEL PENJAGA GERBANG (ResNet50) BERHASIL DIMUAT.")

    # Model Klasifikasi Penyakit
    MODEL_PATH_MOBILENET = os.path.join(BASE_DIR, 'models', 'mobilenet_v2_825-125-5.h5')
    MODEL_PATH_EFFICIENTNET = os.path.join(BASE_DIR, 'models', 'efficientnet_v2m_825-125-5.h5')
    MODEL_PATH_RESNET = os.path.join(BASE_DIR, 'models', 'resnet101_825-125-5.h5')

    mobilenet_model = tf.keras.models.load_model(MODEL_PATH_MOBILENET)
    efficientnet_model = tf.keras.models.load_model(MODEL_PATH_EFFICIENTNET)
    resnet_model = tf.keras.models.load_model(MODEL_PATH_RESNET)
    print("SEMUA MODEL KLASIFIKASI PENYAKIT BERHASIL DIMUAT.")
except Exception as e:
    print(f"ERROR SAAT MEMUAT MODEL: {e}")
    gatekeeper_model, mobilenet_model, efficientnet_model, resnet_model = None, None, None, None
print("="*30)

# --- Logika Penjaga Gerbang Hibrida ---

# Aturan 1: DENYLIST (Penolakan Langsung)
# Jika salah satu dari ini terdeteksi dengan keyakinan cukup, gambar PASTI ditolak.
DENYLIST_KEYWORDS = [
    # Manusia & Hewan
    'person', 'face', 'man', 'woman', 'hand', 'hair', 'foot', 'leg', 'arm', 'cat', 'dog', 'animal',
    # Teknologi & Kantor
    'computer', 'keyboard', 'screen', 'monitor', 'mouse', 'laptop', 'remote_control', 'book', 'paper', 'document', 'television', 
    # Pakaian & Kain
    'jean', 'shirt', 'clothing', 'fabric', 'textile', 'shoe', 'wig', 'hat',
    # Kendaraan
    'car', 'vehicle', 'wheel', 'truck',
    # Benda Dalam & Luar Ruangan
    'building', 'table', 'chair', 'desk', 'wall', 'floor', 'ceiling', 'bottle', 'cup', 'shop', 'iron', 'wardrobe', 
    'medicine_chest', 'sofa', 'couch', 'bed', 'lamp', 'curtain', 'plate', 'bowl', 'knife', 'fork'
]

# Aturan 2: ALLOWLIST (Kriteria Penerimaan)
# Jika lolos DENYLIST, salah satu dari ini HARUS ada agar gambar diterima.
ALLOWLIST_KEYWORDS = [
    # Bagian Tanaman & Tanaman Umum (Inti)
    'leaf', 'plant', 'flower', 'vine', 'garden', 'vegetable', 'foliage', 'stem', 'branch',
    # Konteks Spesifik Tomat & Lingkungan
    'tomato', 'greenhouse', 'pot', 'planter', 'soil', 'trellis',
    # Kategori Umum yang Relevan
    'fruit', 'produce',
    #tambahan
    'knot', 'vase', 'shoji', 'pedestal'
]

# ==============================================================================
# FUNGSI HELPER (LOGIKA PREDIKSI)
# ==============================================================================

def is_image_a_leaf(image_path):
    """
    Menggunakan ResNet50 dengan logika hibrida yang disempurnakan 
    (Pemeriksaan Kecerahan + Pencocokan Kata Utuh + OVERRIDE + DENYLIST + ALLOWLIST).
    """
    if not gatekeeper_model:
        return True # Lewati jika model gagal dimuat

    try:
        img = Image.open(image_path)

        # --- Aturan -1: Pemeriksaan Kualitas Pencahayaan ---
        grayscale_img = img.convert('L')
        brightness = np.mean(np.array(grayscale_img))
        
        # Ambang batas kecerahan (0=hitam, 255=putih)
        MIN_BRIGHTNESS = 50
        MAX_BRIGHTNESS = 220

        if brightness < MIN_BRIGHTNESS:
            print(f"BRIGHTNESS CHECK FAILED: Image is too dark (Brightness: {brightness:.2f}). REJECTING.")
            return False
        if brightness > MAX_BRIGHTNESS:
            print(f"BRIGHTNESS CHECK FAILED: Image is too bright (Brightness: {brightness:.2f}). REJECTING.")
            return False

        img_rgb = img.convert('RGB')
        img_resized = img_rgb.resize((224, 224))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        processed_img = preprocess_input(img_array)

        predictions = gatekeeper_model.predict(processed_img)
        decoded_predictions = decode_predictions(predictions, top=5)[0]

        print(f"Gatekeeper Predictions: {[(p[1], f'{p[2]*100:.2f}%') for p in decoded_predictions]}")

        top_denylist_confidence = 0
        top_allowlist_confidence = 0

        for _, label, confidence in decoded_predictions:
            # --- PERBAIKAN: Gunakan pencocokan kata utuh ---
            label_words = set(label.lower().split('_'))
            
            if any(keyword in label_words for keyword in DENYLIST_KEYWORDS):
                top_denylist_confidence = max(top_denylist_confidence, confidence)
            if any(keyword in label_words for keyword in ALLOWLIST_KEYWORDS):
                top_allowlist_confidence = max(top_allowlist_confidence, confidence)
        
        # --- Aturan 0: Pengecualian (Override) ---
        if top_allowlist_confidence > 0.7 and top_allowlist_confidence > (top_denylist_confidence * 2):
            print(f"OVERRIDE RULE TRIGGERED: Allowlist confidence ({top_allowlist_confidence:.2f}) outweighs denylist ({top_denylist_confidence:.2f}). ACCEPTING.")
            return True

        # --- Aturan 1: Pemeriksaan DENYLIST ---
        if top_denylist_confidence > 0.30:
            print(f"DENYLIST RULE TRIGGERED: Denylist confidence at {top_denylist_confidence:.2f}. REJECTING.")
            return False

        # --- Aturan 2: Pemeriksaan ALLOWLIST ---
        if top_allowlist_confidence > 0.05:
            print(f"ALLOWLIST RULE TRIGGERED: Allowlist confidence at {top_allowlist_confidence:.2f}. ACCEPTING.")
            return True

        # --- Aturan 3: Default Tolak ---
        print("DEFAULT REJECT: Image did not trigger denylist, but no allowed keywords were found.")
        return False
        
    except Exception as e:
        print(f"Error during gatekeeper check: {e}")
        return False # Fail-safe yang lebih aman



def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_prediction_analysis(pred_mobilenet, pred_efficientnet, pred_resnet):
    """
    Menganalisis prediksi dari tiga model untuk memberikan hasil yang lebih komprehensif.
    - Menghitung skor rata-rata per kelas.
    - Mengidentifikasi 3 prediksi teratas.
    - Menghitung "Skor Konflik" (standar deviasi) untuk prediksi teratas.
    """
    # 1. Hitung skor rata-rata untuk setiap kelas
    average_confidences = (pred_mobilenet + pred_efficientnet + pred_resnet) / 3
    
    # 2. Dapatkan 3 indeks teratas dari skor rata-rata
    top_3_indices = np.argsort(average_confidences)[-3:][::-1]
    
    # 3. Siapkan daftar untuk hasil teratas
    top_results = []
    for i in top_3_indices:
        top_results.append({
            "name": CLEAN_CLASS_NAMES[i],
            "score": round(float(average_confidences[i]) * 100, 2)
        })
        
    # 4. Hitung Skor Konflik untuk prediksi teratas
    top_pred_index = top_3_indices[0]
    confidences_for_top_pred = [
        pred_mobilenet[top_pred_index],
        pred_efficientnet[top_pred_index],
        pred_resnet[top_pred_index]
    ]
    conflict_score = np.std(confidences_for_top_pred) * 100  # Jadikan persentase

    # Pastikan top_results memiliki 3 elemen, isi dengan None jika kurang
    while len(top_results) < 3:
        top_results.append({"name": "N/A", "score": 0})

    return {
        "top_prediction": top_results[0],
        "secondary_prediction": top_results[1],
        "tertiary_prediction": top_results[2],
        "conflict_score": round(float(conflict_score), 2)
    }

def get_models():
    # Sekarang kembalikan semua 4 model
    return gatekeeper_model, mobilenet_model, efficientnet_model, resnet_model

# ==============================================================================
# DATA PENANGANAN
# ==============================================================================
penanganan_data = {
    "Tomato Early blight": {
        "indonesian_name": "Hawar Daun",
        "deskripsi": "Penyakit ini disebabkan oleh jamur Alternaria linariae (sebelumnya dikenal sebagai A. solani) dan pertama kali terlihat pada tanaman berupa bintik-bintik kecil berwarna cokelat, terutama pada daun yang lebih tua. Bintik-bintik tersebut membesar dan lingkaran konsentris dalam pola seperti mata banteng mungkin terlihat di tengah area yang terinfeksi. Jaringan di sekitar bintik-bintik tersebut mungkin berubah warna menjadi kuning. Jika suhu tinggi dan kelembapan tinggi terjadi pada saat ini, sebagian besar daun akan mati. Lesi pada batang serupa dengan yang terdapat pada daun dan kadang-kadang mengelilingi tanaman jika terjadi dekat garis tanah (collar rot). Pada buah, lesi dapat mencapai ukuran yang cukup besar, biasanya melibatkan hampir seluruh buah. Lingkaran konsentris juga terdapat pada buah. Buah yang terinfeksi seringkali jatuh. Jamur ini bertahan hidup pada sisa-sisa tanaman yang terinfeksi di tanah, pada biji, pada tanaman tomat liar, dan inang solanaceae lainnya, seperti kentang Irlandia, terong, dan black nightshade (gulma umum yang terkait).",
        "penanganan": "Gunakan varietas tomat yang tahan atau toleran terhadap penyakit. Gunakan benih yang bebas patogen dan jangan tanam tanaman yang terinfeksi di lapangan. Lakukan rotasi tanaman, basmi gulma dan tanaman tomat liar, jaga jarak antar tanaman agar tidak saling bersentuhan, tutupi tanaman dengan mulsa, berikan pupuk secara tepat, hindari menyiram daun tomat dengan air irigasi, dan jaga agar tanaman tumbuh dengan vigor. Potong dan buang cabang dan daun bawah yang terinfeksi. Untuk mengurangi keparahan penyakit, uji tanah kebun setiap tahun dan jaga tingkat kalium yang cukup. Tambahkan kapur ke tanah sesuai hasil uji tanah. Berikan pupuk kalsium nitrat secara bulanan pada tanaman tomat untuk pertumbuhan yang optimal. Jika penyakit cukup parah untuk memerlukan pengendalian kimia, pilih salah satu fungisida berikut: mancozeb (sangat baik); chlorothalonil atau fungisida tembaga (baik). Ikuti petunjuk pada label.",
        "slug": "tomato-early-blight", "image_url": "https://via.placeholder.com/400x200?text=Early+Blight",
        "external_link": "https://hgic.clemson.edu/factsheet/tomato-diseases-disorders/"
    },
    "Tomato Bacterial spot": {
        "indonesian_name": "Bercak Bakteri",
        "deskripsi": "Penyakit ini disebabkan oleh beberapa spesies bakteri Xanthomonas (utama oleh Xanthomonas perforans), yang menginfeksi tomat hijau tetapi tidak tomat merah. Cabai juga terinfeksi. Penyakit ini lebih sering terjadi selama musim hujan. Kerusakan pada tanaman meliputi bercak pada daun dan buah, yang mengakibatkan penurunan hasil panen, gugur daun, dan buah yang terbakar sinar matahari. Gejala penyakit ini berupa bintik-bintik kecil, berbentuk sudut hingga tidak beraturan, dan basah pada daun, serta bintik-bintik yang sedikit menonjol hingga bersisik pada buah. Bintik-bintik pada daun mungkin memiliki halo kuning. Pusat bintik-bintik tersebut mengering dan seringkali pecah. Bakteri ini bertahan selama musim dingin pada tanaman tomat liar dan sisa-sisa tanaman yang terinfeksi. Cuaca lembap mendukung perkembangan penyakit ini.",
        "penanganan": "Hanya gunakan benih dan tanaman yang telah disertifikasi bebas penyakit. Hindari area yang ditanami cabai atau tomat pada tahun sebelumnya. Hindari penyiraman dari atas dengan menggunakan sistem irigasi tetes atau irigasi parit. Buang dan buang semua material tanaman yang terinfeksi penyakit. Potong tanaman untuk meningkatkan sirkulasi udara. Penyemprotan dengan fungisida tembaga akan memberikan pengendalian yang cukup baik terhadap penyakit bakteri.",
        "slug": "tomato-bacterial-spot", "image_url": "https://via.placeholder.com/400x200?text=Bacterial+Spot",
        "external_link": "https://hgic.clemson.edu/factsheet/tomato-diseases-disorders/"
    },
    "Tomato Late blight": {
        "indonesian_name": "Busuk Daun",
        "deskripsi": "Late blight adalah penyakit serius yang menyerang kentang dan tomat, disebabkan oleh patogen jamur air Phytophthora infestans. Penyakit ini terutama merusak tanaman pada cuaca sejuk dan lembap. Patogen ini dapat menyerang semua bagian tanaman. Lesi pada daun muda berukuran kecil dan tampak sebagai bintik-bintik gelap yang basah. Bintik-bintik ini akan cepat membesar, dan jamur putih akan muncul di tepi area yang terinfeksi pada permukaan bawah daun. Defoliasi total (pengeringan dan layu daun dan batang) dapat terjadi dalam 14 hari sejak gejala pertama muncul. Buah tomat yang terinfeksi mengembangkan bercak berkilau, gelap, atau berwarna zaitun, yang dapat menutupi area yang luas.",
        "penanganan": "Jaga agar daun tetap kering. Letakkan kebun Anda di tempat yang mendapat sinar matahari pagi. Berikan ruang ekstra antara tanaman, dan hindari penyiraman dari atas, terutama pada sore hari. Beli benih dan tanaman yang telah disertifikasi bebas penyakit. Hancurkan tanaman tomat dan kentang liar, serta gulma keluarga nightshade. Jangan komposkan kentang busuk yang dibeli dari toko. Cabut dan hancurkan tanaman yang terinfeksi. Jika penyakit cukup parah untuk memerlukan pengendalian kimia, pilih salah satu fungisida berikut: chlorothalonil (sangat baik), fungisida tembaga, atau mancozeb (baik). Tanam varietas yang tahan penyakit.",
        "slug": "tomato-late-blight", "image_url": "https://via.placeholder.com/400x200?text=Late+Blight",
        "external_link": "https://hgic.clemson.edu/factsheet/tomato-diseases-disorders/"
    },
    "Tomato Leaf Mold": {
        "indonesian_name": "Jamur Daun",
        "deskripsi": "Jamur Passalora fulva menyebabkan busuk daun. Jamur ini pertama kali ditemukan pada daun tua di dekat tanah, di mana sirkulasi udara buruk dan kelembapan tinggi. Gejala awal berupa bintik-bintik hijau pucat atau kekuningan pada permukaan atas daun, yang kemudian membesar dan berubah menjadi kuning khas. Dalam kondisi lembap, bintik-bintik pada permukaan bawah daun ditutupi oleh pertumbuhan abu-abu berbulu halus dari spora yang dihasilkan oleh jamur. Ketika infeksi parah, bintik-bintik tersebut menyatu, dan daun mati. Kadang-kadang, jamur menyerang batang, bunga, dan buah.",
        "penanganan": "Sisa tanaman harus dibersihkan dari ladang. Pemasangan tiang dan pemangkasan untuk meningkatkan sirkulasi udara dapat membantu mengendalikan penyakit. Jarak tanam tanaman tomat lebih jauh agar sirkulasi udara antara tanaman lebih baik. Hindari membasahi daun saat penyiraman. Rotasi tanam dengan sayuran selain tomat. Penggunaan program fungisida pencegahan dengan chlorothalonil, mancozeb, atau fungisida tembaga dapat mengendalikan penyakit.",
        "slug": "tomato-leaf-mold", "image_url": "https://via.placeholder.com/400x200?text=Leaf+Mold",
        "external_link": "https://hgic.clemson.edu/factsheet/tomato-diseases-disorders/"
    },
    "Tomato Septoria Leaf Spot": {
        "indonesian_name": "Bercak Daun Septoria",
        "deskripsi": "Penyakit merusak pada daun, tangkai daun, dan batang tomat (buah tidak terinfeksi) ini disebabkan oleh jamur Septoria lycopersici. Infeksi biasanya terjadi pada daun bagian bawah dekat tanah, setelah tanaman mulai berbunga. Berbagai bintik kecil berbentuk lingkaran dengan pinggiran gelap mengelilingi pusat berwarna krem muncul pada daun yang lebih tua. Bintik-bintik hitam kecil, yang merupakan tubuh pembentuk spora, dapat dilihat di pusat bintik-bintik tersebut. Daun yang sangat bercak akan menguning, mati, dan rontok dari tanaman.",
        "penanganan": "Sebagian besar varietas tomat yang ditanam saat ini rentan terhadap penyakit bercak daun Septoria. Rotasi tanaman selama 3 tahun dan sanitasi (pembuangan sisa tanaman) akan mengurangi jumlah inokulum. Jangan gunakan irigasi overhead. Penggunaan fungisida secara berulang dengan chlorothalonil (sangat baik) atau fungisida tembaga, atau mancozeb (baik) akan mengendalikan penyakit tersebut.",
        "slug": "tomato-septoria-leaf-spot", "image_url": "https://via.placeholder.com/400x200?text=Septoria+Leaf+Spot",
        "external_link": "https://hgic.clemson.edu/factsheet/tomato-diseases-disorders/"
    },
    "Tomato Spider Mites": {
        "indonesian_name": "Tungau Laba-laba",
        "deskripsi": "Kutu laba-laba bercak dua adalah spesies kutu laba-laba yang paling umum menyerang tanaman sayuran dan buah-buahan di New England. Kutu laba-laba dapat ditemukan pada tanaman tomat, terong, kentang, tanaman merambat seperti melon, mentimun, dan tanaman lainnya. Tungau laba-laba bercak dua merupakan salah satu hama terpenting pada terong. Mereka dapat memiliki hingga 20 generasi per tahun dan disukai oleh kelebihan nitrogen serta kondisi kering dan berdebu. Wabah sering disebabkan oleh penggunaan insektisida spektrum luas yang mengganggu musuh alami yang membantu mengendalikan populasi tungau.",
        "penanganan": "Hindari ladang yang berumput liar dan jangan menanam terong di dekat tanaman pakan legum. Hindari penggunaan insektisida spektrum luas pada awal musim untuk hama lainnya. Jangan berlebihan dalam pemupukan. Wabah hama dapat memburuk akibat pemupukan nitrogen yang berlebihan. Irigasi overhead atau periode hujan yang berkepanjangan dapat membantu mengurangi populasi hama. Untuk pengendalian, gunakan produk selektif seperti bifenazate, abamectin, spirotetramat, atau spiromesifen. Produk organik (OMRI) meliputi sabun insektisida, minyak neem, atau minyak kedelai. Lakukan 2 aplikasi dengan selang waktu 5-7 hari.",
        "slug": "tomato-spider-mites", "image_url": "https://via.placeholder.com/400x200?text=Spider+Mites",
        "external_link": "https://www.umass.edu/agriculture-food-environment/vegetable/fact-sheets/two-spotted-spider-mite"
    },
    "Tomato Target Spot": {
        "indonesian_name": "Bercak / Bintik Target",
        "deskripsi": "Bintik target pada tomat disebabkan oleh patogen jamur Corynespora cassiicola. Penyakit ini terjadi pada tomat yang ditanam di lapangan di daerah tropis dan subtropis di seluruh dunia. Infeksi bintik target mengurangi hasil panen secara tidak langsung dengan mengurangi area fotosintesis dan secara langsung dengan mengurangi daya jual buah melalui bintik-bintik pada buah. Patogen C. cassiicola memiliki rentang inang yang luas, menginfeksi lebih dari 500 spesies tanaman. Jamur ini berfungsi sebagai nekrotrof (membunuh jaringan saat menginfeksi), saprofit (bertahan hidup pada sisa-sisa tanaman), dan epifit (tumbuh di atas tetapi tidak menginfeksi jaringan tanaman).",
        "penanganan": "Praktik budidaya meliputi peningkatan sirkulasi udara, menghindari pemupukan berlebihan, pemangkasan, mengelola gulma, dan rotasi tanaman. Aplikasi rutin fungisida adalah strategi utama. Produk yang mengandung chlorothalonil, mancozeb, copper oxychloride, azoxystrobin, pyraclostrobin, dan boscalid telah terbukti memberikan pengendalian yang baik.",
        "slug": "tomato-target-spot", "image_url": "https://via.placeholder.com/400x200?text=Target+Spot",
        "external_link": "https://www.vegetables.bayer.com/ca/en-ca/resources/agronomic-spotlights/target-spot-of-tomato.html"
    },
    "Tomato Mosaic Virus": {
        "indonesian_name": "Virus Mosaik",
        "deskripsi": "ToMV adalah galur spesifik TMV yang terutama menginfeksi tanaman tomat. Virus menginfeksi tanaman dengan memasuki luka mikroskopis di dalam sel tanaman. TMV dan ToMV dapat bertahan hidup di sisa-sisa tanaman yang terinfeksi seperti daun dan akar, serta dapat tetap dorman hingga dua tahun. TMV jarang menyebabkan seluruh tanaman inang mati, tetapi virus dapat sangat mengurangi kualitas dan hasil buah. Gejala virus ini bervariasi tergantung pada tahap infeksi, tanaman inang, strain virus, dan kondisi lingkungan.",
        "penanganan": "Tidak ada obat untuk tanaman yang terinfeksi virus. Praktik pengelolaan pencegahan dan budidaya sangat diperlukan. Sterilkan semua alat, peralatan, dan permukaan kerja sebelum penanaman. Buang semua tanaman terinfeksi dan tanah terkontaminasi. Lakukan rotasi tanaman dengan tanaman non-inang. Gunakan varietas tomat yang tahan terhadap virus (mengandung gen ketahanan Tm-1, Tm-2, dan Tm-2²). Kurangi gulma di dalam dan sekitar ladang.",
        "slug": "tomato-mosaic-virus", "image_url": "https://via.placeholder.com/400x200?text=Mosaic+Virus",
        "external_link": "https://content.ces.ncsu.edu/tobamoviruses-that-affect-tomato-tmv-tomv-tobrfv"
    },
    "Tomato Yellow Leaf Curl Virus": {
        "indonesian_name": "Virus Keriting Daun Kuning",
        "deskripsi": "Virus Kuning Daun Tomat (TYLCV) tidak ditularkan melalui benih, tetapi ditularkan oleh kutu putih. Penyakit ini sangat merusak hasil panen. Setelah terinfeksi, tanaman tomat mungkin tidak menunjukkan gejala selama 2 hingga 3 minggu. Gejala meliputi daun yang melengkung ke atas, tepi daun kuning (klorotik), daun yang lebih kecil dari normal, pertumbuhan tanaman terhambat, dan gugur bunga. Jika tanaman tomat terinfeksi pada tahap awal pertumbuhannya, mungkin tidak akan terbentuk buah.",
        "penanganan": "Pengangkatan tanaman yang menunjukkan gejala awal dapat memperlambat penyebaran penyakit. Jaga agar gulma terkendali. Mulsa reflektif (berwarna aluminium atau perak) dapat digunakan untuk mengurangi aktivitas makan kutu putih. Semprotan dengan konsentrasi rendah minyak hortikultura atau minyak canola dapat berfungsi sebagai pengusir kutu putih. Pada akhir musim, buang semua tanaman yang rentan dan bakar atau buanglah. Gunakan varietas tomat yang tahan terhadap Virus Kuning Daun Tomat.",
        "slug": "tomato-yellow-leaf-curl-virus", "image_url": "https://via.placeholder.com/400x200?text=Yellow+Leaf+Curl",
        "external_link": "https://hgic.clemson.edu/factsheet/tomato-diseases-disorders/"
    },
    "Tomato Fusarium": {
        "indonesian_name": "Layu Fusarium",
        "deskripsi": "Ini adalah penyakit yang disebabkan oleh jamur Fusarium oxysporum dan umumnya terjadi pada cuaca hangat. Gejala awal penyakit pada tanaman kecil adalah layu dan mengeringnya daun bagian bawah disertai hilangnya warna hijau, diikuti oleh layu dan kematian tanaman. Seringkali, daun di satu sisi batang terlebih dahulu berubah menjadi kuning keemasan. Batang tanaman yang layu tidak menunjukkan pembusukan lunak, tetapi ketika dipotong memanjang, batang bagian bawah akan menunjukkan perubahan warna coklat gelap pada pembuluh pengangkut air. Jamur ini berasal dari tanah dan menyebar ke atas dari akar ke sistem pengangkut air batang.",
        "penanganan": "Untuk pengendalian, tanam tanaman di tanah bebas patogen, gunakan bibit bebas penyakit, dan tanam hanya varietas yang memiliki ketahanan minimal terhadap ras 1 dan 2 Fusarium wilt (ditandai dengan FF setelah nama varietas tomat). Meningkatkan pH tanah menjadi 6,5–7,0 dan menggunakan nitrogen nitrat (seperti dalam kalsium nitrat) daripada nitrogen amonia akan menghambat perkembangan penyakit. Tidak ada pengendalian kimia yang tersedia.",
        "slug": "tomato-fusarium", "image_url": "https://via.placeholder.com/400x200?text=Fusarium+Wilt",
        "external_link": "https://hgic.clemson.edu/factsheet/tomato-diseases-disorders/"
    },
    "Tomato Healthy": {
        "indonesian_name": "Tomat Sehat",
        "deskripsi": "Tanaman tomat yang sehat menunjukkan daun hijau cerah, batang kokoh, dan pertumbuhan yang vigor tanpa tanda-tanda penyakit atau hama.",
        "penanganan": "Pertahankan praktik pertanian yang baik: penyiraman teratur, pemupukan seimbang, pencahayaan cukup, dan pengendalian hama preventif. Lakukan pemangkasan untuk sirkulasi udara yang baik.",
        "slug": "tomato-healthy", "image_url": "https://via.placeholder.com/400x200?text=Healthy+Tomato",
        "external_link": "#"
    }
}