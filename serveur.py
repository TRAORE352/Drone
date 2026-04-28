from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import numpy as np
import cv2
from ultralytics import YOLO
import datetime
import io
from PIL import Image

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10*1024*1024)

# Charger le modèle une seule fois
modele = YOLO('best.pt')

@app.route('/')
def index():
    return render_template('index.html')

# Recevoir une image depuis un téléphone et l'analyser
@socketio.on('analyser_image')
def analyser_image(data):
    try:
        # Décoder l'image base64
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image_pil = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image_pil)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Analyser avec YOLOv8
        resultats = modele(image_cv, conf=0.4, verbose=False)

        defaut_trouve = None
        confiance_max = 0

        for r in resultats:
            for boite in r.boxes:
                defaut = modele.names[int(boite.cls)]
                confiance = float(boite.conf)
                if confiance > confiance_max:
                    confiance_max = confiance
                    defaut_trouve = defaut

        # Répondre au téléphone
        emit('resultat_analyse', {
            'defaut': defaut_trouve,
            'confiance': round(confiance_max * 100, 1)
        })

        # Si anomalie — envoyer à tout le monde
        if defaut_trouve and defaut_trouve != 'normal':
            frame_annote = resultats[0].plot()
            chemin = 'detection_temp.jpg'
            cv2.imwrite(chemin, frame_annote)
            envoyer_alerte(defaut_trouve, confiance_max,
                          datetime.datetime.now().strftime('%H:%M:%S'), chemin)

    except Exception as e:
        print(f"Erreur analyse : {e}")
        emit('resultat_analyse', {'defaut': None, 'confiance': 0})

def envoyer_alerte(defaut, confiance, heure, image_path):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    socketio.emit('nouvelle_alerte', {
        'defaut': defaut,
        'confiance': round(confiance * 100, 1),
        'heure': heure,
        'image': image_data
    })

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)