import cv2

# Charger le classificateur Haar pour la détection de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Charger la vidéo
video_capture = cv2.VideoCapture('vid3.mp4')  # Remplacez 'vid3.mp4' par 0 pour utiliser une caméra en temps réel

# Facteur de réduction pour l'affichage
scale_factor = 0.2  # Réduire la taille de la fenêtre à 50%

while True:
    # Lire une image de la vidéo
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Appliquer un prétraitement pour améliorer la détection
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Réduction du bruit
    gray = cv2.equalizeHist(gray)  # Amélioration du contraste

    # Détecter les visages dans l'image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=7, minSize=(50, 50)
    )

    for (x, y, w, h) in faces:
        if w > 50 and h > 50:  # Taille minimale des visages
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


    # Redimensionner le cadre pour un affichage plus petit
    resized_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

    # Afficher le cadre redimensionné
    cv2.imshow('Detection de visages (Haar)', resized_frame)

    # Arrêter la capture si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
video_capture.release()
cv2.destroyAllWindows()
