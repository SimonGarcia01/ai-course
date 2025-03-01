'''
Milton Orlando Sarria
Ejemplo para capturar rostros usando la camara web
primero activa la camara y para cada clase toma cierta cantidad de fotos
al tomar una foto recorta el rostro detectado y lo guarda en una nueva carpeta
con el nombre del cliente
'''

import cv2
import numpy as np
import os


# Inicializar el clasificador de detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Inicializar la cámara
cap = cv2.VideoCapture(0)

destino='data/';
new_dim = (50,50)
continuar = True
fotos_por_clase = int(input("Numero de fotos por clase?: "))


while continuar:
    k=0
    clase = input("Nombre de la clase: ")
    path_clase = destino + clase
    if not os.path.exists(path_clase):
        print("Creando el directorio para: '{}' ".format(path_clase))
        os.makedirs(path_clase)
    while(k<fotos_por_clase): 
    
        # Capturar fotograma de la cámara
        ret, frame = cap.read()    
        # Convertir a escala de grises para la detección de rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
        # Detectar rostros en el fotograma
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
        # Iterar sobre cada rostro detectado
        for (x, y, w, h) in faces:
            # Recortar el rostro de la imagen original
            face_img = frame[y:y+h, x:x+w]
        
            # Redimensionar el rostro a 50x50 píxeles
            face_resized = cv2.resize(face_img, new_dim)
        
            # Guardar el rostro redimensionado en un archivo
            cv2.imwrite(path_clase+'/img_{}.jpg'.format(k), face_resized)
            k=k+1
            
            # Dibujar un rectángulo alrededor del rostro detectado en la imagen original
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
        # Mostrar la imagen original con los rostros detectados
        cv2.imshow('Detección de rostros', frame)
        key =  cv2.waitKey(1)
        if key == 27 :
            break
    cv2.destroyAllWindows()
    cont = input("Desea registrar otra clase? (S/N): ")
    if cont.upper()=='N':
        continuar = False
# Liberar la captura y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
