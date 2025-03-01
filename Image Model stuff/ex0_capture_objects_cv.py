'''
Milton Orlando Sarria
Ejemplo para capturar objetos usando la camara web
primero activa la camara y para cada clase toma cierta cantidad de fotos
al tomar la guarda en una nueva carpeta
con el nombre de la clase

pip install opencv-python
'''

import cv2
import numpy as np
import os


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
        # Convertir a escala de grises 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    
        # Redimensionar NxN píxeles
        gray = cv2.resize(gray, new_dim)
    
        # Guardar  redimensionado en un archivo
        cv2.imwrite(path_clase+'/img_{}.jpg'.format(k), gray)
        k=k+1
            
            
        # Mostrar la imagen original 
        cv2.imshow('Detección de objetos', frame)
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
