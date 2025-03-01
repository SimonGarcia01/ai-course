import numpy as np
from scipy import signal
import math
import matplotlib.pyplot as plt
import cv2
import time 
import pickle
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from skimage import feature

# Inicializar el clasificador de detección de rostros
#
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (255,0,0)
thickness              = 2
lineType               = 2


#usar  modelos pre-entrenados
tipo = int(input("Que tipo de carácteristicas desea usar: \n1) imagen en escala de gris \n2) Local binary patterns \n:  "))
file_tipe = ""
if(tipo==1):
       file_tipe = "_gray.pkl"
else:
       file_tipe = "_lbp.pkl"
#cargar clasificador entrenado 

fh = open ("clasificador"+file_tipe, "rb")
clf = pickle.load(fh)
fh.close()

fh = open ("pca"+file_tipe, "rb")
pca = pickle.load(fh)
fh.close()
#etiquetas
fh = open ("clases"+file_tipe, "rb")
clases = pickle.load(fh)
fh.close()

#habilitar camara
source = 0
cam = cv2.VideoCapture(source)

umbral  = 0.9
new_dim = (50,50)
continuar = True
#realizar un proceso de forma indefinida
while continuar: 
#for i in range(10):
    #
    retval, frame = cam.read()

    # Convertir a escala de grises para la detección de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    lbp_face = np.zeros(new_dim)            
    
    gray = cv2.resize(gray, new_dim)       
    if(tipo==1):
                feats = np.array(gray).ravel()/255
    else:
                lbp_face = feature.local_binary_pattern(gray, 28, 8, method="uniform")
                feats = np.array(lbp_face).ravel()
    #X = np.array(feats).ravel()
            
    X = pca.transform(feats.reshape(1, -1))
            
    #se calcula la clase y la probabilidad para saber si es muy baja
    clase     = clf.predict(X.reshape(1, -1))     
    predict   = clf.predict_proba(X)
    #print(clase)
    #print(predict)
    #print(clases)
    texto= 'Clase: ' + clases[clase[0]] + ' Prob: ' + str(predict[0][clase[0]])
            
    print(texto)
            
    cv2.imshow('Detección de objetos', frame)
    key =  cv2.waitKey(1)
    if key == 27 :
            break
    
cam.release()
cv2.destroyAllWindows()
		
print('\nDone')
