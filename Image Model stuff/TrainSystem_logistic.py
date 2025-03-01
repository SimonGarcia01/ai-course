'''
Milton Orlando Sarria
Ejemplo para entrenar un sistema base empleando regresion logistica
se hace uso de las imagenes que han sido recolectadas previamente


pip install scikit-image
'''
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob 
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pickle
from skimage import feature
import os
#-------------------

## leer todos los archivos
root = 'data'
file_names = glob(root+"/**/*.jpg", recursive=True)
clases = glob(root+"/*")
clases=np.array([clase.split(os.sep)[1] for clase in clases])
print(clases)
#leer las imagenes
frames = []
labels = []
dic_clases = {}
tipo = int(input("Que tipo de carácteristicas desea usar: \n1) imagen en escala de gris \n2) Local binary patterns \n:"))
file_tipe = ""
#-------------------
for file_name in tqdm(file_names):      
    frame = cv2.imread(file_name)
    #frame = frame.resize((100,100))   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    #definir el tipo de caracteristica para el clasificador
    if(tipo==1):
      feats = np.array(gray).ravel()/255
      file_tipe = "_gray.pkl"
    else:
      lbp_face = feature.local_binary_pattern(gray, 28, 8, method="uniform")
      feats = np.array(lbp_face).ravel()
      file_tipe = "_lbp.pkl"
    
    clase = file_name.split(os.sep)[1]

    label = np.where(clases==clase)[0][0]
    labels.append(label)
    frames.append(feats)
    dic_clases[label]=clase
#convertir a array
names = np.array(labels)
X=np.vstack(frames)
print(f'Done loading {names.shape} {X.shape}') 
#-------------------
#image = X[0].reshape((50,50))
#plt.imshow(image,cmap="gray");
#plt.show()

ACC = [] 
x_train, x_test, y_train, y_test = train_test_split(X,names,test_size = 0.3)
#-------------------
#normalizar
n_components = 2
pca = PCA(n_components=n_components) #n_components=0.99
pca.fit(x_train)

x_train=pca.transform(x_train)
x_test=pca.transform(x_test)

#normalizar
if n_components==2:
   # Visualizar los datos
   plt.scatter(x_train[:, 0],x_train[:, 1], c=y_train, cmap='viridis')
   plt.xlabel('Feature 1')
   plt.ylabel('Feature 2')
   plt.title('Datos ')
   plt.show()

#-------------------
print(f"[INFO] train set: {x_train.shape}, test set: {x_test.shape}")
print("[INFO] entrenando....")
#entrenar un clasificador simple
clf = LogisticRegression(solver='lbfgs', max_iter=1000)
clf.fit(x_train,y_train)
print("[INFO] evaluando....")
y = clf.predict(x_test)
acc =(y==y_test).sum()/y.size*100
print('[INFO] Porcentaje de prediccion : ',acc)
#-------------------



#guardar para posterior uso

with open('clasificador'+file_tipe, 'wb') as fh:
   pickle.dump(clf, fh)
   
with open('clases'+file_tipe, 'wb') as fh:
   pickle.dump(dic_clases, fh)


with open('pca'+file_tipe, 'wb') as fh:
   pickle.dump(pca, fh)   