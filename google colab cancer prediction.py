from tensorflow.keras.models import load_model
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

model = load_model('model2.h5')
while True:
    G = cv2.imread(str('/content/drive/MyDrive/Colab Notebooks/mine only/final/TEST/7.jpg'))
    cv2_imshow(G)
    key=cv2.waitKey(4)
    break
cv2.destroyAllWindows()

G = cv2.resize(G, (128,128))
G = np.array(G)
G = np.expand_dims(G, axis=0)
prediction = model.predict(G)
print(prediction)

if np.argmax(prediction) == 0:
            print("Warning, Adenocarcinoma")

elif np.argmax(prediction) == 1:
            print('Warning, Carcinoma')
else:
            print("Normal")