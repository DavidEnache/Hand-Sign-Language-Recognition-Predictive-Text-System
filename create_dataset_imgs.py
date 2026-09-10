import os
import pickle
import mediapipe as mp
import cv2
import numpy as np


# Processa un'immagine per estrarre i landmark della mano
def process_image(hands, image):
    data_aux = np.array([], dtype=np.float32)
    results = hands.process(image)

    # Aumento della luminosità nel caso in cui l'immagine sia troppo scura
    if not results.multi_hand_landmarks:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 50)
        v = np.clip(v, 0, 255)
        hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        results = hands.process(image)

    # Se i landmark sono stati trovati e se la dimensione e' esattamente 1
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Raccolta dati originali della mano
        x_ = np.array([lm.x for lm in hand_landmarks.landmark], dtype=np.float32)
        y_ = np.array([lm.y for lm in hand_landmarks.landmark], dtype=np.float32)

        # Normalizzazione dei dati della mano
        data_aux = np.concatenate([(x_ - np.min(x_)), (y_ - np.min(y_))])     

        if data_aux.size != 42:
            data_aux = np.array([], dtype=np.float32)
        
    return data_aux

def main():
    DATA_DIR = './asl_alphabet_dataset'
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)
    
    data = np.array([], dtype=np.float32).reshape(0, 42)
    labels = np.array([], dtype='<U10')
    test = {letter: 0 for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
    test['del'] = 0
    test['space'] = 0
    test['wait'] = 0

    # Ogni elemento della cartella è una singola lettera
    for dir_ in os.listdir(DATA_DIR):
        
        # Prendo le immagini che fanno riferimento alla cartella e le processo per rilevare i landmark
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_))[:300]:
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            # Utilizzo un formato standard per tutte le immagini passate in input
            img = cv2.resize(img, (640, 480))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Flip orizzontale per ottenere mano destra e sinistra
            img_flip = cv2.flip(img_rgb, 1)

            for img_new in [img_rgb, img_flip]:        
                processed_data = process_image(hands, img_new)
                
                # Se la mano è stata rilevata aggiungo le coordinate al dataset
                if processed_data.size > 0:
                    test[dir_] += 1
                    data = np.vstack([data, processed_data])
                    labels = np.append(labels, dir_)
    
    with open('dataset-asl-sign.pkl', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    
    print(test)

if __name__ == "__main__":
    main()
