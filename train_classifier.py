import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import log_loss, hinge_loss
import matplotlib.pyplot as plt


def main():
    # Caricamento del dataset
    data_dict = pickle.load(open('./dataset-asl-sign.pkl', 'rb'))
    data = np.array(data_dict['data'])
    labels = np.array(data_dict['labels'])

    # Split del dataset
    X_train, _, y_train, _ = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Modelli di classificazione
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=20, class_weight='balanced'),
        "SVM": SVC(probability=True),  # Abilitiamo probability=True per ottenere log_loss, False (default) calcola hinge_loss
        "KNN": KNeighborsClassifier(n_neighbors=9, weights='distance', metric='euclidean')
    }

    # Lista per memorizzare (accuracy, loss, nome modello, oggetto modello)
    results = []
    
    for name, model in models.items():
        # Accuracy con convalida incrociata
        accuracy_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mean_accuracy = np.mean(accuracy_scores) * 100

        # Calcolo della loss
        if hasattr(model, "predict_proba"):
            y_prob = cross_val_predict(model, X_train, y_train, cv=5, method="predict_proba")
            mean_loss = log_loss(y_train, y_prob) * 100
            
        # Calcolo della hinge_loss
        elif hasattr(model, "decision_function"):
            decision_values = cross_val_predict(model, X_train, y_train, cv=5, method="decision_function")
            mean_loss = hinge_loss(y_train, decision_values) * 100
        
        # Calcolo tasso di errore senza l'utilizzo del parametro method
        else:
            y_pred = cross_val_predict(model, X_train, y_train, cv=5)
            mean_loss = np.mean(y_pred != y_train) * 100
        
        print(f"{name} - Accuracy: {mean_accuracy:.2f}% - Loss: {mean_loss:.2f}%")
        
        results.append((mean_accuracy, mean_loss, name, model))
    
    # Sceglie il miglior modello bilanciando accuratezza e perdita
    alpha = 0.2
    best_model = max(results, key=lambda x: x[0] - (alpha * x[1]))
    best_accuracy, best_loss, best_model_name, best_model_instance = best_model
    
    # Allena il miglior modello su tutti i dati di training
    best_model_instance.fit(X_train, y_train)
    
    # Salvataggio del miglior modello
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': best_model_instance}, f)
    
    print(f"\nMiglior modello con cross-validation: {best_model_name} "
          f"con accuracy {best_accuracy:.2f}% e loss {best_loss:.2f}%")


    """
    ### CREAZIONE PLOT A BARRE PER LA SCELTA DEL MIGLIOR MODELLO ###
    model_names = [r[2] for r in results]
    accuracies = [r[0] for r in results]
    losses = [r[1] for r in results]
    balances = [acc - (alpha * loss) for acc, loss in zip(accuracies, losses)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(model_names, accuracies, color=['blue', 'green', 'red'])
    axes[0].set_title("Accuracy")
    axes[0].set_ylim(0, 100)

    axes[1].bar(model_names, losses, color=['blue', 'green', 'red'])
    axes[1].set_title("Loss")
    axes[1].set_ylim(0, 100)

    axes[2].bar(model_names, balances, color=['blue', 'green', 'red'])
    axes[2].set_title("Balance:  Accuracy - (alpha * Loss)")
    axes[2].set_ylim(0, 100)

    plt.tight_layout()
    plt.show()
    """

if __name__ == "__main__":
    main()
