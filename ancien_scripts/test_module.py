import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np

# Charger les fichiers CSV
data_with_errors = pd.read_csv('data_with_errors.csv')
data_corrected = pd.read_csv('data_corrected.csv')

# Vérifier le type de données d'une colonne spécifique
print("Type de données de la colonne 'id' dans data_with_errors :")
print(data_with_errors['id'].dtype)

print("Type de données de la colonne 'Full_Name' dans data_with_errors :")
print(data_with_errors['Full_Name'].dtype)


print("\nType de données de la colonne 'Full_Name' dans data_corrected :")
print(data_corrected['Full_Name'].dtype)

print("\nType de données de la colonne 'id' dans data_corrected :")
print(data_corrected['id'].dtype)




# Entraîner le modèle
model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_test, y_test))

# Évaluation du modèle
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nLoss: {loss}, Accuracy: {accuracy}")

# Prédiction sur les données avec erreurs
predictions = model.predict(X_test)
rounded_predictions = np.round(predictions).flatten().astype(int)  # Assurez-vous de convertir les prédictions en un tableau 1D
print("\nPredictions on test set:")
print(rounded_predictions)

