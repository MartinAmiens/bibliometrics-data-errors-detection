import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# Charger les fichiers CSV
data_with_errors = pd.read_csv('data_with_errors.csv', encoding='utf-8')
data_corrected = pd.read_csv('data_corrected.csv', encoding='utf-8')

print(data_corrected)
print(data_with_errors)

# Marquer les erreurs dans data_with_errors
data_with_errors['is_error'] = 0

for index, row in data_with_errors.iterrows():
    id = row['id']
    full_name = row['Full_Name']
    if id not in data_corrected['id'].values or full_name not in data_corrected.loc[data_corrected['id'] == id, 'Full_Name'].values:
        data_with_errors.at[index, 'is_error'] = 1

# Séparer les noms et les identifiants
X = data_with_errors[['id', 'Full_Name']]
y = data_with_errors['is_error']

# Encodage One-Hot pour la colonne 'Full_Name'
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X[['Full_Name']])

# Concaténer avec les autres caractéristiques si nécessaire (par exemple 'id')
X_encoded = np.concatenate([X[['id']].values, X_encoded.toarray()], axis=1)

# Exemple de séparation des données
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Vérification des dimensions des données
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Exemple de construction et d'entraînement du modèle (à adapter selon votre modèle)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_test, y_test))

# Évaluation du modèle
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nLoss: {loss}, Accuracy: {accuracy}")

# Prédiction sur les données avec erreurs
predictions = model.predict(X_test)
rounded_predictions = np.round(predictions).flatten().astype(int)  # Assurez-vous de convertir les prédictions en un tableau 1D
print("\nPredictions on test set:")
print(rounded_predictions)