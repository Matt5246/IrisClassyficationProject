
# Importowanie niezbędnych bibliotek

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV


# Krok 1: Przygotowanie danych
data = pd.read_csv('./Iris.csv')
X = data.drop('Species', axis=1)
y = data['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,train_size=0.75, random_state=42)
# Load the Iris dataset
iris = load_iris()


# Krok 2: Eksploracja danych
print("Data inside Iris.csv only 5 rows:\n",data.head())
print("\nInformation about dataset:\n",data.describe())
sns.set(style="ticks")
sns.pairplot(sns.load_dataset("iris"), hue="species", markers=["o", "s", "D"], palette='Dark2')
plt.show()

# Krok 3: Wybór modelu
# Rozważ modele K-nearest neighbors (KNN), Support Vector Machines (SVM) i Random Forests

# Krok 4: Trenowanie modelu

model_knn = KNeighborsClassifier(n_neighbors=7)
model_svm = SVC(kernel='linear', C=1)
model_rf = RandomForestClassifier(n_estimators=100)

# Trenowanie modeli
#pierwszy model
model_knn.fit(X_train, y_train)
model_svm.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

# Krok 5: Ocena modelu
# Model KNN
y_pred_knn = model_knn.predict(X_test)
print("\nDokładność klasyfikacji KNN:", accuracy_score(y_test, y_pred_knn))

# Model SVM
y_pred_svm = model_svm.predict(X_test)
print("Dokładność klasyfikacji SVM:", accuracy_score(y_test, y_pred_svm))

# Model Random Forest
y_pred_rf = model_rf.predict(X_test)
print("Dokładność klasyfikacji Random Forest:", accuracy_score(y_test, y_pred_rf))

from sklearn import metrics
print("\naccuracy for model knn:")
print(metrics.classification_report(y_test, y_pred_knn, digits=3))
print("accuracy for model SVM:")
print(metrics.classification_report(y_test, y_pred_svm, digits=3))
print("accuracy for model random forest:")
print(metrics.classification_report(y_test, y_pred_rf, digits=3))
# Krok 6: Interpretacja wyników

# W tym przypadku zakładamy, że model SVM jest wybranym modelem i chcemy zrozumieć, które cechy miały największy wpływ na klasyfikację.
# W modelu SVM istnieje coś takiego jak współczynniki cech (coef_), które wskazują na wpływ każdej cechy na decyzje modelu.

coef = model_svm.coef_
feature_names = X.columns

# Wyświetlenie wpływu cech na klasyfikację
print("Wpływ cech na klasyfikację:")
for i, feature in enumerate(feature_names):
    print(f"{feature}: {coef[0][i]}")

# Krok 7: Walidacja krzyżowa (opcjonalne)

# Przeprowadźmy walidację krzyżową dla modelu SVM
scores_svm = cross_val_score(model_svm, X, y, cv=5)
print("\nWyniki walidacji krzyżowej dla modelu SVM:")
print(scores_svm)
print("\nŚrednia dokładność:", scores_svm.mean())

# Walidacja krzyżowa dla modelu KNN
scores_knn = cross_val_score(model_knn, X, y, cv=5)
print("\nWyniki walidacji krzyżowej dla modelu KNN:")
print(scores_knn)
print("\nŚrednia dokładność:", scores_knn.mean())

# Walidacja krzyżowa dla modelu Random Forest
scores_rf = cross_val_score(model_rf, X, y, cv=5)
print("\nWyniki walidacji krzyżowej dla modelu Random Forest:")
print(scores_rf)
print("\nŚrednia dokładność:", scores_rf.mean())

# Krok 8: Optymalizacja modelu (opcjonalne)

# Możemy eksperymentować z różnymi parametrami modelu, na przykład parametrem C w przypadku SVM.
param_grid = {'C': [0.1, 1, 10, 100]}
grid_search_svm = GridSearchCV(model_svm, param_grid, cv=5)
grid_search_svm.fit(X, y)

# Optymalizacja modelu KNN
param_grid_knn = {'n_neighbors': [3, 5, 7, 9, 11]}
grid_search_knn = GridSearchCV(model_knn, param_grid_knn, cv=5)
grid_search_knn.fit(X, y)

# Optymalizacja modelu Random Forest
param_grid_rf = {'n_estimators': [50, 100, 200, 300]}
grid_search_rf = GridSearchCV(model_rf, param_grid_rf, cv=5)
grid_search_rf.fit(X, y)

# Wyświetlenie najlepszych hiperparametrów
print("Najlepsze hiperparametry dla modelu SVM:", grid_search_svm.best_params_)
print("Najlepsza dokładność:", grid_search_svm.best_score_)

print("Najlepsze hiperparametry dla modelu KNN:", grid_search_knn.best_params_)
print("Najlepsza dokładność dla modelu KNN:", grid_search_knn.best_score_)

print("Najlepsze hiperparametry dla modelu Random Forest:", grid_search_rf.best_params_)
print("Najlepsza dokładność dla modelu Random Forest:", grid_search_rf.best_score_)

# Krok 9: Dokumentacja

