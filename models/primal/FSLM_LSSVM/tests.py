'''Script de testes para o modelo FSLM_LSSVM'''

#-------------------------------------
# Carregando pacotes relevantes
#-------------------------------------
import numpy as np
from sklearn.datasets import make_blobs, load_breast_cancer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style

from models.primal.FSLM_LSSVM.classifier import FixedSizedLMClassifier

#Base sintética pseudoaleatória
style.use("fivethirtyeight")
 
X, y = make_blobs(n_samples = 100, centers = 2, 
               cluster_std = 4, n_features = 10)
 

plt.scatter(X[:, 0], X[:, 1], s = 40, color = 'g')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
plt.clf()

for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1

#FSLM_LSSVM
model_FSLM_LSSVM = FixedSizedLMClassifier()
model_SVM = SVC(kernel = 'rbf', gamma = 1, C = 1)
resultados = model_FSLM_LSSVM.fit(X, y, max_iterations=100)
resultados_SVM = model_SVM.fit(X, y)
alphas = resultados["mult_lagrange"]
alphas
y_hat = model_FSLM_LSSVM.predict(X)
y_hat_SVM = model_SVM.predict(X)
accuracy = accuracy_score(y, y_hat)
accuracy
accuracy_SVM = accuracy_score(y, y_hat_SVM)
accuracy_SVM

# Dataset breast cancer wisconsin
breast_df = load_breast_cancer()
X_breast = breast_df.data
y_breast = breast_df.target


X_breast_scaled = (X_breast - X_breast.min(axis=0)) / (X_breast.max(axis=0) - X_breast.min(axis=0))
X_breast = X_breast_scaled

# Converção de labels para -1 e 1
y_breast = np.where(y_breast == 0, -1, 1)

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_breast, 
                                                    y_breast,
                                                    test_size=0.2)

# Treinamento do modelo FSLM_LSSVM
model_FSLM_LSSVM = FixedSizedLMClassifier()
resultados = model_FSLM_LSSVM.fit(X_train, y_train)
resultados_SVM = model_SVM.fit(X_train, y_train)
alphas = resultados["mult_lagrange"]
alphas

# Predição e avaliação
y_hat = model_FSLM_LSSVM.predict(X_test)
y_hat_SVM = model_SVM.predict(X_test)
accuracy = accuracy_score(y_test, y_hat)
accuracy
accuracy_SVM = accuracy_score(y_test, y_hat_SVM)
accuracy_SVM