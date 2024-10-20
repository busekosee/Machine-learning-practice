#sklearn: Ml library
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt 
#Veri seti incelemesi

cancer = load_breast_cancer()

df = pd.DataFrame(data = cancer.data, columns= cancer.feature_names)
df["target"] = cancer.target

X =cancer.data #feature
y =cancer.target #target

#train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3 ,random_state =42)

#knn mesafe tabanlı olduğu için ölçeklendirme standarlizasyon yapmak gerekiyor kütüphane entegrasyonu vs yapnalıyız
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
#fit etmek transform etmesi gereken parametreleri öğreniyor fit ediyor

X_test = scaler.transform(X_test)



knn = KNeighborsClassifier(n_neighbors=3) #komşu parametre
knn.fit(X_train,y_train) #fit fonksiyonu bize verimizi yani feature ve sample kullanarak knn algoritmasını eğitir

#predict fonksiyonu bize tahmin yapıcak biz ona sample verince targettan oluşan sıfırmı bir mi bize söyliyecek

y_pred =knn.predict(X_test)

#skor karşılaştırmak için
#%94 başarı ulaştık
#x_test ve y_test karşılarştırdık
accuracy = accuracy_score(y_test,y_pred)
print("doğruluk:", accuracy)

#tahmin matrisi
conf_matrix = confusion_matrix(y_test, y_pred)
print("confusion_matrix:")
print(conf_matrix)

#Hiperparametre Ayarlaması
#K değerleri ile accuracy karşılatırmaya denir
accuracy_values = []
k_values = []
for k in range(1,21):
 knn = KNeighborsClassifier(n_neighbors=k)
 knn.fit(X_train, y_train)
 y_pred = knn.predict(X_test)
 accuracy = accuracy_score(y_test,y_pred)
 accuracy_values.append(accuracy)
 k_values.append(k)

plt.figure()
plt.plot(k_values,accuracy_values,marker ="o",linestyle = "-")
plt.title("K değerine göre doğruluk")
plt.xlabel("K değeri")
plt.ylabel("Doğruluk")
plt.xticks(k_values)
plt.grid(True)
# %%
#regration yapıcaz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

X = np.sort(5 * np.random.rand(40,1), axis = 0) #üniform dağılıma göre üretilmiş 40 adet sayı
y = np.sin(X).ravel() #target
#plt.scatter(X,y) #0-5 arasına taşıdık boşluk oldu bunları knn ile doldurcaz 
# add noise
y[::5] += 1 * (0.5 - np.random.rand(8))
#plt.scatter(X,y)

T = np.linspace(0,5, 500)[:, np.newaxis]

for i, weight in enumerate(["uniform","distance"]):
    
 
  knn = KNeighborsRegressor(n_neighbors=5, weights=weight)
  y_pred = knn.fit(X,y).predict(T)
  
  plt.subplot(2,1, i + 1)
  plt.scatter(X,y,color = "green",label = "data")
  plt.plot(T,y_pred,color = "blue", label = "prediction")
  plt.axis("tight")
  plt.legend()
  plt.title("KNN Regressor weight = {}".format(weight))
plt.tight_layout()
plt.show()


