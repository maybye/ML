# Библиотека Scikit-learn для машинного обучения

## Регрессия по случайным точкам на подъеме синусоиды
<details><summary>Листинг</summary>
  
  ```
import numpy as np
import math
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
ridge_estimator = Ridge()
np.random.seed(42)
num_pts = 100
noise_range = 0.2
x_vals, y_vals = [], []
(x_l, x_r) = (-2, 2)
for i in range(num_pts):
x = np.random.uniform(x_l, x_r)
y = np.random.uniform(-noise_range, noise_range) + (2*math.sin(x))
x_vals.append(x)
y_vals.append(y)
x_col = np.reshape(x_vals, [len(x_vals), 1])
ridge_estimator.fit(x_col, y_vals)
y_left = ridge_estimator.predict([[x_l]])
y_right = ridge_estimator.predict([[x_r]])
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(x_vals, y_vals, s=20)
plt.title('original_data')
plt.subplot(1, 2, 2)
plt.scatter(x_vals, y_vals, s=20)
y_left = ridge_estimator.predict([[x_l]])
y_right = ridge_estimator.predict([[x_r]])
plt.plot([x_l, x_r], [y_left, y_right], color='#ff0000', linewidth=3)
plt.title('data with the best line')
plt.show()
```
</details>
<details><summary>Вывод</summary>
  
![image](https://github.com/user-attachments/assets/a2c1d172-ed25-4e3e-a696-cf22bb7e2c06)
</details>

## Классификация с помощью методов predict, decision_function, predict_proba
<details><summary>Листинг</summary>
  
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, 0:2] # we only take the first two features for visualization
y = iris.target
n_features = X.shape[1]
C = 10
kernel = 1.0 * RBF([1.0, 1.0]) # for GPC
# Create different classifiers.
classifiers = {
"L1 logistic": LogisticRegression(
C=C, penalty="l1", solver="saga", multi_class="multinomial", max_iter=10000
),
"L2 logistic (Multinomial)": LogisticRegression(
C=C, penalty="l2", solver="saga", multi_class="multinomial", max_iter=10000
),
"L2 logistic (OvR)": LogisticRegression(
C=C, penalty="l2", solver="saga", multi_class="ovr", max_iter=10000
),
"Linear SVC": SVC(kernel="linear", C=C, probability=True, random_state=0),
"GPC": GaussianProcessClassifier(kernel),
}
n_classifiers = len(classifiers)
plt.figure(figsize=(3 * 2, n_classifiers * 2))
plt.subplots_adjust(bottom=0.2, top=0.95)
xx = np.linspace(3, 9, 100)
yy = np.linspace(1, 5, 100).T
xx, yy = np.meshgrid(xx, yy)
Xfull = np.c_[xx.ravel(), yy.ravel()]
for index, (name, classifier) in enumerate(classifiers.items()):
classifier.fit(X, y)
y_pred = classifier.predict(X)
accuracy = accuracy_score(y, y_pred)
print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
# View probabilities:
probas = classifier.predict_proba(Xfull)
n_classes = np.unique(y_pred).size
for k in range(n_classes):
plt.subplot(n_classifiers, n_classes, index * n_classes + k + 1)
plt.title("Class %d" % k)
if k == 0:
plt.ylabel(name)
imshow_handle = plt.imshow(
probas[:, k].reshape((100, 100)), extent=(3, 9, 1, 5), origin="lower"
)
plt.xticks(())
plt.yticks(())
idx = y_pred == k
if idx.any():
plt.scatter(X[idx, 0], X[idx, 1], marker="o", c="w", edgecolor="k")
ax = plt.axes([0.15, 0.04, 0.7, 0.05])
plt.title("Probability")
plt.colorbar(imshow_handle, cax=ax, orientation="horizontal")
plt.show()
```

</details>

<details><summary>Вывод</summary>
  
  ![image](https://github.com/user-attachments/assets/fb2c01ac-37a7-471f-acba-dfc28945326f)
</details>

## Кластеризация методом k-means случайно сгенерированных кластеров
<details><summary>Листинг</summary>
  
  ```
from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
# Generate sample data
n_samples = 4000
n_components = 4
X, y_true = make_blobs(
n_samples=n_samples, centers=n_components, cluster_std=0.60, random_state=0
)
X = X[:, ::-1]
# Calculate seeds from k-means++
centers_init, indices = kmeans_plusplus(X, n_clusters=4, random_state=0)
plt.figure(1)
colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
for k, col in enumerate(colors):
cluster_data = y_true == k
plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker=".", s=10)
plt.scatter(centers_init[:, 0], centers_init[:, 1], c="b", s=50)
plt.title("K-Means++ Initialization")
plt.xticks([])
plt.yticks([])
plt.show()
```
</details>
<details><summary>Вывод</summary>

  ![image](https://github.com/user-attachments/assets/095082dd-dc6a-4ba0-85db-9cc816a5b3e9)

</details>

## Классификация с помощью нейронной сети и класса MLPClassifier
<details><summary>Листинг</summary>
  
```
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
names = [
"Nearest Neighbors",
"Linear SVM",
"RBF SVM",
"Gaussian Process",
"Decision Tree",
"Random Forest",
"Neural Net",
"AdaBoost",
"Naive Bayes",
"QDA",
]
classifiers = [
KNeighborsClassifier(3),
SVC(kernel="linear", C=0.025),
SVC(gamma=2, C=1),
GaussianProcessClassifier(1.0 * RBF(1.0)),
DecisionTreeClassifier(max_depth=5),
RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
MLPClassifier(alpha=1, max_iter=1000),
AdaBoostClassifier(),
GaussianNB(),
QuadraticDiscriminantAnalysis(),
]
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1,
n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)
datasets = [
make_moons(noise=0.3, random_state=0),
make_circles(noise=0.2, factor=0.5, random_state=1),
linearly_separable,
]
figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
# preprocess dataset, split into training and test part
X, y = ds
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.4, random_state=42
)
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
if ds_cnt == 0:
ax.set_title("Input data")
# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
edgecolors="k")
# Plot the testing points
ax.scatter(
X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
edgecolors="k"
)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xticks(())
ax.set_yticks(())
i += 1
# iterate over classifiers
for name, clf in zip(names, classifiers):
ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
clf = make_pipeline(StandardScaler(), clf)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
DecisionBoundaryDisplay.from_estimator(
clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
)
# Plot the training points
ax.scatter(
X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
)
# Plot the testing points
ax.scatter(
X_test[:, 0],
X_test[:, 1],
c=y_test,
cmap=cm_bright,
edgecolors="k",
alpha=0.6,
)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xticks(())
ax.set_yticks(())
if ds_cnt == 0:
ax.set_title(name)
ax.text(
x_max - 0.3,
y_min + 0.3,
("%.2f" % score).lstrip("0"),
size=15,
horizontalalignment="right",
)
i += 1
plt.tight_layout()
plt.show()
```

</details>
<details><summary>Вывод</summary>
  
  ![image](https://github.com/user-attachments/assets/ac20176d-03b3-4f47-bd6f-019256790bb9)

</details>

## Пример применения линейной регрессии для случайных чисел, расположенных вдоль какой-то кривой линии.
<details><summary>Листинг</summary>

  ```
import matplotlib.pyplot as plt
import numpy as np
rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y);
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
X = x[:, np.newaxis]
model.fit(X, y)
xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)
plt.scatter(x, y)
plt.plot(xfit, yfit);
```
</details>
<details><summary>Вывод</summary>

  ![image](https://github.com/user-attachments/assets/4433416d-8c4a-4701-b97d-460a8e91e8b2)

</details>

## Пример использования наивного Байесова классификатора.
<details><summary>Листинг</summary>

  ```
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
iris = sns.load_dataset('iris')
X_iris = iris.drop('species', axis=1)
y_iris = iris['species']
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest, y_model))
```
</details>
<details><summary>Вывод</summary>

  ```
0.9736842105263158
```
</details>

## Пример использования метода главных компонент (PCA).
<details><summary>Листинг</summary>

  ```
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
iris = sns.load_dataset('iris')
X_iris = iris.drop('species', axis=1)
y_iris = iris['species']
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
from sklearn.decomposition import PCA
model = PCA(n_components=2)
model.fit(X_iris)
X_2D = model.transform(X_iris)
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False);
```
</details>
<details><summary>Вывод</summary>

  ![image](https://github.com/user-attachments/assets/24707c7b-f439-4d19-b385-84426aaed575)

</details>

## Пример загрузки, визуализации и снижения размерности для распознавания цифр.
<details><summary>Листинг</summary>

  ```
from sklearn.datasets import load_digits
digits = load_digits()
import matplotlib.pyplot as plt
fig, axes = plt.subplots(10, 10, figsize=(8, 8),
subplot_kw={'xticks':[], 'yticks':[]},
gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
ax.text(0.05, 0.05, str(digits.target[i]),
transform=ax.transAxes, color='green')
```
</details>
<details><summary>Вывод</summary>

  ![image](https://github.com/user-attachments/assets/9eb4bb36-ad02-4c5f-a9c5-f89c4eee4f3c)

</details>

## Пример классификации цифр.
<details><summary>Листинг</summary>

  ```
from sklearn.datasets import load_digits
digits = load_digits()
import matplotlib.pyplot as plt
fig, axes = plt.subplots(10, 10, figsize=(8, 8),
subplot_kw={'xticks':[], 'yticks':[]},
gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
ax.text(0.05, 0.05, str(digits.target[i]),
transform=ax.transAxes, color='green')
X = digits.data
y = digits.target
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest, y_model))
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, y_model)
import seaborn as sns
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value');
fig, axes = plt.subplots(10, 10, figsize=(8, 8),
subplot_kw={'xticks':[], 'yticks':[]},
gridspec_kw=dict(hspace=0.1, wspace=0.1))
test_images = Xtest.reshape(-1, 8, 8)
for i, ax in enumerate(axes.flat):
ax.imshow(test_images[i], cmap='binary', interpolation='nearest')
ax.text(0.05, 0.05, str(y_model[i]),
transform=ax.transAxes,
color='green' if (ytest[i] == y_model[i]) else 'red')
```
</details>
<details><summary>Вывод</summary>

  ![image](https://github.com/user-attachments/assets/31861951-f080-4d20-b9df-563c041919da)

</details>
