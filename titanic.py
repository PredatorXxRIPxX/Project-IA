import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

titanic = sns.load_dataset('titanic')
titanic.shape

print(titanic)

titanic = titanic[['survived',
                   'pclass',
                   'age',
                   'sex']]
titanic.dropna(axis=0, inplace=True)
titanic["sex"].replace(['male','female'], [0,1], inplace=True)
titanic.head()
print(titanic.head())

X = titanic.drop('survived', axis=1)
y = titanic['survived']

model = KNeighborsClassifier()
model.fit(X,y)
score = model.score(X,y)

prediction = model.predict(X)

plt.scatter(titanic['age'], titanic['pclass'], c=prediction)

plt.show()
