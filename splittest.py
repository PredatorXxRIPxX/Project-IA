from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve,train_test_split,cross_val_score,validation_curve,GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

titanic = sns.load_dataset('titanic')

titanic = titanic[['survived', 'pclass', 'age', 'sex']]

titanic['age'].fillna(titanic['age'].median(), inplace=True)
titanic['sex'].replace({'male', 'female'}, [0, 1], inplace=True)

X = titanic.drop('survived', axis=1)
y = titanic['survived']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

best_configuration = {
    'cross_val': 0,
    'k': 0,
    'cv': 0
}

valscore = []

for k in range(1, 50): 
    model = KNeighborsClassifier(n_neighbors=k)
    
    cv_values = range(2, 11)  

    for cv in cv_values:
        cross_val = np.mean(cross_val_score(model, x_train, y_train, cv=cv, scoring='accuracy'))
        print(f"k={k}, cv={cv}, Cross Validation Score: {cross_val:.4f}")

        if cross_val > best_configuration['cross_val']:
            best_configuration['cross_val'] = cross_val
            best_configuration['k'] = k
            best_configuration['cv'] = cv
        
        valscore.append(cross_val)

print("\nBest Configuration:", best_configuration)


k = range(1, 50)
train_score , val_score = validation_curve(model, x_train, y_train, param_name='n_neighbors',param_range=k, cv=5)
params_grid = {'n_neighbors': k,'metric': ['euclidean', 'manhattan']}

grid = GridSearchCV(model, param_grid=params_grid,cv=5)
grid.fit(x_train, y_train)
model = grid.best_estimator_
print(grid.best_params_)
print(grid.best_score_)
print(grid.best_estimator_)
print(grid.score(x_test, y_test))
score = model.score(x_test, y_test)
print(score)

N,train_score,val_score =learning_curve(model, x_train, y_train, train_sizes=np.linspace(0.1,1.0,5) ,cv=5)

plt.plot(N,train_score.mean(axis=1),label='train')
plt.plot(N,val_score.mean(axis=1),label='validation')
plt.xlabel('train_sizes')
plt.legend()
plt.show()
