# MICE
MICE Imputation implementation using scikit learn.
## Announcement
This repo is created because of the mess of another implementation [scikit-mice](https://github.com/Ouwen/scikit-mice). There are too many errors and the repo has not been maintained for a long time. 

-------------

## Requirements
- scipy
- numpy
- sklearn

-------------
### Documentation:
The MiceImputer class is similar to the sklearn <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html">Imputer</a> class. 

MiceImputer has the same instantiation parameters as Imputer.

The MiceImputer.transform() function takes in three arguments.

| Param                 | Type         | Description                                      |
| --------------------- | ------------ | ------------------------------------------------ |
| `X`                   | `matrix`     | Numpy matrix or python matrix of data.           |
| `model_class`         | `class`      | Scikit-learn model class.                        |
| `iterations`          | `int`        | Int for numbe of interations to run.             |


What is returned by MiceImputer is a tuple of imputed values as well as a matrix of model performance for each iteration and column.
```
(imputed_x, model_specs_matrix)
```

### Example:

```
from sklearn.linear_model import LinearRegression
import skmice

imputer = MiceImputer()
X = [[1, 2], [np.nan, 3], [7, 6]]

X = imputer.transform(X, LinearRegression, 10)

print X

```

What is returned is a MICE imputed matrix running 10 iterations using a simple LinearRegression.