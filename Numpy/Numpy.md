# Numpy Basics

| Syntax      | Description |
| :-----------: | ----------- |
| np.zeros_like(a, dtype)      | Zeros with same dimention as a     |
| np.ones_like(a, dtype)   | Ones with same deimention as a        |
| np.maximum(A,k)      | Takes maximum with everyelemt and K ( Note the differene to np.max() ) |
| v.ravel() | A 1-D array, containing the elements of the input, is returned |
| np.indices((n,m)) | Useful for generating coordinates on coorinates axis for locations for example |

### The known numpy issue

> np.random.choice produces error when probabilities dont add to zero

**Solution** :

```Python
p = np.array(p)
p /= p.sum()  # normalize
np.random.choice([1,2,3,4,5,6,7,8,9, 10], 4, p=p, replace=False)
```
