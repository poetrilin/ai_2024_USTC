# Lab2

## Part 1

### DecisionTreeClassifier

我们首先需要对连续值数据进行二值化：
```python
for col in continue_features:
    mid = (X[col].max() + X[col].min())/2
    X[col] = X[col].apply(lambda x: 1 if x > mid else 0)
```



### PCA+K-means