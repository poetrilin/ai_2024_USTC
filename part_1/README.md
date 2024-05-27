
## DecisionTree

### Description
  这次实验使用决策树对数据进行分类，实验中使用的数据集是肥胖病数据集，数据集中包含了一些关于肥胖病的特征，如性别、年龄、体重指数等，以及是否患有肥胖病的标签。实验中使用决策树对数据进行分类，最终得到了一个准确率为 `0.7636`的分类器。

### 预处理
对连续值特征('Height', 'Weight', 'NCP'等)进行了二值化处理，基于划分点$t$可将$D$ 分为 $D_t^{-}$和 $D_t^{+}$ 两个子集，其中$t=\frac{D_{Max}+D_{min}}{2}$

```python
# binarize continue features
    for col in continue_features:
        mid = (X[col].max() + X[col].min())/2
        X[col] = X[col].apply(lambda x: 1 if x > mid else 0)
```
### 决策树生成

DecisionTree的拟合本质是决策树的生成,DecisionTree的生成过程使用了ID3算法，基于信息增益来选择最优特征进行划分,再递归的生成子树，直到满足停止条件。
其伪代码如下:
```python
def TreeGenerate(self, X: np.ndarray, y: np.ndarray, attr_list: List[int]) 
-> Union[int, dict]:
        # if all samples are in the same class C, label the node with C
        if len(np.unique(y)) == 1:
            return y[0]
        # if the attribute list is empty OR all samples have the same attribute value, label the node with the most common class in y
        if attr_list is None or len(np.unique(X, axis=0)) == 1:
            return np.argmax(np.bincount(y))
        # select the best attribute to split, default ID3 method
        best_attr = self.SelectBestAttribute(X, y, attr_list) # ID3 method
        # split the dataset according to the best attribute
        for value in np.unique(X[:, best_attr]):
            ...
            tree[best_attr][value] = self.TreeGenerate(
                sub_X, sub_y, sub_attr_list)
        return tree
def SelectBestAttribute(self, X: np.ndarray, y: np.ndarray, attr_list: List[int]) -> int:
        """
        Select the best attribute to split the dataset based on information gain.(ID3 method)
        """
        ...
```

### Prediction
据此生成了决策树后，predict时，根据决策树的划分规则，递归的对样本进行分类，直到到达叶子节点，返回该节点的类别。
```python
def PredictOneSample(self, sample: np.ndarray, tree: Union[dict, int]) -> int:
        if not isinstance(tree, dict):
            return tree
        root = list(tree.keys())[0]  # the root node
        root_value = sample[root]
        if root_value not in tree[root]:
            # if no such value in the tree, return the most common class in the subtree
            ...
            return max(candidates, key=candidates.count)
        return self.PredictOneSample(sample, tree[root][root_value])
```
### Evaluation
`accuracy: 0.7636`

## PCA+KMeans

这次的任务是利用PCA和KMeans对Words 进行聚类,先利用预训练模型`GoogleNews-vectors-negative300.bin`对word进行embedding,然后利用PCA对word进行降维，最后利用KMeans对word进行聚类。

### PCA原理
PCA的本质是将原始数据投影到一个新的坐标系中，使得数据在新坐标系中的方差最大，即数据的信息量最大。形式化到公式中,PCA的目标是找到一个投影矩阵$W$，使得数据$X$投影到$W$后的数据$Z$的方差最大。

$W$的本质即核函数$K$的最大d个特征值对应的特征向量，即$W = [v_1, v_2, ..., v_d]$,PCA transform的公式为$Z = \bar XW$,实现如下:
```python 
def fit(self, X: np.ndarray):
        # X: [n_samples, n_features]
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        kernel_matrix = self.kernel_f(X)
        eignvalues, eignvectors = np.linalg.eig(kernel_matrix)
        # most k large eignvalues and corresponding eignvectors
        indices = np.argsort(eignvalues)[::-1]
        self.eignvalues = eignvalues[indices]
        self.eignvectors = eignvectors[:, indices]
        return self

    def transform(self, X: np.ndarray):
        # X: [n_samples, n_features]
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        W = self.eignvectors[:, :self.n_components]
        return X@W
```
### KMeans原理
KMeans的本质是将数据集划分为K个簇，使得每个样本点到其所属簇的中心点的距离最小。KMeans的目标是最小化目标函数$J = \sum_{i=1}^{K}\sum_{x \in C_i}||x - \mu_i||^2$,其中$\mu_i$是第i个簇的中心点，实现如下:

```python
def fit(self, points:np.ndarray):
    # points: (n_samples, n_dims,)
    self.initialize_centers(points)
    for _ in range(self.max_iter):
        new_centers = self.update_centers(points)
        if np.allclose(new_centers, self.centers):  # check convergence, early stop
            break
        self.centers = new_centers
    return self
# Update the centers based on the new assignment of points
def update_centers(self, points:np.ndarray):
    # points: (n_samples, n_dims,)
    clusters = self.assign_points(points)
    new_centers = np.zeros_like(self.centers)
    for k in range(self.n_clusters):
        cluster_points = points[clusters == k]
        if cluster_points.size == 0:  # 如果这个簇没有任何点，那么跳过这次循环
            continue
        new_centers[k] = cluster_points.mean(axis=0)
    return new_centers
```

### 结果
最终得到了一个`KMeans`的聚类器，对`Words`进行了聚类，如下图所示:
![](PCA_KMeans.png)
    