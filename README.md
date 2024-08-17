```markdown
# Example of Overfitting with Random Forest Classifier

This project demonstrates the concept of overfitting using a Random Forest Classifier on a crowdfunding dataset. Overfitting occurs when a model performs exceptionally well on the training data but fails to generalize to new, unseen data, often due to the model being too complex.

## Dataset

The dataset used in this example contains information about crowdfunding projects. It includes the following features:

- `goal`: The funding goal of the project.
- `pledged`: The amount pledged by backers.
- `backers_count`: The number of backers supporting the project.
- `country`: The country where the project is based.
- `staff_pick`: Whether the project was picked by staff (binary).
- `spotlight`: Whether the project was highlighted (binary).
- `category`: The category of the project.
- `days_active`: The number of days the project was active.
- `outcome`: The target variable indicating the success (`1`) or failure (`0`) of the project.

The dataset consists of 1,129 entries and is fully populated, with no missing values.

## Model Training and Overfitting

### Data Preparation

We begin by splitting the dataset into features (`X`) and the target variable (`y`):

```python
X = df.drop(columns=['outcome'])
y = df['outcome']
```

Next, we split the data into training and testing sets:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
```

### Initial Model Training

We use a Random Forest Classifier to fit the training data:

```python
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
```

### Model Evaluation

#### Testing Data Accuracy

The model achieved an accuracy of approximately 95.05% on the testing data:

```python
classifier.score(X_test, y_test)
```

#### Training Data Accuracy

The model achieved a perfect accuracy of 100% on the training data:

```python
classifier.score(X_train, y_train)
```

### Identifying Overfitting

The significant difference between the training accuracy (100%) and the testing accuracy (95.05%) suggests that the model may be overfitting. The model perfectly memorizes the training data but fails to generalize well to new data.

### Model Complexity and Overfitting

To further investigate overfitting, we vary the `max_depth` parameter of the Random Forest model from 1 to 15 and record the accuracy scores for both training and testing data:

```python
depths = range(1, 15)
scores = {'train': [], 'test': [], 'depth': []}

for depth in depths:
    clf = RandomForestClassifier(max_depth=depth)
    clf.fit(X_train, y_train)

    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    scores['depth'].append(depth)
    scores['train'].append(train_score)
    scores['test'].append(test_score)
```

### Results Visualization

The results are stored in a DataFrame and plotted to visualize the relationship between model complexity (`max_depth`) and accuracy:

```python
import pandas as pd

scores_df = pd.DataFrame(scores).set_index('depth')
scores_df.plot()
```

### Observations

- **Training Accuracy**: As the `max_depth` increases, the training accuracy remains high, eventually reaching 100%.
- **Testing Accuracy**: The testing accuracy initially improves as `max_depth` increases but then starts to decrease, indicating overfitting.

This pattern confirms that increasing model complexity (i.e., allowing deeper trees) leads to overfitting, where the model becomes too tailored to the training data, losing its ability to generalize.

## Conclusion

This project demonstrates how overfitting can occur in a machine learning model, particularly with complex models like Random Forests when they are not properly constrained. While the model achieves perfect accuracy on the training data, its performance on unseen testing data reveals that it does not generalize well, a hallmark of overfitting.

## Recommendations

- **Regularization**: Consider techniques such as limiting the `max_depth`, reducing the number of trees, or employing other regularization methods to prevent overfitting.
- **Cross-Validation**: Use cross-validation to ensure that the model's performance is consistent across different subsets of the data.
- **Simpler Models**: Evaluate whether simpler models might perform better on the testing data, even if they do not achieve perfect accuracy on the training data.
```
