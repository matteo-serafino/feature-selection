# Feature Selection Package
Python package for plug and play feature selection techniques, cross-validation and performance evalutation of machine learing models.
If you like the idea or you find usefull this repo in your job, please leave a ⭐ to support this personal project.

1. Feature Selection techiniques (to be tested)
    * [Variance Threshold](#variance-threshold);
    * [Anova](#anova);
    * [Mutual Information](#mutual-information);
    * [Recursive Feature Elimination (RFE)](#recursive-feature-elimination-rfe);
    * [Random Forest Feature Importance](#random-forest-feature-importance);
    * [ReliefF](#relieff);
    * [Cross-correlation removal](#cross-correlation-removal)(to implement);
    * [Cluster quality](#cluster-quality).

To accompany the feature section method this package has also:
* Cross Validation methods with performance metrics
    * K-fold;
    * Leave One Out (LOO);
    * Leave One Subject Out (LOSO).

* Performance Metrics for binary and multi-class tasks:
    * Confusion Matrix Plot (Binary and multi class tasks);
    * Precision (binary tasks);
    * Sensitivity (binary tasks);
    * Specificity (binary tasks);
    * F1 Score (binary tasks);
    * sklearn classification report (Binary and multi class tasks).

Each method returns three outputs:
* `conf_matrix`: confusion Matrix of the 5-fold cross validation using the input model and the selected features;
* `fs_perf`: dataframe with the baseline and the feature selection classification performance, to understand of the feature selection method works for your classification task;
* `feat_selected`: dataframe with the selected features, this dataframe is the input X dataframe with only the selected columns. 

At the moment the package is not available using `pip install <PACKAGE-NAME>`.

For the installation from the source code click **[here](#installation)**.

## Variance Threshold

### Example
```python
from src.feature_selection.feature_selection import FeatureSelection

conf_matrix, fs_perf, feat_selected = FeatureSelection().variance_threshold(clf, X, y, thr=0.5, baseline=True)
```

## Anova

### Example
```python
from src.feature_selection.feature_selection import FeatureSelection

conf_matrix, fs_perf, feat_selected = FeatureSelection().anova(clf, X, y, n_feat=30, baseline=True)
```

## Mutual Information

### Example
```python
from src.feature_selection.feature_selection import FeatureSelection

conf_matrix, fs_perf, feat_selected = FeatureSelection().mutual_info(clf, X, y, n_feat=30, baseline=True)
```

## Recursive Feature Elimination (RFE)

### Example
```python
from src.feature_selection.feature_selection import FeatureSelection

conf_matrix, fs_perf, feat_selected = FeatureSelection().recursive_feature_elimination(clf, X, y, n_feat=30, baseline=True)
```

## Random Forest Feature Importance

### Example
```python
from src.feature_selection.feature_selection import FeatureSelection

conf_matrix, fs_perf, feat_selected = FeatureSelection().random_forest_importance(clf, X, y, threshold=0.8, baseline=True, verbose=True)
```

## ReliefF

### Example
```python
from src.feature_selection.feature_selection import FeatureSelection

conf_matrix, fs_perf, feat_selected = FeatureSelection().relieff(clf, X, y, n_feat=30, baseline=True)
```

## Cross-correlation removal

### Example
```python
from src.feature_selection.feature_selection import FeatureSelection
```

## Cluster quality

### Example
```python
from src.feature_selection.feature_selection import FeatureSelection

conf_matrix, fs_perf, feat_selected = FeatureSelection().cluster_quality(clf, X, y, n_feat=30, baseline=True, verbose=True)
```

## Installation
For the installation from the source code type this command into your terminal window:
```
pip install git+<repository-link>
```
or
```
python -m pip install git+<repository-link>
```
or
```
python3 -m pip install git+<repository-link>
```