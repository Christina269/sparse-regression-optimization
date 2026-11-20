# Sparse Regression and Feature Selection


A machine learning coursework project comparing unconstrained least squares solved by gradient descent with cardinality-constrained regression solved by iterative hard thresholding (IHT).

## Experiments
- Synthetic regression with 100 observations, 50 features and five nonzero coefficients.
- Recovery of coefficient values and support; MSE convergence comparison.
- Iris petal-width regression using the remaining three measurements, with a single-feature IHT model.

## Run
```sh
python -m pip install -r requirements.txt
jupyter notebook sparse_regression.ipynb
```
Run all cells from top to bottom. Iris is bundled with scikit-learn; no external dataset is required.

## Evaluation methodology
The Iris experiment uses a deterministic train/test split, training-only standardization and a fitted intercept via target centering. The coefficient plot covers all 50 synthetic features. The synthetic experiment evaluates in-sample recovery; Iris reports held-out MSE, not classification accuracy. Hard thresholding retains at most `c` coefficients, including when magnitudes tie or `c=0`. The notebook includes a cardinality check.

## Academic context
Undergraduate Machine Learning assignment. Review course publication rules and acknowledge any collaborators before public release.

## Experimental results
| Method | Synthetic training MSE | Coefficient L2 error | Iris test MSE |
|---|---:|---:|---:|
| GD | 0.1113 | 0.5412 | 0.0338 |
| IHT | 0.2198 | 0.0755 | 0.0436 |

IHT recovered all five true nonzero features in this synthetic run and selected petal length in the Iris experiment. Results are from a single seeded experiment, not a general performance guarantee. The executed notebook contains the plots.
