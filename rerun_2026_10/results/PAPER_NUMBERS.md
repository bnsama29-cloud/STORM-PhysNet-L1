# PAPER NUMBERS (auto-generated)

## Claim checks (True = the data supports the sentence)

- ✅ STORM-Bz > matched TF at 1h in every block (seed means)
- ✅ STORM-Bz > matched TF at 1h, pooled paired p<0.05
- ❌ STORM-Bz > matched TF at 6h in every block (seed means)
- ❌ STORM-Bz > matched TF at 6h, pooled paired p<0.05
- ❌ STORM-Bz > matched TF at 12h in every block (seed means)
- ❌ STORM-Bz > matched TF at 12h, pooled paired p<0.05
- ✅ STORM-Bz > residual TF at 1h, pooled paired p<0.05
- ❌ STORM-Bz > residual TF at 6h, pooled paired p<0.05
- ❌ STORM-Bz > residual TF at 12h, pooled paired p<0.05
- ❌ solar-wind inputs help (matched TF > flux-only TF) at 1h, pooled p<0.05
- ✅ solar-wind inputs help (matched TF > flux-only TF) at 6h, pooled p<0.05
- ✅ solar-wind inputs help (matched TF > flux-only TF) at 12h, pooled p<0.05

## Splits

| Block   | Test_period              |   Train_windows |   Val_windows |   Test_windows |   Test_strict_windows |   Test_storm6h | Filled_train   | Filled_test   |   Min_Dst_test |
|:--------|:-------------------------|----------------:|--------------:|---------------:|----------------------:|---------------:|:---------------|:--------------|---------------:|
| A       | 2018-09-07 to 2020-03-04 |           38909 |          9264 |           7536 |                  4283 |             18 | 26.7%          | 28.7%         |            -69 |
| B       | 2015-01-01 to 2016-12-31 |           32296 |          9264 |          14021 |                  8452 |            565 | 32.2%          | 10.4%         |           -198 |
| C       | 2012-01-01 to 2013-12-31 |           33413 |          9264 |          12997 |                  7189 |            354 | 30.5%          | 15.9%         |           -139 |

## Main (seed mean ± sd)

| Block   | Model                       | PE_clim 1h    | PE_clim 6h    | PE_clim 12h   | PE_pers 1h   | PE_pers 6h   | PE_pers 12h   |
|:--------|:----------------------------|:--------------|:--------------|:--------------|:-------------|:-------------|:--------------|
| A       | Persistence                 | 0.966         | 0.515         | 0.179         | 0.000        | 0.000        | 0.000         |
| A       | LSTM                        | 0.935 ± 0.008 | 0.826 ± 0.012 | 0.791 ± 0.005 | -0.94 ± 0.24 | 0.64 ± 0.03  | 0.75 ± 0.01   |
| A       | Transformer (matched)       | 0.971 ± 0.002 | 0.855 ± 0.003 | 0.813 ± 0.005 | 0.14 ± 0.05  | 0.70 ± 0.01  | 0.77 ± 0.01   |
| A       | Transformer + residual head | 0.978 ± 0.001 | 0.860 ± 0.003 | 0.813 ± 0.004 | 0.33 ± 0.03  | 0.71 ± 0.01  | 0.77 ± 0.01   |
| A       | Transformer, flux-only      | 0.968 ± 0.006 | 0.818 ± 0.008 | 0.746 ± 0.013 | 0.05 ± 0.19  | 0.62 ± 0.02  | 0.69 ± 0.02   |
| A       | STORM-Bz                    | 0.978 ± 0.001 | 0.849 ± 0.006 | 0.804 ± 0.007 | 0.34 ± 0.02  | 0.69 ± 0.01  | 0.76 ± 0.01   |
| A       | STORM, no delay             | 0.979 ± 0.000 | 0.853 ± 0.004 | 0.806 ± 0.007 | 0.39 ± 0.01  | 0.70 ± 0.01  | 0.76 ± 0.01   |
| A       | STORM, no gate              | 0.978 ± 0.000 | 0.852 ± 0.003 | 0.806 ± 0.007 | 0.34 ± 0.01  | 0.69 ± 0.01  | 0.76 ± 0.01   |
| B       | Persistence                 | 0.964         | 0.556         | 0.206         | 0.000        | 0.000        | 0.000         |
| B       | LSTM                        | 0.933 ± 0.003 | 0.834 ± 0.007 | 0.778 ± 0.010 | -0.85 ± 0.08 | 0.63 ± 0.02  | 0.72 ± 0.01   |
| B       | Transformer (matched)       | 0.965 ± 0.002 | 0.868 ± 0.006 | 0.808 ± 0.005 | 0.04 ± 0.06  | 0.70 ± 0.01  | 0.76 ± 0.01   |
| B       | Transformer + residual head | 0.972 ± 0.001 | 0.875 ± 0.004 | 0.810 ± 0.006 | 0.24 ± 0.03  | 0.72 ± 0.01  | 0.76 ± 0.01   |
| B       | Transformer, flux-only      | 0.966 ± 0.003 | 0.836 ± 0.004 | 0.746 ± 0.010 | 0.06 ± 0.09  | 0.63 ± 0.01  | 0.68 ± 0.01   |
| B       | STORM-Bz                    | 0.974 ± 0.001 | 0.871 ± 0.004 | 0.799 ± 0.005 | 0.29 ± 0.02  | 0.71 ± 0.01  | 0.75 ± 0.01   |
| B       | STORM, no delay             | 0.975 ± 0.000 | 0.873 ± 0.003 | 0.800 ± 0.005 | 0.31 ± 0.01  | 0.71 ± 0.01  | 0.75 ± 0.01   |
| B       | STORM, no gate              | 0.974 ± 0.000 | 0.870 ± 0.004 | 0.801 ± 0.004 | 0.28 ± 0.01  | 0.71 ± 0.01  | 0.75 ± 0.00   |
| C       | Persistence                 | 0.973         | 0.673         | 0.398         | 0.000        | 0.000        | 0.000         |
| C       | LSTM                        | 0.942 ± 0.005 | 0.853 ± 0.006 | 0.808 ± 0.007 | -1.14 ± 0.17 | 0.55 ± 0.02  | 0.68 ± 0.01   |
| C       | Transformer (matched)       | 0.971 ± 0.002 | 0.879 ± 0.006 | 0.819 ± 0.005 | -0.07 ± 0.07 | 0.63 ± 0.02  | 0.70 ± 0.01   |
| C       | Transformer + residual head | 0.978 ± 0.001 | 0.882 ± 0.006 | 0.821 ± 0.008 | 0.18 ± 0.03  | 0.64 ± 0.02  | 0.70 ± 0.01   |
| C       | Transformer, flux-only      | 0.971 ± 0.006 | 0.848 ± 0.016 | 0.768 ± 0.008 | -0.08 ± 0.21 | 0.53 ± 0.05  | 0.61 ± 0.01   |
| C       | STORM-Bz                    | 0.979 ± 0.000 | 0.876 ± 0.005 | 0.806 ± 0.006 | 0.24 ± 0.01  | 0.62 ± 0.02  | 0.68 ± 0.01   |
| C       | STORM, no delay             | 0.980 ± 0.001 | 0.880 ± 0.004 | 0.811 ± 0.005 | 0.26 ± 0.02  | 0.63 ± 0.01  | 0.69 ± 0.01   |
| C       | STORM, no gate              | 0.979 ± 0.001 | 0.879 ± 0.005 | 0.810 ± 0.008 | 0.23 ± 0.03  | 0.63 ± 0.01  | 0.68 ± 0.01   |

## Block average

| Model                       |   PE_clim 1h |   PE_clim 6h |   PE_clim 12h |   PE_pers 1h |   PE_pers 6h |   PE_pers 12h |   Bagged 1h |   Bagged 6h |   Bagged 12h |
|:----------------------------|-------------:|-------------:|--------------:|-------------:|-------------:|--------------:|------------:|------------:|-------------:|
| LSTM                        |        0.937 |        0.838 |         0.792 |        -0.97 |         0.61 |          0.72 |       0.946 |       0.86  |        0.815 |
| Transformer (matched)       |        0.969 |        0.867 |         0.813 |         0.04 |         0.68 |          0.74 |       0.975 |       0.884 |        0.833 |
| Transformer + residual head |        0.976 |        0.872 |         0.815 |         0.25 |         0.69 |          0.75 |       0.979 |       0.888 |        0.836 |
| Transformer, flux-only      |        0.968 |        0.834 |         0.753 |         0.01 |         0.6  |          0.66 |       0.975 |       0.855 |        0.779 |
| STORM-Bz                    |        0.977 |        0.865 |         0.803 |         0.29 |         0.67 |          0.73 |       0.979 |       0.881 |        0.825 |
| STORM, no delay             |        0.978 |        0.869 |         0.806 |         0.32 |         0.68 |          0.73 |       0.98  |       0.883 |        0.827 |
| STORM, no gate              |        0.977 |        0.867 |         0.805 |         0.28 |         0.68 |          0.73 |       0.978 |       0.882 |        0.828 |

## Paired tests

| A                           | B                           | Block   | Horizon   |     dPE |     t |        p |   d_z |   n |
|:----------------------------|:----------------------------|:--------|:----------|--------:|------:|---------:|------:|----:|
| STORM-Bz                    | Transformer (matched)       | A       | 1h        |  0.0067 | 11.68 | 9.71e-07 |  3.69 |  10 |
| STORM-Bz                    | Transformer (matched)       | A       | 6h        | -0.0065 | -3.36 | 0.00836  | -1.06 |  10 |
| STORM-Bz                    | Transformer (matched)       | A       | 12h       | -0.0087 | -3.1  | 0.0127   | -0.98 |  10 |
| STORM-Bz                    | Transformer (matched)       | B       | 1h        |  0.0091 | 11.5  | 1.1e-06  |  3.64 |  10 |
| STORM-Bz                    | Transformer (matched)       | B       | 6h        |  0.0029 |  1.3  | 0.225    |  0.41 |  10 |
| STORM-Bz                    | Transformer (matched)       | B       | 12h       | -0.0089 | -4.11 | 0.00263  | -1.3  |  10 |
| STORM-Bz                    | Transformer (matched)       | C       | 1h        |  0.0085 | 13.57 | 2.68e-07 |  4.29 |  10 |
| STORM-Bz                    | Transformer (matched)       | C       | 6h        | -0.003  | -1.18 | 0.267    | -0.37 |  10 |
| STORM-Bz                    | Transformer (matched)       | C       | 12h       | -0.0135 | -7.88 | 2.49e-05 | -2.49 |  10 |
| STORM-Bz                    | Transformer (matched)       | pooled  | 1h        |  0.0081 | 19.44 | 3.52e-18 |  3.55 |  30 |
| STORM-Bz                    | Transformer (matched)       | pooled  | 6h        | -0.0022 | -1.53 | 0.136    | -0.28 |  30 |
| STORM-Bz                    | Transformer (matched)       | pooled  | 12h       | -0.0104 | -7.79 | 1.36e-08 | -1.42 |  30 |
| STORM-Bz                    | Transformer + residual head | A       | 1h        |  0.0004 |  1.03 | 0.329    |  0.33 |  10 |
| STORM-Bz                    | Transformer + residual head | A       | 6h        | -0.0116 | -7.81 | 2.67e-05 | -2.47 |  10 |
| STORM-Bz                    | Transformer + residual head | A       | 12h       | -0.0089 | -3.8  | 0.00423  | -1.2  |  10 |
| STORM-Bz                    | Transformer + residual head | B       | 1h        |  0.0016 |  4.42 | 0.00167  |  1.4  |  10 |
| STORM-Bz                    | Transformer + residual head | B       | 6h        | -0.0037 | -2.62 | 0.0277   | -0.83 |  10 |
| STORM-Bz                    | Transformer + residual head | B       | 12h       | -0.0109 | -3.89 | 0.0037   | -1.23 |  10 |
| STORM-Bz                    | Transformer + residual head | C       | 1h        |  0.0015 |  6.03 | 0.000195 |  1.91 |  10 |
| STORM-Bz                    | Transformer + residual head | C       | 6h        | -0.0059 | -2.77 | 0.0219   | -0.87 |  10 |
| STORM-Bz                    | Transformer + residual head | C       | 12h       | -0.0156 | -5.25 | 0.000525 | -1.66 |  10 |
| STORM-Bz                    | Transformer + residual head | pooled  | 1h        |  0.0012 |  5.55 | 5.45e-06 |  1.01 |  30 |
| STORM-Bz                    | Transformer + residual head | pooled  | 6h        | -0.0071 | -6.23 | 8.57e-07 | -1.14 |  30 |
| STORM-Bz                    | Transformer + residual head | pooled  | 12h       | -0.0118 | -7.37 | 4.09e-08 | -1.34 |  30 |
| STORM-Bz                    | Transformer, flux-only      | A       | 1h        |  0.0099 |  4.88 | 0.000871 |  1.54 |  10 |
| STORM-Bz                    | Transformer, flux-only      | A       | 6h        |  0.0311 |  9.99 | 3.6e-06  |  3.16 |  10 |
| STORM-Bz                    | Transformer, flux-only      | A       | 12h       |  0.0584 | 12.92 | 4.09e-07 |  4.09 |  10 |
| STORM-Bz                    | Transformer, flux-only      | B       | 1h        |  0.0083 |  8.36 | 1.56e-05 |  2.64 |  10 |
| STORM-Bz                    | Transformer, flux-only      | B       | 6h        |  0.0356 | 19.03 | 1.41e-08 |  6.02 |  10 |
| STORM-Bz                    | Transformer, flux-only      | B       | 12h       |  0.0532 | 14.8  | 1.26e-07 |  4.68 |  10 |
| STORM-Bz                    | Transformer, flux-only      | C       | 1h        |  0.0087 |  4.73 | 0.00108  |  1.49 |  10 |
| STORM-Bz                    | Transformer, flux-only      | C       | 6h        |  0.0276 |  5.79 | 0.000264 |  1.83 |  10 |
| STORM-Bz                    | Transformer, flux-only      | C       | 12h       |  0.0378 | 12.82 | 4.38e-07 |  4.05 |  10 |
| STORM-Bz                    | Transformer, flux-only      | pooled  | 1h        |  0.0089 |  9.49 | 2.13e-10 |  1.73 |  30 |
| STORM-Bz                    | Transformer, flux-only      | pooled  | 6h        |  0.0314 | 15.55 | 1.31e-15 |  2.84 |  30 |
| STORM-Bz                    | Transformer, flux-only      | pooled  | 12h       |  0.0498 | 18.83 | 8.37e-18 |  3.44 |  30 |
| STORM-Bz                    | LSTM                        | A       | 1h        |  0.043  | 16.29 | 5.5e-08  |  5.15 |  10 |
| STORM-Bz                    | LSTM                        | A       | 6h        |  0.0223 |  4.22 | 0.00225  |  1.33 |  10 |
| STORM-Bz                    | LSTM                        | A       | 12h       |  0.013  |  4.83 | 0.000933 |  1.53 |  10 |
| STORM-Bz                    | LSTM                        | B       | 1h        |  0.0414 | 39.68 | 2.04e-11 | 12.55 |  10 |
| STORM-Bz                    | LSTM                        | B       | 6h        |  0.0373 | 11.97 | 7.84e-07 |  3.79 |  10 |
| STORM-Bz                    | LSTM                        | B       | 12h       |  0.0209 |  5.43 | 0.000415 |  1.72 |  10 |
| STORM-Bz                    | LSTM                        | C       | 1h        |  0.0373 | 24.82 | 1.34e-09 |  7.85 |  10 |
| STORM-Bz                    | LSTM                        | C       | 6h        |  0.0225 |  8.7  | 1.13e-05 |  2.75 |  10 |
| STORM-Bz                    | LSTM                        | C       | 12h       | -0.0018 | -0.51 | 0.619    | -0.16 |  10 |
| STORM-Bz                    | LSTM                        | pooled  | 1h        |  0.0406 | 36.1  | 1.17e-25 |  6.59 |  30 |
| STORM-Bz                    | LSTM                        | pooled  | 6h        |  0.0273 | 10.91 | 8.83e-12 |  1.99 |  30 |
| STORM-Bz                    | LSTM                        | pooled  | 12h       |  0.0107 |  4.16 | 0.000255 |  0.76 |  30 |
| STORM-Bz                    | STORM, no delay             | A       | 1h        | -0.0015 | -9.59 | 5.05e-06 | -3.03 |  10 |
| STORM-Bz                    | STORM, no delay             | A       | 6h        | -0.0047 | -3.86 | 0.00385  | -1.22 |  10 |
| STORM-Bz                    | STORM, no delay             | A       | 12h       | -0.0012 | -0.53 | 0.606    | -0.17 |  10 |
| STORM-Bz                    | STORM, no delay             | B       | 1h        | -0.0008 | -3.38 | 0.00814  | -1.07 |  10 |
| STORM-Bz                    | STORM, no delay             | B       | 6h        | -0.0021 | -2.38 | 0.041    | -0.75 |  10 |
| STORM-Bz                    | STORM, no delay             | B       | 12h       | -0.0011 | -0.84 | 0.424    | -0.27 |  10 |
| STORM-Bz                    | STORM, no delay             | C       | 1h        | -0.0006 | -3.29 | 0.00938  | -1.04 |  10 |
| STORM-Bz                    | STORM, no delay             | C       | 6h        | -0.0044 | -2.58 | 0.0299   | -0.81 |  10 |
| STORM-Bz                    | STORM, no delay             | C       | 12h       | -0.0056 | -3.56 | 0.00613  | -1.13 |  10 |
| STORM-Bz                    | STORM, no delay             | pooled  | 1h        | -0.001  | -7.44 | 3.34e-08 | -1.36 |  30 |
| STORM-Bz                    | STORM, no delay             | pooled  | 6h        | -0.0037 | -4.88 | 3.56e-05 | -0.89 |  30 |
| STORM-Bz                    | STORM, no delay             | pooled  | 12h       | -0.0026 | -2.48 | 0.0192   | -0.45 |  30 |
| STORM-Bz                    | STORM, no gate              | A       | 1h        | -0.0001 | -0.42 | 0.687    | -0.13 |  10 |
| STORM-Bz                    | STORM, no gate              | A       | 6h        | -0.0032 | -1.84 | 0.0986   | -0.58 |  10 |
| STORM-Bz                    | STORM, no gate              | A       | 12h       | -0.0014 | -0.59 | 0.569    | -0.19 |  10 |
| STORM-Bz                    | STORM, no gate              | B       | 1h        |  0.0005 |  1.52 | 0.163    |  0.48 |  10 |
| STORM-Bz                    | STORM, no gate              | B       | 6h        |  0.0013 |  0.73 | 0.485    |  0.23 |  10 |
| STORM-Bz                    | STORM, no gate              | B       | 12h       | -0.002  | -1.11 | 0.296    | -0.35 |  10 |
| STORM-Bz                    | STORM, no gate              | C       | 1h        |  0.0003 |  1.73 | 0.118    |  0.55 |  10 |
| STORM-Bz                    | STORM, no gate              | C       | 6h        | -0.0031 | -1.19 | 0.266    | -0.38 |  10 |
| STORM-Bz                    | STORM, no gate              | C       | 12h       | -0.0039 | -1.8  | 0.105    | -0.57 |  10 |
| STORM-Bz                    | STORM, no gate              | pooled  | 1h        |  0.0002 |  1.64 | 0.113    |  0.3  |  30 |
| STORM-Bz                    | STORM, no gate              | pooled  | 6h        | -0.0017 | -1.36 | 0.184    | -0.25 |  30 |
| STORM-Bz                    | STORM, no gate              | pooled  | 12h       | -0.0024 | -2.04 | 0.0509   | -0.37 |  30 |
| Transformer + residual head | Transformer (matched)       | A       | 1h        |  0.0064 |  8.82 | 1.01e-05 |  2.79 |  10 |
| Transformer + residual head | Transformer (matched)       | A       | 6h        |  0.0052 |  2.96 | 0.016    |  0.94 |  10 |
| Transformer + residual head | Transformer (matched)       | A       | 12h       |  0.0002 |  0.1  | 0.92     |  0.03 |  10 |
| Transformer + residual head | Transformer (matched)       | B       | 1h        |  0.0075 | 11.94 | 8.04e-07 |  3.78 |  10 |
| Transformer + residual head | Transformer (matched)       | B       | 6h        |  0.0066 |  3.21 | 0.0106   |  1.02 |  10 |
| Transformer + residual head | Transformer (matched)       | B       | 12h       |  0.002  |  0.74 | 0.476    |  0.24 |  10 |
| Transformer + residual head | Transformer (matched)       | C       | 1h        |  0.0069 | 10    | 3.57e-06 |  3.16 |  10 |
| Transformer + residual head | Transformer (matched)       | C       | 6h        |  0.003  |  0.92 | 0.383    |  0.29 |  10 |
| Transformer + residual head | Transformer (matched)       | C       | 12h       |  0.002  |  0.54 | 0.603    |  0.17 |  10 |
| Transformer + residual head | Transformer (matched)       | pooled  | 1h        |  0.0069 | 17.8  | 3.76e-17 |  3.25 |  30 |
| Transformer + residual head | Transformer (matched)       | pooled  | 6h        |  0.0049 |  3.55 | 0.00133  |  0.65 |  30 |
| Transformer + residual head | Transformer (matched)       | pooled  | 12h       |  0.0014 |  0.87 | 0.391    |  0.16 |  30 |
| Transformer (matched)       | Transformer, flux-only      | A       | 1h        |  0.0031 |  1.44 | 0.182    |  0.46 |  10 |
| Transformer (matched)       | Transformer, flux-only      | A       | 6h        |  0.0376 | 14.26 | 1.75e-07 |  4.51 |  10 |
| Transformer (matched)       | Transformer, flux-only      | A       | 12h       |  0.0671 | 14.99 | 1.14e-07 |  4.74 |  10 |
| Transformer (matched)       | Transformer, flux-only      | B       | 1h        | -0.0008 | -0.51 | 0.623    | -0.16 |  10 |
| Transformer (matched)       | Transformer, flux-only      | B       | 6h        |  0.0327 | 14.51 | 1.5e-07  |  4.59 |  10 |
| Transformer (matched)       | Transformer, flux-only      | B       | 12h       |  0.0621 | 16.01 | 6.39e-08 |  5.06 |  10 |
| Transformer (matched)       | Transformer, flux-only      | C       | 1h        |  0.0002 |  0.11 | 0.913    |  0.04 |  10 |
| Transformer (matched)       | Transformer, flux-only      | C       | 6h        |  0.0306 |  6.75 | 8.41e-05 |  2.13 |  10 |
| Transformer (matched)       | Transformer, flux-only      | C       | 12h       |  0.0513 | 15.41 | 8.91e-08 |  4.87 |  10 |
| Transformer (matched)       | Transformer, flux-only      | pooled  | 1h        |  0.0008 |  0.77 | 0.448    |  0.14 |  30 |
| Transformer (matched)       | Transformer, flux-only      | pooled  | 6h        |  0.0336 | 17.55 | 5.48e-17 |  3.2  |  30 |
| Transformer (matched)       | Transformer, flux-only      | pooled  | 12h       |  0.0602 | 24.02 | 1.08e-20 |  4.39 |  30 |

## Gate / delay

```
{
 "tau_mean": 1.4955562353134155,
 "tau_sd": 0.019656971096992493,
 "tau_saturated_frac": 0.923218730103606,
 "tau_saturated_range_over_blocks": [
  0.8879972301300298,
  0.9727622851437131
 ],
 "gate_mean_storm": 0.44106483459472656,
 "gate_mean_quiet": 0.42394131422042847,
 "n_storm_windows_pooled": 10590,
 "gate_AUC_storm_vs_quiet_mean": 0.7603389035580865,
 "gate_AUC_sd": 0.10201232724152647,
 "gate_AUC_n": 30
}
```