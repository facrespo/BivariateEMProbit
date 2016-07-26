# BivariateEMProbit
This repository has the function and examples to run the Bivariate and Univariate Probit using the EM-Algorithm applied in Bioinformatics.

We use the libraries: Numpy, Scipy, Sympy, Math, statsmodels.api, and Python 3.5 with Anaconda.

The emoneProbitf.py is the function to univariate Probit calculated with EM algorithm with multivariate X predictor.

The embiprobit.py is the function to bivariate Probir calculated with EM algorithm with multivariate X predictor.

The sample_green.py is one ejecution using the file ejemplo_greenvf.csv, the result is show in screen. The constant is agregated in X with function X = sm.add_constant(X, prepend=False).

cargar_archivo_prueba_3.py is an example to calculate models with SNPs, it use the file Consulta1.csv. Consulta1.csv have the variables: FID, IID, Hyp0, hyp1, SEX, age0, age1, rs8063330_A, rs8062734_A, rs8058318_A, rs892244_A, rs745519_A, rs7185064_A, rs16957838_A, rs8049208_A.
We get Hyp0 as Y1, hyp1 as Y2, and X with SEX, age1, and some SNPs. For this cases cargar_archivo_prueba_3.py make 9 models with SNps and one model without SNPs. The idea is proof if the SNPs generate a better models that without theirs. 
The output of cargar_archivo_prueba_3.py are resultado.txt and time.txt. In resultado.txt appear the model we get its. In time.txt appear the running times.
