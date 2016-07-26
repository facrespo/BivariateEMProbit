import numpy as np;
import pickle;
import csv;
import io;
import six;
#import pandas;
#from pandas.io.parsers import read_csv;
#from io import StringIO;

#import sys;
#import builtins;

from scipy import stats;
import matplotlib as plt;
import statsmodels.api as sm;
import statsmodels.formula.api as smf;
from statsmodels.formula.api import logit, probit, poisson, ols;
import emoneProbitf;
import embiprobitcontinue;
import classicmodel;
from time import time;
#from emoneProbit import emProbit;

#datos=pandas.read_csv('Eco2-Credito.csv', index_col=0);
#print(datos);
datos= csv.reader(open("ejemplo_greenvf1.csv"),delimiter=",");
n=95;
r=1;
p=2;

datosn=np.zeros((n,p+2));

for i,fila in enumerate(datos):
    if i==0:
         Vars=fila;
    else:
         for j in range(0,p+2):
            datosn[i-1,j]=float(fila[j]);
         #numpy.genfromtxt(StringIO(fila), delimiter=",")
          #print(fila);

#print(datosn);

datosn=np.asmatrix(datosn);

#print(datosn);

Y1=np.zeros((n,r));
Y2=np.zeros((n,r));
X=np.zeros((n,p));

Y1=datosn[0:n,0];
Y2=datosn[0:n,1];
X[0:n,0:p]=datosn[0:n,2:(p+2)];

#print(Y1);
#print(Y2);
#print(X);

#mod = sm.OLS(Y, X);

#res = mod.fit();

#print(res.summary());

Xexog = sm.add_constant(X, prepend=False);

#Xexog2=np.zeros((n,2));

#for i in range(0,n):
#    Xexog2[i,0] = X[i,1];
#    Xexog2[i,1] = X[i,2];


#print(Xexog2);
#print(Xexog);

#logit_mod = sm.Logit(Y, Xexog);

#logit_res = logit_mod.fit();

#print(logit_res.params);

#print(logit_res.summary());

###probit_mod = sm.Probit(Y1, Xexog);

#probit_mod2 = sm.Probit(Y, Xexog2);

###probit_res = probit_mod.fit();

#probit_res2 = probit_mod2.fit();

###print(probit_res.params);

#probit_margeff = probit_res.get_margeff();

#print(probit_margeff.summary());

###print(probit_res.summary());

#print(probit_res2.params);

#print(probit_res2.summary());

error=0.0001;
maxiter=80;
X=Xexog;
perturbation=0.000001;
alpha=0.05;
delta_grid=0.1;
ipower=0; #is 1 if you like to get the power function
abseps=1e-6; #Tolerance to normal bivariate distribution
p_special_test=2;

#est = sm.OLS(Y2, X);
#est = est.fit();
#print(est.summary());

#probit_mod = sm.Probit(Y1, X);
#probit_res = probit_mod.fit();
#print(probit_res.params);
#print(probit_res.summary());

start_time = time();
rho, Sigma, iteracion, Btt1, Btt2, vt1, vt2, error_predicciont1, error_predicciont2, Rsquare1, Rsquare2, lognull, logmodelocompleto, logmwSNP, scoret1, scoret2, score_rho, pseudorsquared, llrf, llrf_pvalue, marginalt, Desv_b1, Desv_b2, Desv_rho, Tfinal1, Tfinal2, P_value1, P_value2, tcortemb, testmb, pvaluemb, tcortepspecial, testpspecial, pvaluepspecial, logverotppy1, tcortepspecialy1, testpspecialy1, pvaluepspecialy1, logverotppy2, tcortepspecialy2, testpspecialy2, pvaluepspecialy2 = embiprobitcontinue.embiprobitcontinue(Y1, Y2, X, maxiter, error, perturbation, alpha, abseps, p_special_test);
#rho, Sigma, iteracion, Btt1, Btt2, vt1, vt2, error_predicciont1, error_predicciont2, Rsquare1, Rsquare2, lognull, logmodelocompleto, logmwSNP, scoret1, scoret2, score_rho, pseudorsquared, llrf, llrf_pvalue, marginalt, Desv_b1, Desv_b2, Desv_rho, Tfinal1, Tfinal2, P_value1, P_value2, tcortemb, testmb, pvaluemb, tcortepspecial, testpspecial, pvaluepspecial, logverotppy1, tcortepspecialy1, testpspecialy1, pvaluepspecialy1, logverotppy2, tcortepspecialy2, testpspecialy2, pvaluepspecialy2 = embiprobit.embiprobit(Y1, Y2, X, maxiter, error, perturbation, alpha, abseps, p_special_test);


print(rho);
print(Sigma);
print(iteracion);
print(Btt1);
print(Btt2);
#print(vprediccion);
print(error_predicciont1);
print(error_predicciont2);
print(Rsquare1);
print(Rsquare2);
print(lognull);
print(logmodelocompleto);
print(logmwSNP);
print(scoret1);
print(scoret2);
print(score_rho);
print(pseudorsquared);
print(llrf);
print(llrf_pvalue);
print(marginalt);
print(Desv_b1);
print(Desv_b2);
print(Desv_rho);
print(Tfinal1);
print(Tfinal2);
print(P_value1);
print(P_value2);
print(tcortemb);
print(testmb);
print(pvaluemb);
print(tcortepspecial);
print(testpspecial);
print(pvaluepspecial);
print(logverotppy1);
print(tcortepspecialy1);
print(testpspecialy1);
print(pvaluepspecialy1);
print(logverotppy2);
print(tcortepspecialy2);
print(testpspecialy2);
print(pvaluepspecialy2);

time_lecture = time() - start_time;
print(time_lecture);
start_time = time();

mod = sm.OLS(Y2, X);
res = mod.fit();
print(res.summary());

time_lecture = time() - start_time;
print(time_lecture);
start_time = time();


Bponderadorm1, vprediccionm1, error_prediccionm1, Rsquarem1, lognullm1, logverwsnpm1, logverofm1, scorefm1, pseudorsquaredm1, llrfm1, llrf_pvaluem1, effect_marginalm1, Desv_finalm1, Tfinalm1, P_valuem1, tcortem1, testm1, pvaluem1, tcortepspecialm1, testpspecialm1, pvaluepspecialm1 = classicmodel.linealmodel(Y2, X, maxiter, error, perturbation, alpha, delta_grid, ipower, p_special_test);

print(Bponderadorm1);
print(error_prediccionm1);
print(Rsquarem1);
print(lognullm1);
print(logverwsnpm1);
print(logverofm1);
print(scorefm1);
print(pseudorsquaredm1);
print(llrfm1);
print(llrf_pvaluem1);
print(effect_marginalm1);
print(Desv_finalm1);
print(Tfinalm1);
print(P_valuem1);
print(tcortem1);
print(testm1);
print(pvaluem1);
print(tcortepspecialm1);
print(testpspecialm1);
print(pvaluepspecialm1);

time_lecture = time() - start_time;
print(time_lecture);
start_time = time();


iteracionesm2, Bponderadorm2, vprediccionm2, error_prediccionm2, Rsquarem2, lognullm2, logverwsnpm2, logverofm2, scorefm2, pseudorsquaredm2, llrfm2, llrf_pvaluem2, effect_marginalm2, Desv_finalm2, Tfinalm2, P_valuem2, tcortem2, testm2, pvaluem2, tcortepspecialm2, testpspecialm2, pvaluepspecialm2 = emoneProbit.emProbit(Y1, X, maxiter, error, perturbation, alpha, delta_grid, ipower, p_special_test);

print(iteracionesm2);
print(Bponderadorm2);
#print(vprediccion);
print(error_prediccionm2);
print(Rsquarem2);
print(lognullm2);
print(logverwsnpm2);
print(logverofm2);
print(scorefm2);
print(pseudorsquaredm2);
print(llrfm2);
print(llrf_pvaluem2);
print(Desv_finalm2);
print(P_valuem2);
print(tcortem2);
print(testm2);
print(pvaluem2);
print(tcortepspecialm2);
print(testpspecialm2);
print(pvaluepspecialm2);

time_lecture = time() - start_time;
print(time_lecture);
start_time = time();
