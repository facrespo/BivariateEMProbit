import numpy as np;
import pickle;
import csv;
import io;
import six;
import pdb,time,sys;

from scipy import stats;
from scipy import linalg;
import statsmodels.api as sm;
import emoneProbitf;
import embiprobit;
import os;
import random;
import argparse;
description = """
This program provides basic genome-wide association (GWAS) functionality.  
You provide a phenotype, covariables and genotype file and the program outputs a result file with information about each SNP, including the association p-value. 
The input file are all standard plink formatted with the first two columns specifiying the individual and family ID.  
For the phenotype file, we accept either NA or -9 to denote missing values.  

Usage:

      python run_program.py --file File --dd ; --p number of covariates --n number of cases --nsnps number of SNPs that you like use --pprob number of variable that we like to calculate the p-value --error numerical error --maxiter maximal number of iteration --grid grid for integration --alpha level of statistical test --eps tolerance of  normal bivariate distribution --output outputfile.txt

      python run_program.py --file Consulta1.txt --dd ; --p 3 --n 882 --nsnps 8 --pprob 3 --output result
"""

parser = argparse.ArgumentParser(description);
parser.add_argument("--file",dest="file",help="file to run the model");
parser.add_argument("--dd",dest="delimiter",help="delimiter of file");
parser.add_argument("--p",dest="p",help="number of covariates");
parser.add_argument("--n",dest="n",help="number of cases");
parser.add_argument("--nsnps",dest="phenotypes",help="number of SNPS that we use in the calculate");
parser.add_argument("--pprob",dest="pprob",help="variable that we like calculate of test (for definition is the first variable)",default=1);
parser.add_argument("--error",dest="error",help="numerical error (default 0.00001)",default=0.00001);
parser.add_argument("--maxiter",dest="maxiter",help="maximal iteration (default 80)",default=80);
parser.add_argument("--grid",dest="grid",help="Grid for integration (default 0.1)",default=0.1);
parser.add_argument("--alpha",dest="alpha",help="alpha to test (default 0.05)",default=0.05);
parser.add_argument("--eps",dest="eps",help="tolerance of normal bivariate distribution",default=1e-6);
parser.add_argument("--output",dest="output",help="name of output file in txt format with ',' separator ", default="result");

args = parser.parse_args();

datos= csv.reader(open(args.file),delimiter=args.delimiter);
r=1;
p=int(args.p);
n=int(args.n);
error=float(args.error);
maxiter=float(args.maxiter);
delta_grid=float(args.grid);
alpha=float(args.alpha);
p_special_test=int(args.pprob);
abseps=float(args.eps);
fenotipos=int(args.phenotypes);
output_file=args.output;

columnas=fenotipos+p+1;

perturbation=0.000001;


fich=open(output_file +".txt","w");
foo=open(args.file);
datos= csv.reader(foo,delimiter=args.delimiter);
Vars=next(datos);
foo.close();
fich.write("Model,Correlation,number_iteration_bivariate,");
for ii in range(0,p+1):
    if ii<=(p-2):
        fich.write(Vars[2+ii] +"_1" + str(ii+1) + ",");
    if ii==(p-1):
        fich.write("SNP_1" + str(ii+1) + ",");
    if ii==p:
        fich.write("Constant_1" + str(ii+1) + ",");
for ii in range(0,p+1):
    if ii<=(p-2):
        fich.write(Vars[2+ii] +"_2" + str(ii+1) + ",");
    if ii==(p-1):
        fich.write("SNP_2"  + str(ii+1) + ",");
    if ii==p:
        fich.write("Constant_2"  + str(ii+1) + ",");
fich.write("ErrorVariable1,ErrorVariable2,CorrelationEstimateReal1,CorrelationEstimateReal2,loglike_only_constant,loglikemodelcomplet,loglikewithoutSNP,pseudorsquared,test_log_likehood,p-value,testcortebivariateconstantH01,estadisticobivariateconstantH01, pvaluebivariateconstantH01, testcortebivariateSNPH01,estadisticobivariateSNPH01, pvaluebivariateSNPH01, logverotsy1, tcortepspecialsy1, testpspecialsy1, pvaluepspecialsy1, logverotsy2, tcortepspecialsy2, testpspecialsy2, pvaluepspecialsy2, number_iteration_modelY1,");
for ii in range(0,p+1):
    if ii<=(p-2):
        fich.write(Vars[2+ii] +"_1,");
    if ii==(p-1):
        fich.write("SNP_1" + ",");
    if ii==p:
        fich.write("Constant_1"  + ",");
fich.write("ErrorVariableY1,CorrelationEstimateRealY1,loglike_only_constant_m1,loglike_without_SNPY1,logmodelm1,pseudorsquared_m1,test_likehood_m1,p-value_m1,testcorteY1constantH01,estadisticoY1constantH01, pvalueY1constantH01, testcorteY1SNPH01,estadisticoY1SNPH01, pvalueY1SNPH01, number_iteration_modelY2,");
for ii in range(0,p+1):
    if ii<=(p-2):
        fich.write(Vars[2+ii] +"_2,");
    if ii==(p-1):
        fich.write("SNP_2" + ",");
    if ii==p:
        fich.write("Constant_2"  + ",");
fich.write("ErrorVariableY2,CorrelationEstimateRealY2,loglike_only_constant_m2,loglike_without_SNPY2,logmodelm2,pseudorsquared_m2,test_likehood2,p-value_m2,testcorteY2constantH01,estadisticoY2constantH01, pvalueY2constantH01, testcorteY2SNPH01,estadisticoY2SNPH01, pvalueY2SNPH01\n");
fich.close();

for l in range(0,fenotipos):
              foo=open(args.file);
              datos= csv.reader(foo,delimiter=args.delimiter);
              #print(datos);
              #print("Modelo:",l);
              Y1=np.zeros((n,r));
              Y2=np.zeros((n,r)); 
              X=np.zeros((n,p));
              for i,fila in enumerate(datos):
                  if i==0:
                     Vars=fila;
                  else:
                      t=1
                      for j in range(0,(columnas)):
                          if j==0: 
                             Y1[i-1]=float(fila[j]);
                          elif j==1:
                             Y2[i-1]=float(fila[j]);
                          if j>=2:
                             if (t<=(p-1)):
                                 if j==(t+1):
                                    X[i-1,(t-1)]=float(fila[j]);
                                    t=t+1;
                             if (t==p):
                                 if j==(t+1+l):
                                    X[i-1,(t-1)]=float(fila[j]);
                             if j>=(t+1+l):
                                j=columnas;
                                                              
              foo.close();

              Xcont = sm.add_constant(X, prepend=False); 
              X=np.asmatrix(Xcont);

              rho, Sigma, iteracion, Btt1, Btt2, vt1, vt2, error_predicciont1, error_predicciont2, Rsquare1, Rsquare2, lognull, logmodelocompleto, logmwSNP, scoret1, scoret2, score_rho, pseudorsquared, llrf, llrf_pvalue, marginalt, Desv_b1, Desv_b2, Desv_rho, Tfinal1, Tfinal2, P_value1, P_value2, tcortemb, testmb, pvaluemb, tcortepspecial, testpspecial, pvaluepspecial, logverotppy1, tcortepspecialy1, testpspecialy1, pvaluepspecialy1, logverotppy2, tcortepspecialy2, testpspecialy2, pvaluepspecialy2 = embiprobit.embiprobit(Y1, Y2, X, maxiter, error, perturbation, alpha, abseps, p_special_test);

              iteracionesm1, Bponderadorm1, vprediccionm1, error_prediccionm1, Rsquarem1, lognullm1, logmreducpm1, logverofm1, scorefm1, pseudorsquaredm1, llrfm1, llrf_pvaluem1, effect_marginalm1, Desv_finalm1, Tfinalm1, P_valuem1, tcortem1, testm1, pvaluem1, tcortepspecialm1, testpspecialm1, pvaluepspecialm1 = emoneProbitf.emProbit(Y1, X, maxiter, error, perturbation, alpha, p_special_test);

              iteracionesm2, Bponderadorm2, vprediccionm2, error_prediccionm2, Rsquarem2, lognullm2, logmreducpm2, logverofm2, scorefm2, pseudorsquaredm2, llrfm2, llrf_pvaluem2, effect_marginalm2, Desv_finalm2, Tfinalm2, P_valuem2, tcortem2, testm2, pvaluem2, tcortepspecialm2, testpspecialm2, pvaluepspecialm2 = emoneProbitf.emProbit(Y2, X, maxiter, error, perturbation, alpha, p_special_test);

              fich=open(output_file +".txt","a");
              fich.write(Vars[p+1+l] + "," + str(rho) + "," + str(iteracion) + "," + str(Btt1[0,0]) + "," + str(Btt1[1,0]) + "," + str(Btt1[2,0]) + "," + str(Btt1[3,0]) + ",");
              fich.write(str(Btt2[0,0]) + "," + str(Btt2[1,0]) + "," + str(Btt2[2,0]) + "," + str(Btt2[3,0]) + "," + str(error_predicciont1[0,0]) + ",");
              fich.write(str(error_predicciont2[0,0]) + "," + str(Rsquare1) + "," + str(Rsquare2) + "," + str(lognull) + "," + str(logmodelocompleto) + "," + str(logmwSNP) + ",");
              fich.write(str(pseudorsquared) + "," + str(llrf) + "," + str(llrf_pvalue) + ",");
              fich.write(str(tcortemb) + "," + str(testmb) + "," + str(pvaluemb) + "," +  str(tcortepspecial) + "," + str(testpspecial) + "," + str(pvaluepspecial) + ",");
              fich.write(str(logverotppy1) + "," + str(tcortepspecialy1) + "," + str(testpspecialy1) + "," + str(pvaluepspecialy1) + "," + str(logverotppy2) + "," + str(tcortepspecialy2) + "," + str(testpspecialy2) + "," + str(pvaluepspecialy2) + ",");              
              fich.write(str(iteracionesm1) + ",");
              for ii in range(0,p+1):
                  fich.write(str(Bponderadorm1[ii,0])+",");
              fich.write(str(error_prediccionm1[0,0]) + ",");
              fich.write(str(Rsquarem1) + "," + str(lognullm1) + "," + str(logmreducpm1) + ","  + str(logverofm1) + "," + str(pseudorsquaredm1) + "," + str(llrfm1) + "," + str(llrf_pvaluem1) + ",");
              fich.write(str(tcortem1) + "," + str(testm1) + "," + str(pvaluem1) + "," +  str(tcortepspecialm1) + "," + str(testpspecialm1) + "," + str(pvaluepspecialm1) + ",");
              fich.write(str(iteracionesm2) + ",");
              for ii in range(0,p+1):
                  fich.write(str(Bponderadorm2[ii,0])+",");              
              fich.write(str(error_prediccionm2[0,0]) + ",");
              fich.write(str(Rsquarem2) + "," + str(lognullm2) + "," + str(logmreducpm2) + "," + str(logverofm2) + "," + str(pseudorsquaredm2) + "," + str(llrfm2) + "," + str(llrf_pvaluem2) + ",");
              fich.write(str(tcortem2) + "," + str(testm2) + "," + str(pvaluem2) + "," +  str(tcortepspecialm2) + "," + str(testpspecialm2) + "," + str(pvaluepspecialm2) + "\n");                         
              fich.close();

