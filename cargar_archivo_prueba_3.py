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
import embiprobit;
from time import time;
#from emoneProbit import emProbit;

#datos=pandas.read_csv('Eco2-Credito.csv', index_col=0);
#print(datos);
#datos= csv.reader(open("Consulta1.csv"),delimiter=",");
n=882;
columnas=15;
fenotipos=8;
r=1;
p=3;

error=0.01;
maxiter=80;
perturbation=0.000001;
alpha=0.05;
delta_grid=0.1;
ipower=0; #is 1 if you like to get the power function
abseps=1e-6; #Tolerance to normal bivariate distribution
p_special_test=3;

fich=open("resultado.txt","w");
fich.write("Model,Correlation,number_iteration_bivariate,B1[1],B1[2],B1[3],B1[4],B2[1],B2[2],B2[3],B2[4],ErrorVariable1,ErrorVariable2,CorrelationEstimateReal1,CorrelationEstimateReal2,logwconstant,logmodelcomplet,logwSNP,pseudorsquared,test_log_likehood,p-value,Desesv_B1[1],Desesv_B1[2],Desesv_B1[3],Desesv_B1[4],Desesv_B2[1],Desesv_B2[2],Desesv_B2[3],Desesv_B2[4],Desv_rho,Test_B1[1],Test_B1[2],Test_B1[3],Test_B1[4],Test_B2[1],Test_B2[2],Test_B2[3],Test_B2[4],pvalue_B1[1],pvalue_B1[2],pvalue_B1[3],pvalue_B1[4],pvalue_B2[1],pvalue_B2[2],pvalue_B2[3],pvalue_B2[4], testcortebivariateconstantH01,estadisticobivariateconstantH01, pvaluebivariateconstantH01, testcortebivariateSNPH01,estadisticobivariateSNPH01, pvaluebivariateSNPH01, logverotsy1, tcortepspecialsy1, testpspecialsy1, pvaluepspecialsy1, logverotsy2, tcortepspecialsy2, testpspecialsy2, pvaluepspecialsy2, number_iteration_modelY1,B1[1],B1[2],B1[3],B1[4],ErrorVariable1,CorrelationEstimateRealY1,logverconstm1,logverwSNPY1,logmodelm1,pseudorsquared,test_likehood,p-value,Desesv_B1[1],Desesv_B1[2],Desesv_B1[3],Desesv_B1[4],Test_B1[1],Test_B1[2],Test_B1[3],Test_B1[4],pvalue_B1[1],pvalue_B1[2],pvalue_B1[3],pvalue_B1[4], testcorteY1constantH01,estadisticoY1constantH01, pvalueY1constantH01, testcorteY1SNPH01,estadisticoY1SNPH01, pvalueY1SNPH01, number_iteration_modelY2,B2[1],B2[2],B2[3],B2[4],ErrorVariable2,CorrelationEstimateRealY2,logverconstm2,logverwSNPY2,logmodelm2,pseudorsquared2,test_likehood2,p-value2,Desesv_B2[1],Desesv_B2[2],Desesv_B2[3],Desesv_B2[4],Test_B2[1],Test_B2[2],Test_B2[3],Test_B2[4],pvalue_B2[1],pvalue_B2[2],pvalue_B2[3],pvalue_B2[4], testcorteY2constantH01,estadisticoY2constantH01, pvalueY2constantH01, testcorteY2SNPH01,estadisticoY2SNPH01, pvalueY2SNPH01\n");
fich.close();

fich=open("time.txt","w");
fich.write("Model,Time_Open_File_SNP,Time_Bivariate,Time_Y1_Probit,Time_Y2_Probit,Time_write_results\n");
fich.close();

for l in range(-1,fenotipos):
      if (l==-1):
              start_time = time();
              p=2;
              Y1=np.zeros((n,r));
              Y2=np.zeros((n,r)); 
              X=np.zeros((n,p));
              foo=open("Consulta1.csv");
              datos= csv.reader(foo,delimiter=",");
              
              for i,fila in enumerate(datos):
                  if i==0:
                     Vars=fila;
                  else:
                      for j in range(0,(columnas)):
                          if j<=1:
                             j=1;
                          elif j==2: 
                             Y1[i-1]=float(fila[j]);
                          elif j==3:
                             Y2[i-1]=float(fila[j]);
                          elif j==4:
                              X[i-1,0]=float(fila[j]);
                          elif j==6:
                              X[i-1,1]=float(fila[j]);  
                          else:
                              j=columnas;
                      
          

              foo.close();
              print("Modelo:",l);
              Xcont = sm.add_constant(X, prepend=False); 
              X=np.asmatrix(Xcont);

              time_lecture = time() - start_time;
              start_time = time();
               
              rho, Sigma, iteracion, Btt1, Btt2, vt1, vt2, error_predicciont1, error_predicciont2, Rsquare1, Rsquare2, lognull, logmodelocompleto, logmwSNP, scoret1, scoret2, score_rho, pseudorsquared, llrf, llrf_pvalue, marginalt, Desv_b1, Desv_b2, Desv_rho, Tfinal1, Tfinal2, P_value1, P_value2, tcortemb, testmb, pvaluemb, tcortepspecial, testpspecial, pvaluepspecial, logverotppy1, tcortepspecialy1, testpspecialy1, pvaluepspecialy1, logverotppy2, tcortepspecialy2, testpspecialy2, pvaluepspecialy2 = embiprobit.embiprobit(Y1, Y2, X, maxiter, error, perturbation, alpha, abseps, p_special_test);

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
              print(logmwSNP)
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

              time_snp = time() - start_time;
              start_time = time();

              iteracionesm1, Bponderadorm1, vprediccionm1, error_prediccionm1, Rsquarem1, lognullm1, logmreducpm1, logverofm1, scorefm1, pseudorsquaredm1, llrfm1, llrf_pvaluem1, effect_marginalm1, Desv_finalm1, Tfinalm1, P_valuem1, tcortem1, testm1, pvaluem1, tcortepspecialm1, testpspecialm1, pvaluepspecialm1 = emoneProbitf.emProbit(Y1, X, maxiter, error, perturbation, alpha, p_special_test);

              print(iteracionesm1);
              print(Bponderadorm1);
              #print(vprediccionm1);
              print(error_prediccionm1);
              print(Rsquarem1);
              print(lognullm1);
              print(logmreducpm1);
              print(logverofm1);
              print(scorefm1);
              print(pseudorsquaredm1);
              print(llrfm1);
              print(llrf_pvaluem1);
              print(Desv_finalm1);
              print(Tfinalm1);
              print(P_valuem1);
              print(tcortem1);
              print(testm1);
              print(pvaluem1);
              print(tcortepspecialm1);
              print(testpspecialm1);
              print(pvaluepspecialm1);

              time_y1 = time() - start_time;
              start_time = time();
                            
              iteracionesm2, Bponderadorm2, vprediccionm2, error_prediccionm2, Rsquarem2, lognullm2, logmreducpm2, logverofm2, scorefm2, pseudorsquaredm2, llrfm2, llrf_pvaluem2, effect_marginalm2, Desv_finalm2, Tfinalm2, P_valuem2, tcortem2, testm2, pvaluem2, tcortepspecialm2, testpspecialm2, pvaluepspecialm2 = emoneProbitf.emProbit(Y2, X, maxiter, error, perturbation, alpha, p_special_test);

              print(iteracionesm2);
              print(Bponderadorm2);
              #print(vprediccionm2);
              print(error_prediccionm2);
              print(Rsquarem2);
              print(lognullm2);
              print(logmreducpm2);
              print(logverofm2);
              print(scorefm2);
              print(pseudorsquaredm2);
              print(llrfm2);
              print(llrf_pvaluem2);
              print(Desv_finalm2);
              print(Tfinalm2);
              print(P_valuem2);
              print(tcortem2);
              print(testm2);
              print(pvaluem2);
              print(tcortepspecialm2);
              print(testpspecialm2);
              print(pvaluepspecialm2);

              time_y2 = time() - start_time;
              start_time = time();
                            
              fich=open("resultado.txt","a");
              fich.write("Inicial," + str(rho) + "," + str(iteracion) + "," + str(Btt1[0,0]) + "," + str(Btt1[1,0]) + "," + str(0) + "," + str(Btt1[2,0]) + ",");
              fich.write(str(Btt2[0,0]) + "," + str(Btt2[1,0]) + "," + str(0) + "," + str(Btt2[2,0]) + "," + str(error_predicciont1[0,0]) + ",");
              fich.write(str(error_predicciont2[0,0]) + "," + str(Rsquare1) + "," + str(Rsquare2) + "," + str(lognull) + "," + str(logmodelocompleto) + "," + str(logmwSNP) + ",");
              fich.write(str(pseudorsquared) + "," + str(llrf) + "," + str(llrf_pvalue) + "," +  str(Desv_b1[0,0]) + "," + str(Desv_b1[1,0]) + "," + str(0) + ",");
              fich.write(str(Desv_b1[2,0]) +  "," + str(Desv_b2[0,0]) + "," + str(Desv_b2[1,0]) + "," + str(0) + ",");
              fich.write(str(Desv_b2[2,0]) +  "," + str(Desv_rho[0]) + ",");
              fich.write(str(Tfinal1[0,0]) + "," + str(Tfinal1[1,0]) + "," + str(0) + "," + str(Tfinal1[2,0]) +  "," + str(Tfinal2[0,0]) + "," + str(Tfinal2[1,0]) + "," + str(0) + "," + str(Tfinal2[2,0]) + ",");
              fich.write(str(P_value1[0,0]) + "," + str(P_value1[1,0]) + "," + str(0) + "," + str(P_value1[2,0]) + "," + str(P_value2[0,0]) + "," + str(P_value2[1,0]) + "," + str(0) + "," + str(P_value2[2,0]) + ",");
              fich.write(str(tcortemb) + "," + str(testmb) + "," + str(pvaluemb) + "," +  str(tcortepspecial) + "," + str(testpspecial) + "," + str(pvaluepspecial) + ",");
              fich.write(str(logverotppy1) + "," + str(tcortepspecialy1) + "," + str(testpspecialy1) + "," + str(pvaluepspecialy1) + "," + str(logverotppy2) + "," + str(tcortepspecialy2) + "," + str(testpspecialy2) + "," + str(pvaluepspecialy2) + ",");
              fich.write(str(iteracionesm1) + "," + str(Bponderadorm1[0,0]) + "," + str(Bponderadorm1[1,0]) + "," + str(0) + "," + str(Bponderadorm1[2,0]) + "," + str(error_prediccionm1[0,0]) + ",");
              fich.write(str(Rsquarem1) + "," + str(lognullm1) + ","  + str(logmreducpm1) + ","  + str(logverofm1) + "," + str(pseudorsquaredm1) + "," + str(llrfm1) + "," + str(llrf_pvaluem1) + ",");
              fich.write(str(Desv_finalm1[0,0]) + "," + str(Desv_finalm1[1,0]) + "," + str(0) + "," + str(Desv_finalm1[2,0]) +  ","  + str(Tfinalm1[0,0]) + "," + str(Tfinalm1[1,0]) + ",");
              fich.write(str(0) + "," + str(Tfinalm1[2,0]) + "," + str(P_valuem1[0,0]) + "," + str(P_valuem1[1,0]) + "," + str(0) + "," + str(P_valuem1[2,0]) + ",");
              fich.write(str(tcortem1) + "," + str(testm1) + "," + str(pvaluem1) + "," +  str(tcortepspecialm1) + "," + str(testpspecialm1) + "," + str(pvaluepspecialm1) + ",");
              fich.write(str(iteracionesm2) + "," + str(Bponderadorm2[0,0]) + "," + str(Bponderadorm2[1,0]) + "," + str(0) + "," + str(Bponderadorm2[2,0]) + "," + str(error_prediccionm2[0,0]) + ",");
              fich.write(str(Rsquarem2) + "," + str(lognullm2) + ","  + str(logmreducpm2) + ","  + str(logverofm2) + "," + str(pseudorsquaredm2) + "," + str(llrfm2) + "," + str(llrf_pvaluem2) + ",");
              fich.write(str(Desv_finalm2[0,0]) + "," + str(Desv_finalm2[1,0]) + "," + str(0) + "," + str(Desv_finalm2[2,0]) +  ","  + str(Tfinalm2[0,0]) + "," + str(Tfinalm2[1,0]) + ",");
              fich.write(str(0) + "," + str(Tfinalm2[2,0]) + "," + str(P_valuem2[0,0]) + "," + str(P_valuem2[1,0]) + "," + str(0) + "," + str(P_valuem2[2,0])  + ",");
              fich.write(str(tcortem2) + "," + str(testm2) + "," + str(pvaluem2) + "," +  str(tcortepspecialm2) + "," + str(testpspecialm2) + "," + str(pvaluepspecialm2) + "\n");
              fich.close();
              
              time_write = time() - start_time;

              fich=open("time.txt","a");
              fich.write("Inicial," + str(time_lecture) + "," + str(time_snp) + "," + str(time_y1) + "," + str(time_y2) + "," + str(time_write) + "\n");
              fich.close();


      else:
              start_time = time();
              foo=open("Consulta1.csv");
              datos= csv.reader(foo,delimiter=",");
              #print(datos);
              print("Modelo:",l);
              p=3;
              Y1=np.zeros((n,r));
              Y2=np.zeros((n,r)); 
              X=np.zeros((n,p));
              for i,fila in enumerate(datos):
                  if i==0:
                     Vars=fila;
                  else:
                      for j in range(0,(columnas)):
                          if j<=1:
                             j=1;
                          elif j==2: 
                             Y1[i-1]=float(fila[j]);
                          elif j==3:
                             Y2[i-1]=float(fila[j]);
                          elif j==4:
                              X[i-1,0]=float(fila[j]);
                          elif j==6:
                              X[i-1,1]=float(fila[j]);
                          elif j==(7+l):
                              X[i-1,2]=float(fila[j]);  
                          else:
                              j=columnas;
                      



              foo.close();

              Xcont = sm.add_constant(X, prepend=False); 
              X=np.asmatrix(Xcont);

              time_lecture = time() - start_time;
              start_time = time();
              
              rho, Sigma, iteracion, Btt1, Btt2, vt1, vt2, error_predicciont1, error_predicciont2, Rsquare1, Rsquare2, lognull, logmodelocompleto, logmwSNP, scoret1, scoret2, score_rho, pseudorsquared, llrf, llrf_pvalue, marginalt, Desv_b1, Desv_b2, Desv_rho, Tfinal1, Tfinal2, P_value1, P_value2, tcortemb, testmb, pvaluemb, tcortepspecial, testpspecial, pvaluepspecial, logverotppy1, tcortepspecialy1, testpspecialy1, pvaluepspecialy1, logverotppy2, tcortepspecialy2, testpspecialy2, pvaluepspecialy2 = embiprobit.embiprobit(Y1, Y2, X, maxiter, error, perturbation, alpha, abseps, p_special_test);

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
              print(logmwSNP)
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

              time_snp = time() - start_time;
              start_time = time();
              
              #probit_mod = sm.Probit(Y1, X);
              #probit_res = probit_mod.fit();
              #print(probit_res.params);
              #print(probit_res.summary());
              #probit_margeff = probit_res.get_margeff();
              #print(probit_margeff.summary());

              iteracionesm1, Bponderadorm1, vprediccionm1, error_prediccionm1, Rsquarem1, lognullm1, logmreducpm1, logverofm1, scorefm1, pseudorsquaredm1, llrfm1, llrf_pvaluem1, effect_marginalm1, Desv_finalm1, Tfinalm1, P_valuem1, tcortem1, testm1, pvaluem1, tcortepspecialm1, testpspecialm1, pvaluepspecialm1 = emoneProbitf.emProbit(Y1, X, maxiter, error, perturbation, alpha, p_special_test);

              print(iteracionesm1);
              print(Bponderadorm1);
              #print(vprediccionm1);
              print(error_prediccionm1);
              print(Rsquarem1);
              print(lognullm1);
              print(logmreducpm1);
              print(logverofm1);
              print(scorefm1);
              print(pseudorsquaredm1);
              print(llrfm1);
              print(llrf_pvaluem1);
              print(Desv_finalm1);
              print(Tfinalm1);
              print(P_valuem1);
              print(tcortem1);
              print(testm1);
              print(pvaluem1);
              print(tcortepspecialm1);
              print(testpspecialm1);
              print(pvaluepspecialm1);

              time_y1 = time() - start_time;
              start_time = time();

              iteracionesm2, Bponderadorm2, vprediccionm2, error_prediccionm2, Rsquarem2, lognullm2, logmreducpm2, logverofm2, scorefm2, pseudorsquaredm2, llrfm2, llrf_pvaluem2, effect_marginalm2, Desv_finalm2, Tfinalm2, P_valuem2, tcortem2, testm2, pvaluem2, tcortepspecialm2, testpspecialm2, pvaluepspecialm2 = emoneProbitf.emProbit(Y2, X, maxiter, error, perturbation, alpha, p_special_test);

              print(iteracionesm2);
              print(Bponderadorm2);
              #print(vprediccionm2);
              print(error_prediccionm2);
              print(Rsquarem2);
              print(lognullm2);
              print(logmreducpm2);
              print(logverofm2);
              print(scorefm2);
              print(pseudorsquaredm2);
              print(llrfm2);
              print(llrf_pvaluem2);
              print(Desv_finalm2);
              print(Tfinalm2);
              print(P_valuem2);
              print(tcortem2);
              print(testm2);
              print(pvaluem2);
              print(tcortepspecialm2);
              print(testpspecialm2);
              print(pvaluepspecialm2);

              time_y2 = time() - start_time;
              start_time = time();

              fich=open("resultado.txt","a");
              fich.write(Vars[7+l] + "," + str(rho) + "," + str(iteracion) + "," + str(Btt1[0,0]) + "," + str(Btt1[1,0]) + "," + str(Btt1[2,0]) + "," + str(Btt1[3,0]) + ",");
              fich.write(str(Btt2[0,0]) + "," + str(Btt2[1,0]) + "," + str(Btt2[2,0]) + "," + str(Btt2[3,0]) + "," + str(error_predicciont1[0,0]) + ",");
              fich.write(str(error_predicciont2[0,0]) + "," + str(Rsquare1) + "," + str(Rsquare2) + "," + str(lognull) + "," + str(logmodelocompleto) + "," + str(logmwSNP) + ",");
              fich.write(str(pseudorsquared) + "," + str(llrf) + "," + str(llrf_pvalue) + "," +  str(Desv_b1[0,0]) + "," + str(Desv_b1[1,0]) + "," + str(Desv_b1[2,0]) + ",");
              fich.write(str(Desv_b1[3,0]) +  "," + str(Desv_b2[0,0]) + "," + str(Desv_b2[1,0]) + "," + str(Desv_b2[2,0]) + ",");
              fich.write(str(+ Desv_b2[3,0]) +  "," + str(Desv_rho[0]) + ",");
              fich.write(str(+ Tfinal1[0,0]) + "," + str(Tfinal1[1,0]) + "," + str(Tfinal1[2,0]) + "," + str(Tfinal1[3,0]) +  "," + str(Tfinal2[0,0]) + "," + str(Tfinal2[1,0]) + "," + str(Tfinal2[2,0]) + "," + str(Tfinal2[3,0]) + ",");
              fich.write(str(P_value1[0,0]) + "," + str(P_value1[1,0]) + "," + str(P_value1[2,0]) + "," + str(P_value1[3,0]) + "," + str(P_value2[0,0]) + "," + str(P_value2[1,0]) + "," + str(P_value2[2,0]) + "," + str(P_value2[3,0]) + ",");
              fich.write(str(tcortemb) + "," + str(testmb) + "," + str(pvaluemb) + "," +  str(tcortepspecial) + "," + str(testpspecial) + "," + str(pvaluepspecial) + ",");
              fich.write(str(logverotppy1) + "," + str(tcortepspecialy1) + "," + str(testpspecialy1) + "," + str(pvaluepspecialy1) + "," + str(logverotppy2) + "," + str(tcortepspecialy2) + "," + str(testpspecialy2) + "," + str(pvaluepspecialy2) + ",");              
              fich.write(str(iteracionesm1) + "," + str(Bponderadorm1[0,0]) + "," + str(Bponderadorm1[1,0]) + "," + str(Bponderadorm1[2,0]) + "," + str(Bponderadorm1[3,0]) + "," + str(error_prediccionm1[0,0]) + ",");
              fich.write(str(Rsquarem1) + "," + str(lognullm1) + "," + str(logmreducpm1) + ","  + str(logverofm1) + "," + str(pseudorsquaredm1) + "," + str(llrfm1) + "," + str(llrf_pvaluem1) + ",");
              fich.write(str(Desv_finalm1[0,0]) + "," + str(Desv_finalm1[1,0]) + "," + str(Desv_finalm1[2,0]) + "," + str(Desv_finalm1[3,0]) +  ","  + str(Tfinalm1[0,0]) + "," + str(Tfinalm1[1,0]) + ",");
              fich.write(str(Tfinalm1[2,0]) + "," + str(Tfinalm1[3,0]) + "," + str(P_valuem1[0,0]) + "," + str(P_valuem1[1,0]) + "," + str(P_valuem1[2,0]) + "," + str(P_valuem1[3,0]) + ",");
              fich.write(str(tcortem1) + "," + str(testm1) + "," + str(pvaluem1) + "," +  str(tcortepspecialm1) + "," + str(testpspecialm1) + "," + str(pvaluepspecialm1) + ",");
              fich.write(str(iteracionesm2) + "," + str(Bponderadorm2[0,0]) + "," + str(Bponderadorm2[1,0]) + "," + str(Bponderadorm2[2,0]) + "," + str(Bponderadorm2[3,0]) + "," + str(error_prediccionm2[0,0]) + ",");
              fich.write(str(Rsquarem2) + "," + str(lognullm2) + "," + str(logmreducpm2) + "," + str(logverofm2) + "," + str(pseudorsquaredm2) + "," + str(llrfm2) + "," + str(llrf_pvaluem2) + ",");
              fich.write(str(Desv_finalm2[0,0]) + "," + str(Desv_finalm2[1,0]) + "," + str(Desv_finalm2[2,0]) + "," + str(Desv_finalm2[3,0]) +  ","  + str(Tfinalm2[0,0]) + "," + str(Tfinalm2[1,0]) + ",");
              fich.write(str(Tfinalm2[2,0]) + "," + str(Tfinalm2[3,0]) + "," + str(P_valuem2[0,0]) + "," + str(P_valuem2[1,0]) + "," + str(P_valuem2[2,0]) + "," + str(P_valuem2[3,0]) + ",");
              fich.write(str(tcortem2) + "," + str(testm2) + "," + str(pvaluem2) + "," +  str(tcortepspecialm2) + "," + str(testpspecialm2) + "," + str(pvaluepspecialm2) + "\n");                         
              fich.close();

              time_write = time() - start_time;
              
              fich=open("time.txt","a");
              fich.write(Vars[7+l] + "," + str(time_lecture) + "," + str(time_snp) + "," + str(time_y1) + "," + str(time_y2) + "," + str(time_write) + "\n");
              fich.close();

