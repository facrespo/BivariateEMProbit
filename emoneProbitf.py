''' emProbit(Y, X, maxiter=100, error=0.01, perturbation=0.001, alpha=0.05, p_special_test=1):
Parameters Y an Bivariate Vector with 0 or 1 with the same raws with X
X is the explicative variables, the constant is colocated in the Execution Problems.
maxiter is the maximal iteration number for default is 100.
Error is the error of tolerance to stop the look for the loglikehood in the EM Algorithm. For default is  0.01.
perturbation is a variable to perturb the Matrix X^tX if you like invert its. The default value is 0.001.
alpha is the value to look for the cut point of each test, for default is 0.05.
p_special_test is the variable p of the matrix X that we like to get the multivariate p-value, for default is 1.

Results:

iteracion is the number of iteration that Em Algorithm use to look for the optimal.
Bt is the ponderators of X.
vt is the predicción to Y, if vt[i]>0 then Y[i] estimate is 1. 
error_predicciont the cuadratic difference of error of Y respect the Y estimate, 
Rsquare the correlation of Y estimate with Y.
lognull is the loglikehood only the constant or last variable.
logmreducp is the loglikehood without the variable p_special_test. 
logverot is the loglikehood using the complete X.
scoret is the variation of ponderators, this function appear in (Green, 2008).
pseudorsquared is the value calculed as proportion of loglikehood (Green, 2008).  
llrf is the test of -2 diference of likelihood complete model less model with constant only
llrf_pvalue is the p-value of llrf
marginalt is the marginal value that appear in (Green, 2008).
Desv_final is the standard desviation of each ponderator.
Tfinal is the value ponderator/standard desviation for each ponderator
P_value is the p-value of each test using the t-student distribution as linear standard model,  
tcorte1 is the cut point of multilinneal test to the model without constant,
testm1 is the test value of multilinneal test to the model without constant
pvaluem1 is the p-value of testm1
tcortespecial is the cut point of multilinneal test to the model without p_special_test variable.
testspecial is the test value of multilinneal test to the model without p_special_test variable.
pvalpspecial is the p-value of testspecial value.


Greene, W. (1998). Econometric Analysis, Third edition. Prentice-Hall,
New Jersey.

'''

import numpy as np;
import numpy.linalg as la;
import scipy.stats;
from scipy import stats, special, optimize;
from scipy.stats import norm, chisqprob;
import scipy.linalg as scl;
#import sympy as sy;
import math;
#import statsmodels.api as sm;

FLOAT_EPS = np.finfo(float).eps;

def calculo_B0(X, Z, p2, perturbation):
    B0=np.zeros((p2,1));
    B0=np.dot((scl.pinv((perturbation*np.identity(p2))+np.dot(np.transpose(X),X))),np.dot(np.transpose(X),Z));
    return B0;

def calculo_B(X, v, Omega, p):
    H=np.dot(Omega,np.dot(np.transpose(X),v));
    H1=scl.pinv(0*np.identity(p)+np.dot(Omega,np.dot(np.dot(np.transpose(X),X),Omega)));
    D1=np.transpose(np.diagonal(H1));
    B=np.dot(Omega,np.dot(H1,H));
    return B, D1;


def calculo_margina1_v(x):
    v1=x+(stats.norm._pdf(x)/(stats.norm._sf((-1)*x)));
    v2=x-(stats.norm._pdf(x)/(stats.norm._cdf((-1)*x)));     
    return v1, v2;

c_m_v = np.vectorize(calculo_margina1_v); 

def calculo_v(Y, Xbeta, n):
    v1=np.zeros((n, 1));
    v2=np.zeros((n, 1));
    v=np.zeros((n, 1));
    v1, v2 = c_m_v(Xbeta);
    v=(np.multiply(Y,v1)+(np.multiply(np.ones((n, 1))-Y,v2)));    
    return v;

def convertir_v(v, n):
    v=np.asmatrix(v);
    Y=np.zeros((n, 1));
    Y[:,nonzero(v>0)[0]]=1;
    return Y;

def generar_qq(YY1, n, p):
    q1=np.zeros((n, 1));
    q1=((2*YY1)-np.ones((n, 1)));
    return q1;

def diagonal_beta(p,beta):
    I=np.identity(p);
    for i in range(0,p):
        I[i,i]=abs(beta[i]);
    return I;

def funcion_margina1_loglike(x, y):
    return y*(stats.norm.logcdf(x))+(1-y)*(stats.norm.logsf(x));

f_m_loglike = np.vectorize(funcion_margina1_loglike); 

def loglike(Y, Xbeta, n, p):
    return np.sum(f_m_loglike(Xbeta, Y));

def marginal_score(qq, x):
    return qq*stats.norm._cdf(qq*x)/np.clip(stats.norm._cdf(qq*x), FLOAT_EPS, 1 - FLOAT_EPS);
    
f_m_score = np.vectorize(marginal_score);

def score(Y, X, Xb, q, n, p):
        L=np.zeros((n, 1));
        L=f_m_score(q,Xb);
        return np.asmatrix(np.transpose(np.dot(np.transpose(L),X)));

def m_marginal(x, bb):
        return stats.norm._pdf(x)*np.transpose(bb);
        
f_m_marginal = np.vectorize(m_marginal);

def marginal(Xb, beta, n, p):
        suma=np.zeros((p,1));
        L=np.zeros((n,p));
        L=f_m_marginal(Xb,np.ones((n,1))*np.transpose(beta));
        suma=np.asmatrix(np.mean(L,0));
        return np.transpose(suma);

def tstudent(Z, n, p):
    return 2*(1.0 - scipy.stats.t(n-p-1).cdf(Z));  


def prsquared(lognull,logverot):
    return 1 - (logverot/lognull);


def llr(lognull,logverot):
    return -2*(lognull - logverot);

  
def llr_pvalue(X,llrf):
    df_model = float(la.matrix_rank(X) - 1);
    return stats.chisqprob(llrf, df_model);


def test_additional_predictors2(logver, logvernull, n, p, alpha, error):
    loglambbda=(logvernull)-(logver);
    r=p-1;
    q=p-2;
    m=1;
    estadistico=-2*loglambbda;
    gl=m*(r-q);
    tcorte=stats.chi2.ppf(1-(alpha/2), gl, loc=0, scale=1);
    pvalue2=stats.chi2.sf((estadistico), gl, loc=0, scale=1);
    return tcorte, estadistico, pvalue2;


def calculo_error(Y, Yt, n, p):
    YM=Y-Yt;
    suma=np.dot(YM.transpose(),YM);
    return suma;        

def marginal_desviaciones_betas(y, xbeta):
    pedazo1=(y*((stats.norm._pdf(xbeta)+xbeta*stats.norm._cdf(xbeta))/(stats.norm._cdf(xbeta)*stats.norm._cdf(xbeta))))+((1-y)*((stats.norm._pdf(xbeta)-xbeta*stats.norm._sf(xbeta))/(stats.norm._sf(xbeta)*stats.norm._sf(xbeta))));
    return np.asmatrix(stats.norm._pdf(xbeta))*pedazo1;        

f_marginal_desviaciones = np.vectorize(marginal_desviaciones_betas);

def desviaciones_betas(Y, X, Xb, n, p):
    suma=np.zeros((p, p));
    desviaciones=np.zeros((p,1));
    pedazo=f_marginal_desviaciones(Y, Xb);
    for i in range(0,n):
        MX=np.dot(np.transpose(X[i,None]),X[i,None]);
        pedazo3=pedazo[i,0]*MX;
        suma=suma+pedazo3;
      
    suma=scl.pinv(suma);
    std_error=np.asmatrix(np.sqrt(np.diag(np.abs(suma))));
    desviaciones=np.transpose(std_error);
    return desviaciones;        

def emProbitLittle(Ym, Xm, maxiter1, error1, perturbation1):
     
    n1=Xm.shape[0];
    p1=Xm.shape[1];
    Ym=np.matrix(Ym);
      
    Bt1=calculo_B0(Xm, Ym, p1, perturbation1);
    XBt1 = np.asmatrix(np.dot(Xm,Bt1));
    logverot1=loglike(Ym, XBt1, n1, p1);
    Omegat1=np.identity(p1);
    vti1=calculo_v(Ym, XBt1, n1);
    Btt1, Dtt1=calculo_B(Xm, vti1, Omegat1, p1);
    XBtt1 = np.asmatrix(np.dot(Xm,Btt1));
    logverott1=loglike(Ym, XBtt1, n1, p1);
    iteracion1=0;

    while ((np.linalg.norm(logverott1-logverot1)/np.linalg.norm(logverott1))> error1) and (iteracion1 <= (maxiter1-1)):
         logverot1=logverott1;
         Bt1=Btt1;
         Dt1=Dtt1;
         Omegat1=np.identity(p1);
         XBt1 = np.asmatrix(np.dot(Xm,Bt1));
         vti1=calculo_v(Ym, XBt1, n1);
         Btt1, Dtt1=calculo_B(Xm, vti1, Omegat1, p1);
         XBtt1 = np.asmatrix(np.dot(Xm,Btt1));
         logverott1=loglike(Ym, XBtt1, n1, p1);
         iteracion1=iteracion1+1;


    Bt1=Btt1;
    Dt1=Dtt1;
    vti1=calculo_v(Ym, XBtt1, n1);
    logverot1= logverott1;
    return iteracion1, Bt1, Dt1, vti1, XBtt1, logverot1;

 
def emProbit(Y, X, maxiter=100, error=0.01, perturbation=0.001, alpha=0.05, p_special_test=1):
    n=X.shape[0];
    p=X.shape[1];
    Y=np.matrix(Y);
    X=np.matrix(X);
    q1=generar_qq(Y, n, p);
    iteracion0, Bnull, D0t, v0t, XBnull, lognull=emProbitLittle(Y, np.ones((n,1)), maxiter, error, perturbation);
    iteracion, Bt, Dt, vt, XBt, logverot=emProbitLittle(Y, X, maxiter, error, perturbation);
    marginalt=marginal(XBt, Bt, n, p);
    scoret=score(Y, X, XBt, q1, n, p);
    error_predicciont=calculo_error(q1, vt, n, p);
    Desv_final=desviaciones_betas(Y, X, XBt, n, p);
    XN=np.zeros((n, 2));
    Tfinal=np.zeros((p, 1));
    P_value=np.zeros((p, 1));
         
    for i in range(0,p):
         if (abs(Desv_final[i])<= error):
             Tfinal[i]=Bt[i]*(1e100);
             P_value[i] = 0;
         else:
             Tfinal[i]=(Bt[i])/(Desv_final[i]);
             P_value[i] = tstudent(np.abs(Tfinal[i]), n, p);
  
    XN=np.asmatrix(np.concatenate((vt,q1),1).transpose());
    Corr=(np.corrcoef(XN));
    Rsquare=Corr[0,1];
    pseudorsquared=prsquared(lognull,logverot);
    llrf=llr(lognull,logverot);
    llrf_pvalue=llr_pvalue(X,llrf);
    if (p>1):
        Bconst=np.zeros((p-1, 1));
        Bnueva=np.zeros((p, 1));
        Bnueva=calculo_B0(X, vt, p, perturbation);
        iteracionconst, Bconst, Dconst, vconst, XBconst, logmreduc=emProbitLittle(Y, X[:,0:(p-1)], maxiter, error, perturbation);
        tcorte1, testm1, pvaluem1 = test_additional_predictors2(logverot, logmreduc, n, p, alpha, error);
        Btp=np.zeros((p, 1));
        Btp2=np.zeros((p-1, 1));
        Xesp=np.zeros((n, (p-1)));
        Xesp=np.asmatrix(Xesp);
        j=0;
        for i in range(0,p):
            if i==(p_special_test-1):
                i=i;
            else:
                aux=np.asmatrix(X[:,i]);
                if aux.shape[0]==1:
                    Xesp[:,j]=aux.transpose();
                else:
                    Xesp[:,j]=aux;
                j=j+1;
        iteracionepl, Btp2, Dreduct, vreduct, XBreduct,logmreducp=emProbitLittle(Y, Xesp, maxiter, error, perturbation);
        j=0;
        for i in range(0,p):
            if i==(p_special_test-1):
                Btp[i]=0;
            else:
                Btp[i]=Btp2[j];
                j=j+1;
        tcortespecial, testspecial, pvalpspecial = test_additional_predictors2(logverot, logmreducp, n, p, alpha, error);
    else:
        logmreduc=0;
        logmreducp=0;        
        tcorte1=0;
        testm1=0;
        pvaluem1=0;
        tcortespecial=0;
        testspecial=0;
        pvalpspecial=0;          
    return iteracion, Bt, vt, error_predicciont, Rsquare, lognull, logmreducp, logverot, scoret, pseudorsquared, llrf, llrf_pvalue, marginalt, Desv_final, Tfinal, P_value,  tcorte1, testm1, pvaluem1, tcortespecial, testspecial, pvalpspecial;
