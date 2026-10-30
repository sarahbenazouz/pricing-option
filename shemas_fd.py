import numpy as np, time
from scipy.linalg import solve_banded
from bs import S0,K,T,r,sigma,bs_analytic
def coeffs(S,dt,dS):
    al=0.5*sigma**2*S**2*dt/dS**2-0.5*r*S*dt/dS
    ga=0.5*sigma**2*S**2*dt/dS**2+0.5*r*S*dt/dS
    return al,ga
def solve(scheme,N,M,Smax=3*K,full=False):
    S=np.linspace(0,Smax,N+1); dS=Smax/N; dt=T/M
    U=np.maximum(S-K,0.); Si=S[1:N]; al,ga=coeffs(Si,dt,dS)
    be=1+sigma**2*Si**2*dt/dS**2+r*dt
    ab=np.zeros((3,N-1))
    th={'imp':1.0,'cn':0.5}.get(scheme,0)
    if th>0:
        ab[0,1:]=-th*ga[:-1]; ab[1]=1+th*(be-1); ab[2,:-1]=-th*al[1:]
    for n in range(M):
        tn=n*dt; tp=(n+1)*dt
        UNc=Smax-K*np.exp(-r*tn); UNp=Smax-K*np.exp(-r*tp)
        Ui=U[1:N]
        if scheme=='exp':
            new=al*U[0:N-1]+(1-(be-1))*Ui+ga*U[2:N+1]
            U=np.concatenate([[0],new,[UNp]]); 
        else:
            w=1-th
            rhs=Ui+w*(al*U[0:N-1]-(be-1)*Ui+ga*U[2:N+1])
            rhs[-1]+=th*ga[-1]*UNp
            x=solve_banded((1,1),ab,rhs)
            U=np.concatenate([[0],x,[UNp]])
    return (S,U) if full else float(np.interp(S0,S,U))
ex=float(bs_analytic(S0,K,T,r,sigma))
if __name__=="__main__":  # démo rapide
    print("exact",ex)
    for sc,M in [('exp',10000),('imp',1000),('cn',500)]:
        t=time.time(); p=solve(sc,200,M); print(sc,M,p,abs(p-ex),time.time()-t)
    print("--- table (N,M)")
    for N,M in [(50,2000),(100,2000),(200,2000),(400,8000),(800,32000)]:
        print(N,M,[ (round(solve(s,N,M),6), "%.2e"%abs(solve(s,N,M)-ex)) for s in ['exp','imp','cn']])
    print("--- time conv (N=1600)")
    prev={}
    for M in [10,20,40,80,160,320]:
        for s in ['imp','cn']:
            e=abs(solve(s,1600,M)-ex); print(M,s,"%.3e"%e, (np.log2(prev[s]/e) if s in prev else None)); prev[s]=e
    print("--- space conv (M=20000 imp/cn)")
    prev={}
    for N in [25,50,100,200,400,800]:
        for s in ['cn']:
            e=abs(solve(s,N,4000)-ex); print(N,s,"%.3e"%e,(np.log2(prev[s]/e) if s in prev else None)); prev[s]=e
