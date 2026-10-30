import numpy as np, time
from bs import S0,K,T,r,sigma,bs_analytic
def put_amer(N,M,th=0.5,Smax=3*K,omega=1.3,tol=1e-10,american=True):
    S=np.linspace(0,Smax,N+1); dS=Smax/N; dt=T/M
    g=np.maximum(K-S,0.); U=g.copy(); Si=S[1:N]
    al=0.5*sigma**2*Si**2*dt/dS**2-0.5*r*Si*dt/dS
    ga=0.5*sigma**2*Si**2*dt/dS**2+0.5*r*Si*dt/dS
    be=1+sigma**2*Si**2*dt/dS**2+r*dt
    lo=-th*al; di=1+th*(be-1); up=-th*ga
    Sstar=np.zeros(M+1); Sstar[0]=K; its=0
    for n in range(M):
        tp=(n+1)*dt
        left_n=K*np.exp(-r*n*dt) if not american else K  # S=0 BC
        left_p=K*np.exp(-r*tp) if not american else K
        Ui=U[1:N]; w=1-th
        rhs=Ui+w*(al*U[0:N-1]-(be-1)*Ui+ga*U[2:N+1])
        rhs[0]-=lo[0]*left_p; # right BC =0
        x=np.maximum(Ui,g[1:N]) if american else Ui.copy()
        for it in range(10000):
            err=0.
            for j in range(N-1):
                xl=x[j-1] if j>0 else left_p
                xr=x[j+1] if j<N-2 else 0.
                y=(rhs[j]-lo[j]*xl-up[j]*xr)/di[j]
                v=x[j]+omega*(y-x[j])
                if american and v<g[j+1]: v=g[j+1]
                err+=(v-x[j])**2; x[j]=v
            its+=1
            if err<tol**2: break
        U=np.concatenate([[left_p],x,[0.]])
        if american:
            ex=np.where((U[1:N]-g[1:N]<1e-8)&(S[1:N]<K))[0]
            Sstar[n+1]=S[1+ex.max()] if len(ex) else np.nan
    return S,U,Sstar,its
def crr(n,american=True):
    dt=T/n;u=np.exp(sigma*np.sqrt(dt));d=1/u;p=(np.exp(r*dt)-d)/(u-d)
    S=S0*u**np.arange(n,-n-1,-2); V=np.maximum(K-S,0)
    for i in range(n-1,-1,-1):
        S=S0*u**np.arange(i,-i-1,-2); V=np.exp(-r*dt)*(p*V[:-1]+(1-p)*V[1:])
        if american: V=np.maximum(V,K-S)
    return V[0]
if __name__=="__main__":
    ref=crr(4000); print("CRR am",ref,"CRR eu",crr(4000,False),"BS put",bs_analytic(S0,K,T,r,sigma,'put'))
    for N,M in [(100,100),(200,200),(400,400)]:
        t=time.time();S,U,Ss,its=put_amer(N,M);p=np.interp(S0,S,U);print(N,M,p,abs(p-ref),its,time.time()-t,Ss[-1])
