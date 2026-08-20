"""Leave-one-mass-out: fit without the 1.906 g records, test transfer onto them.
This is the only honest way to choose a fitting method here, because the pure
tests must stay untouched and the failure mode is specifically mass transfer."""
import pickle, os, functools, numpy as np
from scipy.optimize import least_squares
import grey_box
print=functools.partial(print,flush=True)
C=pickle.load(open('cache.pkl','rb')); recs=C['records']; P=C['P']; dt=C['dt']
NS=len(P['spring_coefficients']); NB=len(P['bl_coefficients'])
v0=np.concatenate([P['spring_coefficients'],P['bl_coefficients'],[P['damping']]])
def unpack(v):
    s=v[:NS]
    return {'spring_coefficients':s,'spring_derivative':np.polyder(s),
            'damping':float(v[-1]),'bl_coefficients':v[NS:NS+NB],
            'position_low_mm':P['position_low_mm'],'position_high_mm':P['position_high_mm']}
LIGHT=[r for r in recs if abs(r['total_mass_g']-3.333)<0.01]
FITSET=[r for r in recs if not r['is_pure_test'] and abs(r['total_mass_g']-3.333)>=0.01]
print('fit on',len(FITSET),'records; transfer test on',len(LIGHT),'light-mass records')
MAXN=int(os.environ.get('SAMP','3000'))
bundle=[]
for r in FITSET:
    pad=r['pad']; meas=r['outputs'][pad:,0].astype(np.float64)
    if len(meas)>MAXN:
        r=dict(r); keep=pad+MAXN
        r['current']=r['current'][:keep]; r['outputs']=r['outputs'][:keep]; r['samples']=keep
        meas=meas[:MAXN]
    bundle.append((r,meas,float(np.std(meas))+1e-9))
def resid(v,lam):
    t=unpack(v); out=[]
    for r,meas,sc in bundle:
        out.append((grey_box.simulate_fast(r,t,dt)[r['pad']:]-meas)/sc/np.sqrt(len(meas)))
    if lam>0:
        sc0=np.where(np.abs(v0)<1e-12,1.0,np.abs(v0))
        out.append(lam*(v-v0)/sc0)
    return np.concatenate(out)
def transfer(params):
    e=[]
    for r in LIGHT:
        pad=r['pad']; m=r['outputs'][pad:,0]
        e.append(float(np.sqrt(np.mean((m-grey_box.simulate_fast(r,params,dt)[pad:])**2))))
    return float(np.mean(e))
print(f'{"method":28s} {"fitRMSEcost":>12s} {"transferRMSE(mm)":>18s}')
print(f'{"equation error (current)":28s} {float(np.sum(resid(v0,0)**2)):12.5f} {transfer(unpack(v0)):18.4f}')
for lam in [0.0,0.05,0.2,0.5]:
    s=least_squares(lambda v: resid(v,lam), v0,
                    x_scale=np.where(np.abs(v0)<1e-12,1.0,np.abs(v0)),
                    diff_step=1e-4, max_nfev=int(os.environ.get('NFEV','25')))
    print(f'{"refined lambda="+str(lam):28s} {float(np.sum(resid(s.x,0)**2)):12.5f} {transfer(unpack(s.x)):18.4f}')
