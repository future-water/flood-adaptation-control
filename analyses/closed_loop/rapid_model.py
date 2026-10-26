"""Fixed-parameter RAPID routing: network order and Muskingum coefficients for the matched_chunk kernel."""
import numpy as np
from collections import deque

ip=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
dp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')

class RAPID:
    def __init__(self,comid,downstream,K_seconds,x,dt=900):
        self.n=len(comid); self.dt=dt
        assert 86400%dt==0
        self.steps=int(86400/dt)
        pos={int(c):i for i,c in enumerate(comid)}
        unknown=[int(c) for c in downstream if int(c)>0 and int(c) not in pos]
        if unknown:raise ValueError(f'Unknown downstream IDs {unknown[:5]}')
        self.down=np.ascontiguousarray([pos.get(int(c),-1) for c in downstream],dtype=np.int64)
        degrees=np.bincount(self.down[self.down>=0],minlength=self.n)
        ready=deque(np.where(degrees==0)[0]);order=[]
        while ready:
            i=ready.popleft();order.append(i);j=self.down[i]
            if j>=0:
                degrees[j]-=1
                if degrees[j]==0:ready.append(j)
        if len(order)!=self.n:raise ValueError('Network has a cycle')
        self.order=np.ascontiguousarray(order,dtype=np.int64)
        K=np.asarray(K_seconds,dtype=float);x=np.asarray(x,dtype=float)
        assert np.all(K>0) and np.all((x>=0)&(x<=.5))
        denom=dt/2+K*(1-x)
        self.c1=np.ascontiguousarray((dt/2-K*x)/denom)
        self.c2=np.ascontiguousarray((dt/2+K*x)/denom)
        self.c3=np.ascontiguousarray((K*(1-x)-dt/2)/denom)
        assert np.max(abs(self.c1+self.c2+self.c3-1))<1e-12
