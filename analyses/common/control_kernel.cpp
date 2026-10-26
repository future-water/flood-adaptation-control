#include <Accelerate/Accelerate.h>
#include <vector>
#include <cmath>
#include <algorithm>

// Exact gated recurrence of the pipeline (simulate_cl_flood_only), batched over LQR penalty values. No fast-math.
// U, force, states and each A/K are column-major. Statistics: channel,n,policy.
extern "C" int simulate_batch(int n, int r, int nt, int m, int nw, int ny,
    const double *basis, const double *mu, const double *thr, const double *z0,
    const double *aa, const double *kk, const double *force, const int *windows,
    const int *years, double *stats, double *amax,
    double *ztrace, double *utrace) {
    const size_t nm = (size_t)n*m, rm = (size_t)r*m;
    std::vector<double> z(rm), next(rm), x(nm), ze(rm), u(rm), up(nm);
    std::fill(stats, stats+9*nm, 0.);
    if (amax) std::fill(amax, amax+nm*ny, 0.);
    for (int j=0;j<m;j++) for (int i=0;i<r;i++) z[i+r*j]=z0[i];
    if (ztrace) std::copy(z.begin(),z.end(),ztrace);
    for (int t=0;t<nt;t++) {
        const int w=windows[t], yr=years[t];
        if (w<0 || w>=nw || yr<0 || yr>=ny) return -2;
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n,m,r,1.,basis,n,z.data(),r,0.,x.data(),n);
        for (int j=0;j<m;j++) for (int i=0;i<n;i++) {
            const size_t k=i+(size_t)n*j;
            const double raw=x[k]+mu[i], physical=std::max(raw,0.);
            if (!std::isfinite(raw)) return t+1;
            const double ex=std::max(raw-thr[i],0.);
            x[k]=ex;
            stats[3*nm+k]+=ex;
            stats[4*nm+k]+=(ex>0.);
            stats[5*nm+k]=std::max(stats[5*nm+k],physical);
            stats[6*nm+k]+=(raw<0.);
            if (amax) amax[(size_t)yr*nm+k]=std::max(amax[(size_t)yr*nm+k],physical);
        }
        cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,r,m,n,1.,basis,n,x.data(),n,0.,ze.data(),r);
        for (int j=0;j<m;j++) {
            const double *gain=kk+((size_t)w*m+j)*r*r;
            cblas_dgemv(CblasColMajor,CblasNoTrans,r,r,-1.,gain,r,ze.data()+r*j,1,0.,u.data()+r*j,1);
        }
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n,m,r,1.,basis,n,u.data(),r,0.,up.data(),n);
        for (int j=0;j<m;j++) for (int i=0;i<n;i++) {
            const size_t k=i+(size_t)n*j;
            const double control=up[k];
            if (!std::isfinite(control)) return t+1;
            const double atten=std::max(-control,0.), augment=std::max(control,0.);
            stats[k]+=std::abs(control);
            stats[nm+k]+=atten;
            stats[2*nm+k]+=augment;
            if (x[k]>0.) {
                stats[7*nm+k]+=atten;
                stats[8*nm+k]+=augment;
            }
        }
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,r,m,r,1.,aa+(size_t)w*r*r,r,z.data(),r,0.,next.data(),r);
        for (int j=0;j<m;j++) for (int i=0;i<r;i++) {
            const size_t k=i+(size_t)r*j;
            next[k]+=u[k]+force[i+(size_t)r*t];
        }
        z.swap(next);
        if (ztrace) std::copy(z.begin(),z.end(),ztrace+(size_t)(t+1)*rm);
        if (utrace) std::copy(u.begin(),u.end(),utrace+(size_t)t*rm);
    }
    return 0;
}
