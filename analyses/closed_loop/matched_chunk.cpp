#include <vector>
#include <algorithm>
// Exact daily-coordinate recurrence R[k+1] = F^96 R[k] + u[k].
// Fixed Muskingum propagator of the RAPID network (coefficients from rapid_model.RAPID); carry state across chunks.
extern "C" void matched_chunk(int n,int days,int steps,const long long* order,
 const long long* downstream,const double* c1,const double* c2,const double* c3,
 const double* control,const double* initial,double* outputs,double* final){
 std::vector<double> q(initial,initial+n),next(n),up(n);
 for(int d=0;d<days;d++){
  for(int s=0;s<steps;s++){
   std::fill(up.begin(),up.end(),0.0);
   for(int i=0;i<n;i++)if(downstream[i]>=0)up[downstream[i]]+=q[i];
   for(int i=0;i<n;i++)next[i]=c2[i]*up[i]+c3[i]*q[i];
   for(int j=0;j<n;j++){
    int i=order[j],target=downstream[i];
    if(target>=0)next[target]+=c1[target]*next[i];
   }
   q.swap(next);
  }
  for(int i=0;i<n;i++){q[i]+=control[d*n+i];outputs[d*n+i]=q[i];}
 }
 std::copy(q.begin(),q.end(),final);
}
