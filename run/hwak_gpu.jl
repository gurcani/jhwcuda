using jhwcuda
using DifferentialEquations
t0=0.0
t1=300.0
hw=hwak(1024,1024,C=1.0,ν=1e-3,D=1e-3,t1=300.0,wecontinue=false)
tspan=(hw.t0,hw.t1)
cbshow = PeriodicCallback(fshowcb_default,1.0)
fsave = [(r)->fsavecb_default(r,"fields"),]
dtsave = [1.0,]
cbls=vcat([cbshow],[PeriodicCallback(fsave[l],dtsave[l]) for l in range(1,length(fsave))])
cbs=CallbackSet(cbls...)
prob=ODEProblem(rhs!,hw.zk,tspan,hw,callback=cbs,
                save_on=false,save_everystep=false,save_start=false,save_end=false,
                abstol=1e-10,reltol=1e-9
                )
jhwcuda.ct=time_ns()/1e9
sol=solve(prob,Tsit5())
