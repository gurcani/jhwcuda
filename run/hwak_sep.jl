using jhwcuda
using DifferentialEquations
using SparseArrays,LinearAlgebra

t0=0.0
t1=300.0
hw=hwak(1024,1024,C=1.0,ν=1e-3,D=1e-3,t1=300.0,wecontinue=false)
tspan=(hw.t0,hw.t1)
cbshow = PeriodicCallback(fshowcb_default,1.0)
fsave = [(r)->fsavecb_default(r,"fields"),]
dtsave = [1.0,]
cbls=vcat([cbshow],[PeriodicCallback(fsave[l],dtsave[l]) for l in range(1,length(fsave))])
cbs=CallbackSet(cbls...)

jachw=init_spjac(hw)

function fimp!(du,u,p,t)
    du.=jachw*u
end

function fjac!(J,u,p,t)
    J.=jachw
end

fn=SplitFunction(fimp!,rhs_exp!,jac=fjac!,jac_prototype=jachw)
prob=SplitODEProblem(fn,hw.zk,tspan,hw,callback=cbs,
                     save_on=false,save_everystep=false,save_start=false,save_end=false,
                     abstol=1e-10,reltol=1e-9)

sv=KenCarp47(autodiff=false,concrete_jac=true,nlsolve=NLFunctional())
jhwcuda.ct=time_ns()/1e9
sol=solve(prob,sv)
