module jhwcuda

export hwak,fshowcb_default,fsavecb_default,rhs!,rhs_exp!,rhs_imp!,init_spjac

using FFTW
using DifferentialEquations
using SparseArrays

### IF USING CUDA:

using CUDA
jhwar=CuArray
jhwplan=CUDA.CUFFT.CuFFTPlan
jhwspmat=CUSPARSE.CuSparseMatrixCSC

### IF USING CPU:

# jhwar=Array
# jhwplan=FFTW.rFFTWPlan
# jhwspmat=SparseMatrixCSC

using HDF5
include("mlsarray.jl")
include("h5tools.jl")
ct=time_ns()/1e9

struct params
    C::Float64
    κ::Float64
    ν::Float64
    D::Float64
end

struct hwak
    Npx::Int64
    Npy::Int64
    Nx::Int64
    Ny::Int64
    sl::slicelist
    kx::jhwar
    ky::jhwar
    zk::jhwar
    planb::jhwplan
    planf::jhwplan
    fl::HDF5.File
    ps::params
    t0::Float64
    t1::Float64
end

function hwak(Npx=1024,Npy=1024;
              Lx=12*π,Ly=12*π,flname="out.h5",
              C=1.0,κ=1.0,ν=1e-3,D=1e-3,
              wecontinue=false,t0=0.0,t1=100.0
              )
    Nx,Ny=2*Int(floor(Npx/3)),2*Int(floor(Npy/3))
    sl=slicelist(Nx,Ny,Npx,Npy)
    kx,ky=init_kgrid(Nx,Ny,Lx,Ly)
    planb=plan_brfft(jhwar(zeros(ComplexF64,(Int(Npy/2)+1,Npx))),Npy,(1,2))
    planf=plan_rfft(jhwar(zeros(Float64,(Npy,Npx))),(1,2))
    ps=params(C,κ,ν,D)
    if(!wecontinue)
        fl=h5open(flname,"w";swmr=true)
        zk=init_fields(kx,ky)
    else
        fl=h5open(flname,"r+";swmr=true)
        zk=jhwar(fl["last/uk"][])
        t0=fl["last/t"][]
    end
    return hwak(Npx,Npy,Nx,Ny,sl,kx,ky,zk,planb,planf,fl,ps,t0,t1)
end

function init_fields(kx,ky,w=5.0,A=1e-4)
    N=length(kx)
    zk=jhwar(zeros(ComplexF64,2*N))
    phik=@view zk[1:N]
    nk=@view zk[N+1:end]
    phik[:]=A*exp.(-kx.^2/2/w^2-ky.^2/w^2).*exp.(1im*2π*jhwar(rand(N)))
    nk[:]=A*exp.(-kx.^2/2/w^2-ky.^2/w^2).*exp.(1im*2π*jhwar(rand(N)))
    return zk
end

function rhs!(dzkdt,zk,hw,t)
    p,sl,kx,ky=hw.ps,hw.sl,hw.kx,hw.ky
    C,κ,ν,D = p.C,p.κ,p.ν,p.D
    N=sl.N
    Φk=@view zk[1:N]
    nk=@view zk[N+1:end]
    dΦkdt=@view dzkdt[1:N]
    dnkdt=@view dzkdt[N+1:end]    
    ksqr=kx.^2+ky.^2
    sigk=(ky.>0)
    ∂xΦ=irfft(1im*kx.*Φk,hw)
    ∂yΦ=irfft(1im*ky.*Φk,hw)
    Ω=irfft(-ksqr.*Φk,hw)
    n=irfft(nk,hw)
    dΦkdt.=(-1im*kx.*rfft(∂yΦ.*Ω,hw) + 1im*ky.*rfft(∂xΦ.*Ω,hw) - C*(Φk - nk).*sigk)./ksqr - ν*sigk.*ksqr.*Φk
    dnkdt.=1im*kx.*rfft(∂yΦ.*n,hw) - 1im*ky.*rfft(∂xΦ.*n,hw) + sigk.*(C .- 1im*κ*ky).*Φk - (C .+ D*ksqr).*sigk.*nk
end

function rhs_exp!(dzkdt,zk,hw,t)
    p,sl,kx,ky=hw.ps,hw.sl,hw.kx,hw.ky
    C,κ,ν,D = p.C,p.κ,p.ν,p.D
    N=sl.N
    Φk=@view zk[1:N]
    nk=@view zk[N+1:end]
    dΦkdt=@view dzkdt[1:N]
    dnkdt=@view dzkdt[N+1:end]    
    ksqr=kx.^2+ky.^2
    ∂xΦ=irfft(1im*kx.*Φk,hw)
    ∂yΦ=irfft(1im*ky.*Φk,hw)
    Ω=irfft(-ksqr.*Φk,hw)
    n=irfft(nk,hw)
    dΦkdt.=(-1im*kx.*rfft(∂yΦ.*Ω,hw) + 1im*ky.*rfft(∂xΦ.*Ω,hw))./ksqr
    dnkdt.=1im*kx.*rfft(∂yΦ.*n,hw) - 1im*ky.*rfft(∂xΦ.*n,hw)
end

function rhs_imp!(dzkdt,zk,hw,t)
    p,sl,kx,ky=hw.ps,hw.sl,hw.kx,hw.ky
    C,κ,ν,D = p.C,p.κ,p.ν,p.D
    N=sl.N
    Φk=@view zk[1:N]
    nk=@view zk[N+1:end]
    dΦkdt=@view dzkdt[1:N]
    dnkdt=@view dzkdt[N+1:end]    
    ksqr=kx.^2+ky.^2
    sigk=(ky.>0)
    dΦkdt.=-C*(Φk - nk).*sigk./ksqr - ν*sigk.*ksqr.*Φk
    dnkdt.=sigk.*(C .- 1im*κ*ky).*Φk - (C .+ D*ksqr).*sigk.*nk
end

function init_spjac(hw)
    p,kx,ky=hw.ps,hw.kx,hw.ky
    C,κ,ν,D = p.C,p.κ,p.ν,p.D
    N=hw.sl.N
    ksqr=Array(kx.^2+ky.^2)
    sigk=Array((ky.>0))
    ky=Array(ky)
    J = spzeros(ComplexF64,2*N, 2*N)
    for j in 1:N
       ϕj=j
       nj=j+N
       J[ϕj,ϕj] = -C*sigk[j]/ksqr[j] - ν*sigk[j]*ksqr[j]
       J[ϕj,nj] = C*sigk[j]/ksqr[j]
       J[nj,ϕj] = sigk[j]*(C-1im*κ*ky[j])
       J[nj,nj] = -(C+D*ksqr[j])*sigk[j]
    end
    return jhwspmat(J)
end

function fshowcb_default(r)
    u,t=r.u,r.t
    println("t=",t," , ",time_ns()/1e9-ct," secs elapsed, u^2 =",sum(u.*conj(u)))
end

function fsavecb_default(r,flag)
    zk=r.u
    t=r.t
    hw=r.p
    kx,ky=hw.kx,hw.ky
    N=hw.sl.N
    Φk=@view zk[1:N]
    nk=@view zk[N+1:end]
    if(flag=="fields")
        Ω=irfft_unpad(-(kx.^2+ky.^2).*Φk,hw)
        n=irfft_unpad(nk,hw)
        save_data(hw.fl,flag;ext_flag=true,om=Ω,n=n,t=t)
    end
    save_data(hw.fl,"last",ext_flag=false;t=t,uk=Array(zk))
end

function irfft(vk::AbstractArray,hw::hwak,sl::slicelist = hw.sl)
    planb=hw.planb
    vkp=jhwar(zeros(ComplexF64,(Int(sl.Npy/2)+1,sl.Npx)))
    vkp[sl].=vk
    vkp[sl.itbarp].=conj(vkp[sl.itbar])
    return planb*vkp
end

function irfft_unpad(vk::AbstractArray,hw::hwak)
    sl=slicelist(hw.Nx,hw.Ny)
    vkp=jhwar(zeros(ComplexF64,(Int(sl.Ny/2)+1,sl.Nx)))
    vkp[sl].=vk
    vkp[sl.itbarp].=vkp[sl.itbar]
    return brfft(vkp,sl.Ny)
end

function rfft(vp::AbstractArray,hw::hwak,sl::slicelist = hw.sl)
    planf=hw.planf
    return (planf*vp)[sl]/(sl.Npx*sl.Npy)
end

end # module jhwcuda
