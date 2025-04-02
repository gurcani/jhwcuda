struct slicelist{T}
    Nx::Int64
    Ny::Int64
    Npx::Int64
    Npy::Int64
    N::Int64
    it::T
    itbar::CartesianIndices{2}
    itbarp::CartesianIndices{2}
    r::Vector{Int64}
    c::Vector{Int64}
end

function slicelist(Nx::Int64,Ny::Int64,Npx::Int64=Nx,Npy::Int64=Ny)
    sl=[CartesianIndices((2:Int(Ny/2),1:1)),
        CartesianIndices((1:Int(Ny/2),2:Int(Nx/2))),
        CartesianIndices((2:Int(Ny/2),Npx-Int(Nx/2)+2:Npx))]
    N=sum([length(l) for l in sl])
    it=Base.Iterators.flatten(sl)
    itbar=CartesianIndices((1:1,2:Int(Nx/2)))
    itbarp=CartesianIndices((1:1,Npx:-1:Npx-Int(Nx/2)+2))
    IJ′=reinterpret(Int, reshape(collect(it), 1, :))
    r,c=collect(view(IJ′, 1, :)), collect(view(IJ′, 2, :))
    return slicelist(Nx,Ny,Npx,Npy,N,it,itbar,itbarp,r,c)
end

Base.iterate(sl::slicelist)=Base.iterate(sl.it)
Base.iterate(sl::slicelist,state)=Base.iterate(sl.it,state)
Base.to_index(sl::slicelist) = collect(sl.it)

function init_kgrid(Nx::Int64,Ny::Int64,Lx::Float64,Ly::Float64)
    sl=slicelist(Nx,Ny,Nx,Ny)
    kxl,kyl=collect(Base.Iterators.flatten(((0:Int(Nx/2)-1), (-Int(Nx/2):-1)))), collect(0:Int(Ny/2))
    dkx,dky=2π/Lx,2π/Ly
    N=sl.N
    kx,ky=zeros(Float64,N),zeros(Float64,N)
    for (i,l) in zip(range(1,N),sl)
        lx=l.I[2]
        ly=l.I[1]
        kx[i]=kxl[lx]*dkx
        ky[i]=kyl[ly]*dky
    end
    return jhwar(kx),jhwar(ky)
end

# function init_kgrid(sl::slicelist,Lx::Float64,Ly::Float64)
#     kxl,kyl=collect(Base.Iterators.flatten(((0:Int(Nx/2)-1), (-Int(Nx/2):-1)))), collect(0:Int(Ny/2))
#     dkx,dky=2π/Lx,2π/Ly
#     kx,ky=zeros(Float64,N),zeros(Float64,N)
#     for (i,l) in zip(range(1,sl.N),sl)
#         lx=l.I[2]
#         ly=l.I[1]
#         kx[i]=kxl[lx]*dkx
#         ky[i]=kyl[ly]*dky
#     end
#     return kx,ky
# end

function irft(vk::AbstractArray,sl::slicelist)
    vkp=jhwar(zeros(ComplexF64,(Int(sl.Npy/2)+1,sl.Npx)))
    vkp[sl].=vk
    vkp[sl.itbarp].=vkp[sl.itbar]
    return brfft(vkp,sl.Npy)
end

function rft(vp::AbstractArray,sl::slicelist)
    vkp=rfft(vp)
    return vkp[sl]/(slp.Npx*slp.Npy)
end

# function SparseArrays.sparse(phik::Vector{ComplexF64},slp::slicelist,Npx::Int64,Npy::Int64)
#      phikp=sparse(slp.r,slp.c,phik,Int(Npy/2)+1,Npx)
#      phikp[1,Npx:-1:Npx-Int(Nx/2)+2]=conj(phikp[1,2:Int(Nx/2)])
# end

# function rft(planf::FFTW.rFFTWPlan,u::Matrix{Float64},sl::slicelist)
#     uk=hw.planf*u
#     return(uk[sl])
# end

# function SparseArrays.sparse(IJ::Vector{<:CartesianIndex}, v, m, n)
#     IJ′ = reinterpret(Int, reshape(IJ, 1, :))
#     return sparse(view(IJ′, 1, :), view(IJ′, 2, :), v, m, n)
# end

# struct wasp_grid{T}
#     N::Int64
#     index::Int64
#     dk::Float64
#     it::T
# end

# function wasp_grid(it,index::Int64,dk::Float64)
#     if(Base.IteratorSize(it)==Base.SizeUnknown())
#         N=sum([length(l) for l in it.it])
#     else
#         N=length(it)
#     end
#     return wasp_grid(N,index,dk,it)
# end

# Base.getindex(vin::wasp_grid,i::Int64 = vin.it[i.I[vin.index]]
# Base.getindex(vin::wasp_grid,i::CartesianIndex{2}) = vin[i.I[vin.index]]
