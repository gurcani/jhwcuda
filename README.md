# jhwcuda
julia version of the pseudo-spectral cuda Hasegawa-Wakatani solver.

Edit the file run/hwak_gpu.jl with the parameter values as you like.

Then you can run it either directly as:

```
cd run
julia --project=../ hwak_gpu.jl
```
or with multiple threads using the additional `-t N` option. 

I have also added a splitODE version, which can be run with:

```
julia --project=../ hwak_sep.jl
```

Also you can switch between CUDA and cpu array versions by commenting and uncommenting the following lines in jwhcuda.jl

```julia
### IF USING CUDA:

using CUDA
jhwar=CuArray
jhwplan=CUDA.CUFFT.CuFFTPlan
jhwspmat=CUSPARSE.CuSparseMatrixCSC

### IF USING CPU:

# jhwar=Array
# jhwplan=FFTW.rFFTWPlan
# jhwspmat=SparseMatrixCSC

```
