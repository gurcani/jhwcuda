using HDF5

struct h5flstr
    fl::HDF5.File
end

function save_data(fl::HDF5.File,group::String;ext_flag=false,kwargs...)
    if !(group in keys(fl))
        create_group(fl,group)
    end
    grp=fl[group]
    for (symk,val) in kwargs
        if(ext_flag==false)
            if (String(symk) in keys(grp))
                delete_object(grp,String(symk))
            end
            grp[String(symk)]=val
        else
            if !(String(symk) in keys(grp))
                sz=size(val)
                d = create_dataset(grp, String(symk),eltype(val), ((sz...,1), (sz...,-1)), chunk=(sz...,1))
            else
                d=grp[String(symk)]
                sz=size(d)
                HDF5.set_extent_dims(d,(sz[1:end-1]...,sz[end]+1))
            end
            d[:,:,end]=val
            flush(d)
        end
    end
    flush(grp)
end
        
