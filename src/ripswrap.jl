mutable struct pers_diagram{precT<:AbstractPrecision}
    bars::Vector{Vector{Tuple{Float64,Float64}}}
    prec::Union{precT, Nothing}
    errbars::Union{Nothing, Vector{Vector{NTuple{2, NTuple{2,Float64}}} }}
    dimscomplete::Int
end

function print_rips(io::IO, ed::SparseEdges)
    pos_next = 1
    for i=1:ed.numpoints
        #pos_next > length(ed.edges) && break
        while pos_next <= length(ed.edges)
            u,v,d = ed.edges[pos_next]
            u != i && break
            pos_next += 1
            print(io, v-1, " ", d, " ")
        end
        i<ed.numpoints && print(io, "\n")
    end
    nothing
end

function readpers(f::IO)
       dimstart = r"^persistence intervals in dim (\d+):$"
       perspair = r"\[(\d+(?:\.\d+)?),((?:\d+(?:\.\d+)?)| )\)"
       donedim = r"^done dim (\d+)$"
       perspairs = Vector{Vector{Tuple{Float64, Float64}}}()
       done=true
       nprint = 10
       for line in readlines(f) 
            m=match(dimstart, line)
            if m!= nothing 
                dim = parse(Int, m.captures[1])
                dim == length(perspairs)|| error("parse error: Skipped dim?")
                push!(perspairs, Vector{Tuple{Float64, Float64}}())
                done=false
                continue
            end
            m=match(donedim, line)
            if m!= nothing 
                dim = parse(Int, m.captures[1])
                ( (dim +1 == length(perspairs) ) && (done==false ) ) || error("parse error: nested dim?")
                done = true
                continue
            end
            m=match(perspair,line)
            if m!=nothing
                (done==false) || error("parse error: unexpected persistence pair")
                birth = parse(Float64, m.captures[1])
                death = (m.captures[2]==" " ? Inf : parse(Float64, m.captures[2]))
                push!(perspairs[length(perspairs)], (birth, death))
            end
            #ignore other lines
        end
        #@show done, perspairs
        return (done, perspairs)
    end

function apply_prec(prec, perspairs)
    errbars = Vector{Vector{NTuple{2, NTuple{2,Float64}}}}(undef, length(perspairs))
    Rmax = Float64(prec.Rmax)
    for i=1:length(perspairs)
        pp = perspairs[i]
        pp_e = Vector{NTuple{2, NTuple{2,Float64}}}(undef, length(pp))
        errbars[i] = pp_e
        
        for (j,(b,d)) in enumerate(pp)
            if d == Inf
                pp_e[j] = ( (0.0, 0.0) , (Inf,Inf))
                continue
            else
                _b = min(Rmax,eval_down(prec, b))
                _d =  min(Rmax,eval_down(prec, d))
                b =  min(Rmax,b)
                d = min(Rmax, d)
                pp_e[j] = ((_b,b) , (_d,d))
            end
        end

    end
    errbars
end

function runrips(ed::SparseEdges; dim=1, modulus=2)
    if modulus == 2
        f = open(`$(RIPSER_BIN) --format lower-sparse --dim $(dim)`, "r+")
    else
        f = open(`$(RIPSER_COEFF_BIN) --format lower-sparse --dim $(dim) --modulus $(modulus)`, "r+")
    end
    print_rips(f, ed)
    close(f.in)
    done,perspairs = readpers(f.out)
    close(f)
    done_dims = length(perspairs) - ifelse(done, 0, 1)
    errbars = apply_prec(ed.prec, perspairs)
    return pers_diagram(perspairs, ed.prec,errbars, done_dims)
end
