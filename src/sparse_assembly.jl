

mutable struct SparseEdges{dist_T, point_T, f_T, precT}
    ct::Ctree{dist_T, point_T, f_T}    
    numpoints::Int64
    edges::Vector{Tuple{Int32,Int32, f_T}} 
    evals::Int64
    prec::precT
end

struct APriori end
struct APosteriori end
struct APosteriori2 end
struct SemiAPosteriori end



function sparsify_ctree(ct::Ctree{dist_T, point_T, f_T}, eps_rel::Real, Nmax=0; strategy=APosteriori(), ctimes = Vector{f_T}()) where {f_T,point_T,dist_T}
    eps_rel2 = f_T(eps_rel)
    prec = Linear_prec(eps_rel2)
    if strategy === APriori()
        return sparse_edges_apriori(ct, prec, Nmax)
    elseif strategy === APosteriori()
        return sparse_edges_aposteriori(ct, prec, Nmax; ctimes=ctimes)
    else
        throw("$(strategy) is not a supported strategy")
    end
end

function sparsify_ctree(ct::Ctree{dist_T, point_T, f_T}, prec::AbstractPrecision, Nmax=0; strategy=APosteriori())where {f_T,point_T,dist_T}
    if strategy === APriori()
        return sparse_edges_apriori(ct, prec, Nmax)
    elseif strategy === APosteriori()
        return sparse_edges_aposteriori(ct, prec, Nmax)
    else
        throw("$(strategy) is not a supported strategy")
    end
end


function sparse_edges_apriori(ct::Ctree{dist_T, point_T, f_T}, prec, Nmax) where {dist_T, point_T, f_T}
    if Nmax <= 0 || Nmax >= length(ct.data)
        Nmax = length(ct.data)
        eps_abs = f_T(0)
    else
        Nmax = Int64(Nmax)
        eps_abs = 2*ct.nodes[ct.ibo[Nmax+1]].r
    end
    Rcut = ct.nodes[ct.ibo[2]].r
    prec_res = Affine_cutoff(eps_abs, Rcut, prec)


    evals = 0
    edges = Vector{Tuple{Int32,Int32,f_T}}()

    nodes = ct.nodes
    ibo = ct.ibo
    data = ct.data
    dist = ct.dist

    for i=2:Nmax
        p = nodes[ibo[i]].parent
        d = evaluate(dist, data[p], data[ibo[i]])
        evals += 1
        push!(edges,(Int32(i), nodes[p].bo, d) )
    end
    posL = 1
    while posL < length(edges)
        vv,uu,d_uv = edges[posL] 
        @assert  0 < uu < vv <= Nmax
        posL += 1
        for (u,v) in ((uu,vv),(vv,uu))
            ui=ibo[u]
            vi = ibo[v]
            node_u = ct.nodes[ui]
            node_v = ct.nodes[vi]
            @assert u==node_u.bo
            @assert v==node_v.bo
            @assert v <= Nmax
            cp = node_v.cf 
            while cp > 0            
                node_vc = ct.nodes[cp]
                node_vc.bo <= node_u.bo && @goto contv
                node_vc.bo <= Nmax || break           
                t = eval_Q(prec_res, node_vc.r)             

                (t < d_uv) &&  @goto contv
                d = evaluate(dist, ct.data[cp], ct.data[ui])
                evals += 1
                (t < d) && @goto contv
                push!(edges, (node_vc.bo,u,d))

                @label contv
                cp = node_vc.sn
            end
        end
    end

    sort!(edges; alg=QuickSort)
    SparseEdges(ct, Nmax, edges, evals, prec_res)
end


function _dmin_compute(ct, mind_heap, mind, evals)
    (mind < 0) && return mind
    nodes = ct.nodes
    ibo = ct.ibo
    data = ct.data
    dist = ct.dist
    heapify!(mind_heap)
    clast = (mind, Int32(-1), Int32(-1))
    while length(mind_heap)>0 && ( mind_heap[1][1] < mind )# && (mind < until)
        cand =  heappop!(mind_heap)
        if cand == clast
            @show cand, clast
            @assert cand != clast
        end
        clast = cand
        d_est, meu, mev = cand 
        d_uv = evaluate(dist, ct.data[ibo[meu]], ct.data[ibo[mev]])
        @assert d_est <= d_uv
        evals[] += 1
        if d_uv < mind
            mind = d_uv
            #mind < until && break
        end
        #mind = min(mind, d_uv)
        for (u,v) in ((meu,mev),(mev,meu))
            ui = ibo[u]
            vi = ibo[v]
            node_u = ct.nodes[ui]
            node_v = ct.nodes[vi]

            @assert u==node_u.bo && v==node_v.bo 

            cp = node_v.cf 
            while cp > 0
                node_vc = ct.nodes[cp]
                if node_vc.bo > node_u.bo #&& (@goto loopend)  
                    de = max(d_est, d_uv - 2*node_vc.r)
                    if de < mind 
                        heappush!(mind_heap, (de, node_vc.bo, u))
                    end
                end
                cp = node_vc.sn
            end
        end#children expanded
    end #heap search done.
    empty!(mind_heap)
    return mind
end

function missing_edges!(dst, ct, edge, emin_keys)
    ibo = ct.ibo
    uu,vv,d_uv,_ = edge
   # edgeset = keys(dgedict)
    
   # @assert Nmax >= uu > vv >0
   # posL += 1
    for (u,v) in ((uu,vv),(vv,uu))
        ui=ibo[u]
        vi = ibo[v]
        node_u = ct.nodes[ui]
        node_v = ct.nodes[vi]
        cp = node_v.cf
        while cp > 0
            node_vc = ct.nodes[cp]
            if node_vc.bo > node_u.bo 
                if (node_vc.bo, u) in emin_keys #&& @goto contv
                else
                    # push!(missing_edges, (uu,vv,node_vc.bo, u, d_uv - 2 * node_vc.r))
                    push!(dst, ( d_uv - 2 * node_vc.r ,node_vc.bo, u))
                end
            end
            
            #@label contv
            cp = node_vc.sn
        end
    end


end

function sparse_edges_aposteriori(ct::Ctree{dist_T, point_T, f_T}, prec, Nmax=0; ctimes = Vector{f_T}()) where {dist_T, point_T, f_T}
    if Nmax <= 0 || Nmax >= length(ct.data)
        Nmax = length(ct.data)
        eps_abs = f_T(0)
    else
        Nmax = Int64(Nmax)
        eps_abs = 2*ct.nodes[ct.ibo[Nmax+1]].r
    end
    Rcut = ct.nodes[ct.ibo[2]].r
    prec_res = Affine_cutoff(eps_abs, Rcut, prec)

    evals = 0
    edges = Vector{Tuple{Int32,Int32,f_T, f_T}}() #v ,u, d, d_parent; v>0
    #missing_edges = Vector{Tuple{Int32, Int32, Int32, Int32, f_T}}() #uparent, vparent, u, v, d_estimate
    nodes = ct.nodes
    ibo = ct.ibo
    data = ct.data
    dist = ct.dist
    #
    # Step 1: Build apriori-sparsified tree, keep missing edges
    #
    for i=2:length(ct.nodes)
        p = nodes[ibo[i]].parent
        d = evaluate(dist, data[p], data[ibo[i]])
        evals += 1
        push!(edges,(Int32(i),nodes[p].bo, d, f_T(0)) )
    end
    posL = 1
    while posL < length(edges)
        uu,vv,d_uv,_ = edges[posL] 
        @assert uu > vv >0
        posL += 1
        for (u,v) in ((uu,vv),(vv,uu))
            ui=ibo[u]
            vi = ibo[v]
            node_u = ct.nodes[ui]
            node_v = ct.nodes[vi]
            cp = node_v.cf
            while cp > 0
                node_vc = ct.nodes[cp]
                if node_vc.bo > node_u.bo
                    #node_vc.bo <= node_u.bo && @goto contv        
                    d = evaluate(dist, ct.data[cp], ct.data[ui])
                    evals += 1
                    t = eval_Q(prec_res, node_vc.r)
                    if t < d_uv ||  t < d || node_vc.bo > Nmax
                        #push!(missing_edges, (uu,vv,node_vc.bo, u, d_uv - 2 * node_vc.r))
                    else
                        push!(edges, (node_vc.bo,u,d,d_uv))
                    end
                end
                cp = node_vc.sn
            end
        end
    end

    emin_dict = Dict((u,v)=> d for (u,v,d,_) in edges) #d: shortest descendent edge. nonpositive for essential edges.
    emin_keys = keys(emin_dict)
    for i=2:length(ct.nodes)
        p = nodes[ibo[i]].parent
        emin_dict[Int32(i),nodes[p].bo] = f_T(-Inf)
    end
    #sort!(missing_edges; alg=QuickSort)
    sort!(edges; alg=QuickSort)
    mind_heap = Vector{Tuple{f_T,Int32,Int32}}() #u,v,d_est
    resize!(ctimes, length(ct.nodes))
    fill!(ctimes, zero(f_T))





    #
    # Step two: compute new contraction times, compute emind, mark edges we take
    #
    prec_tmp = Affine_cutoff(eps_abs, f_T(Inf),prec)
    posR = length(edges)
    evalref = Ref(0)
    res_edges = Vector{Tuple{Int32, Int32, f_T}}()
    #mind_heap2 = copy(mind_heap)
    for i=length(ct.nodes):-1:1
        posL = posR
        posR == 0 && continue
        #grab all the edges from the relevant node i, compute bounds on contraction time
        tmin = ctimes[i]
        #println("[$i]: tmin=$(tmin) initially")
        #me_last_e = length(missing_edges)
        while posL > 0
            edge = edges[posL]
            u,v,d_uv,d_par = edge #edges[posL]
            (u != i) && (posL+=1; break)
            posL -= 1
            #put missing edges into mind_heap
            
            mind = emin_dict[u,v]
            empty!(mind_heap)
            missing_edges!(mind_heap, ct, edge, emin_keys)
            for (d_est, meu, mev) in mind_heap
                if mev == u 
                    tmin = max(tmin, ctimes[meu])
                end
            end
            mind = _dmin_compute(ct, mind_heap, mind, evalref)
            dm = mind
            emin_dict[u, v] = mind
            nec = (mind < 0)
            
            if d_uv <= d_par
                if nec || dm <= eval_down(prec_res, d_par)
                    if d_par > tmin 
                        tmin = d_par
                    end
                end
            else
                if nec || dm <= eval_down(prec_res, d_uv)
                    if d_uv > tmin 
                        tmin = d_uv
                    end
                end
            end
            (posL==0) && (posL+=1; break)
        end
        ctimes[i] = tmin
        #println("[$i]: tmin=$(tmin) finally")

        #have computed contraction time; propagate this upwards to parent
        if i > 1
            parent = ct.nodes[ct.nodes[ibo[i]].parent].bo
            ctimes[parent] = max(ctimes[parent], tmin)
        end

        #have computed contraction time, drop unnecessary edges, propagate up the new ctime, propagate up dmin
        for j = posL : posR
            u,v,d_uv,d_par = edges[j]
            dm = emin_dict[u,v]
            nec = (dm < 0)
            if nec
                @assert d_uv <= tmin && d_par <= tmin
                #@goto reg_nec
                #println("[i]: Recursive edge ($u,$v) with d=$d_uv ")
                dm = f_T(-Inf)
                if u <= Nmax && v <= Nmax
                    push!(res_edges, (u,v,d_uv))
                end
            elseif tmin < d_uv <= d_par || tmin <d_par <= d_uv
                @assert dm >= eval_down(prec_res, d_par) 
                #println("[i]: dpar-rejected edge ($u,$v) with d=$d_uv, dpar=$(d_par) dmin=$(dm) at tmin=$tmin dparlim=$(eval_down(prec_res, d_par))")

                #@goto reg_drop
            elseif d_uv <= tmin < d_par
                @assert dm >= eval_down(prec_res, d_par)
                #println("[i]: dpar-rejected edge ($u,$v) with d=$d_uv, dpar=$(d_par) dmin=$(dm) at tmin=$tmin dparlim=$(eval_down(prec_res, d_par))")

                #@goto reg_drop
            elseif d_par <= tmin < d_uv
                @assert dm >= eval_down(prec_res, tmin) 
                #println("[i]: d-rejected edge ($u,$v) with d=$d_uv, dpar=$(d_par) dmin=$(dm) at tmin=$tmin tlim=$(eval_down(prec_res, tmin))")

               # @goto reg_drop
            else
                #may choose to take or not
                if dm <= eval_down(prec_res, tmin) 
                    #take edge
                    #@goto reg_nec
                    #println("[i]: Accepted edge ($u,$v) with d=$d_uv, dpar=$(d_par) dmin=$(dm) at tmin=$tmin tlim=$(eval_down(prec_res, tmin))")
                    dm = f_T(-Inf)
                    push!(res_edges, (u,v,d_uv))

                else
                    @assert dm >= eval_down(prec_res, tmin)
                   # @goto reg_drop
                   #println("[i]: Rejected edge ($u,$v) with d=$d_uv, dpar=$(d_par) dmin=$(dm) at tmin=$tmin tlim=$(eval_down(prec_res, tmin))")

                end
            end
            #@assert false
            #@label reg_nec
            emin_dict[u,v] = dm #f_T(-Inf)
            if ibo[u] > 1
                par = nodes[nodes[ibo[u]].parent].bo
                if par < v                
                    emin_dict[v,par] = min(emin_dict[v,par], dm) #f_T(-Inf) #(emin_dict[v,par][1], true)
                elseif  par > v
                    emin_dict[par,v] = min(emin_dict[par,v], dm) #f_T(-Inf) #(emin_dict[par,v][1], true)
                end
                ctimes[par] = max(tmin, ctimes[par])
            end
        end
        #point handled
        posR = posL-1
    end
    #@assert me_last == 0
    evals += evalref[]
    #res_edges = Vector{Tuple{Int32, Int32, f_T}}()
    #for (u,v,d,dp) in edges
   #     emin_dict[u,v] < 0 || push!(res_edges, (u,v,d)) 
    #end
    sort!(res_edges; alg=QuickSort)
    #@show evals, evalref[], maxheapsize[], length(res_edges), length(edges)#, length(missing_edges)
    return SparseEdges(ct, Nmax, res_edges, evals, prec_res)#, ctimes
end

