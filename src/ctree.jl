

struct Ctree_node{f_T} #T should be either Float32 or Float64
    #radius. Upper bound of d(x, parent) for all descendents of the node, including itself
    r::f_T    
    parent::Int32  #0 for root
    cf::Int32 #first child, 0 for childless nodes
    sn::Int32 #next sibling, 0 for youngest child
    bo::Int32 # position of the node in birth-order or -1
end

# A ctree stores its nodes in a flat array ctree.nodes; all links refer to positions of nodes 
# in the array and are -1 or 0 if unset. The root is at position 1, and is the only one with node.parent<1.
#
# ctree.data[n] stores the datapoint corresponding to ctree.nodes[n]; we never relocate.
# The ctree can be interpreted as a tree of variable branch-out, where the children from a r-sorted linked,
# in descending order
#
# node.r is always an upper bound to dist(z, node.parent), where z ranges over all descendents 
# of node, including node itself 
# 
# We always have node.r <= nodes[node.parent].r
#
# ibo, when set, stores the order-by-r, where ties are broken by the partial tree order.
mutable struct Ctree{dist_T<:Metric, point_T, f_T} 
    dist::dist_T #distance function 
    nodes::Vector{Ctree_node{f_T}} 
    data::Vector{point_T}
    evals::Int64 #number of performed distance evaluations
    ibo::Vector{Int32}  # inverse birth-order. ibo[n] points to the nth node in birth-order.
                        # empty if birth-order has not yet been computed
end



function Ctree(data::Vector{point_T}; dist=Euclidean(), construct::Bool=true) where {point_T}
    f_T = typeof(evaluate(dist, data[1], data[1]))
    return _ctree(f_T, dist, data, construct)
end 

function Ctree(data::Matrix{f_T}; dist=Euclidean(), construct::Bool=true) where {f_T}
    data_ = copy(reshape(reinterpret(SVector{size(data,1),f_T},data), size(data,2)))
    return _ctree(f_T, dist, data_, construct)
end

function _ctree(::Type{f_T}, dist::dist_T, data::Vector{point_T}, construct)  where  {f_T, dist_T<:Metric, point_T}
    nodes = Vector{Ctree_node{f_T}}(undef, length(data))
    ct = Ctree{dist_T, point_T,f_T}(dist, nodes, data, 0, Vector{Int32}())
    if construct
        construct_ct!(ct)
        empirical_rad!(ct)
        reorder_siblings!(ct)
    end
    ct
end

function construct_ct!(ct::Ctree{dist_T, point_T, f_T}) where {dist_T, point_T, f_T}
    ct.nodes[1] = Ctree_node(f_T(Inf), Int32(-1),Int32(-1),Int32(-1), Int32(-1))
    queue = Vector{Tuple{Int32, f_T}}(undef, length(ct.nodes))
    evals = 0
    for i = 2:length(ct.nodes)        
        off = 1
        len = 2
        dbest, pbest = f_T(Inf), Int32(-1)
        queue[1] = (Int32(1), f_T(Inf))
        while(off < len)
            c_id,dp = queue[off]
            off += 1
            #@show i,off, len, c_id
            c = ct.nodes[c_id]
            c.r < dp && continue
            if c.sn > 0
                queue[len] = (c.sn, dp)
                len += 1
            end
            d = evaluate(ct.dist,ct.data[i], ct.data[c_id])
            evals += 1
            if  d < dbest && f_T(4)*d < c.r
                #alternative: d < dbest && roundup2(d)<c.r #
                dbest = d
                pbest = c_id
            end
            if c.cf > 0
                queue[len] = (c.cf, d)
                len += 1
            end            
        end
        parent = ct.nodes[pbest]
        r = f_T(2)*dbest 
        #alternative: r=roundup2(dbest)*2 #
        sp, sn = Int32(-1), parent.cf
        while sn > 0
            sibling = ct.nodes[sn]
            sibling.r <= r && break
            sp, sn = sn, sibling.sn
        end
        node = Ctree_node(r, pbest, Int32(-1), sn, Int32(-1))
        ct.nodes[i]=node
        if sp >0 
            sibling_prev = ct.nodes[sp]
            ct.nodes[sp] = Ctree_node(sibling_prev.r, sibling_prev.parent, sibling_prev.cf, Int32(i), Int32(-1))
        else
            pnew = Ctree_node(parent.r, parent.parent, Int32(i), parent.sn, Int32(-1))
            ct.nodes[pbest] = pnew
        end
    end
    ct.evals = evals
    return ct
end


@noinline function empirical_rad!(ct::Ctree{dist_T, point_T, f_T}) where {dist_T, point_T, f_T}
    evals = 0
    oldr = [n.r for n in ct.nodes]
    nodes = ct.nodes
    data=ct.data
    dist=ct.dist
    for i=2:length(nodes)
        node = nodes[i]
        nodes[i]=Ctree_node(f_T(-Inf), node.parent, node.cf, node.sn, Int32(-1))
    end
    for i=2:length(nodes)
        pt = data[i]
        bs = i
        node = nodes[i]

        rm = f_T(-Inf)
        while node.parent > 0
            rp = max(rm,evaluate(dist, pt, data[node.parent]))
            @assert rp <= oldr[bs]
            evals += 1
            if rp > node.r
                nodes[bs]=Ctree_node(rp, node.parent, node.cf, node.sn, Int32(-1))
            end
            bs = node.parent
            node = nodes[bs]
        end
    end
    ct.evals += evals
    nothing
end




function reorder_siblings!(ct::Ctree{dist_T, point_T, f_T}) where  {dist_T, point_T, f_T}
    nodes = ct.nodes
    
    rv = [node.r for node in nodes]
    perm = sortperm!(Vector{Int32}(undef, length(nodes)), rv; rev=true, alg=MergeSort) #must be stable
    iperm=Vector{Int32}(undef, length(nodes))
    for i=1:length(nodes)
        iperm[perm[i]] = i
    end


    tmp_rp = Vector{Tuple{f_T, Int32}}()


    for i = 1:length(nodes)
        node = nodes[i]
        if node.cf > 0
            push!(tmp_rp, (nodes[node.cf].r, node.cf))
            node = nodes[node.cf]
            while node.sn > 0
                push!(tmp_rp, (nodes[node.sn].r, node.sn))
                node = nodes[node.sn]
            end
            if length(tmp_rp)>1
                 sort!(tmp_rp; rev=true, alg=MergeSort, by=x->x[1])
            end

            node = nodes[i]
            nodes[i] = Ctree_node(node.r, node.parent, tmp_rp[1][2], node.sn, iperm[i])
            for j=1:length(tmp_rp)-1
                rr,jj = tmp_rp[j]
                node = nodes[jj]
                @assert node.r==rr
                nodes[jj] = Ctree_node(node.r, node.parent, node.cf, tmp_rp[j+1][2], iperm[jj])
            end
            rr,j = tmp_rp[length(tmp_rp)]
            node = nodes[j]
            nodes[j]= Ctree_node(node.r, node.parent, node.cf, Int32(-1), iperm[j])
            resize!(tmp_rp,0)
        else
            node = nodes[i]
            nodes[i]= Ctree_node(node.r,node.parent,node.cf, node.sn, iperm[i])
        end
    end

    ct.ibo = perm
    nothing
end



