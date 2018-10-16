
#=
Given an precision description, one can:
    s = eval_up(prec, r): Rips(r) embeds into SRips(s)
    r = eval_down(prec, s): Rips(r) embeds into SRips(s)
    eval_Q:
    2*r <= eval_Q(prec, r - eval_down(prec, r))
=#

abstract type AbstractPrecision end

struct Dummy_prec <: AbstractPrecision end
eval_down(::Dummy_prec, x) = x


struct Linear_prec{fT} <: AbstractPrecision
    eps::fT
    epsi::fT
end
function Linear_prec(eps::fT) where fT
    (eps <= 0 || !isfinite(eps) || isnan(eps)) && throw("invalid precision $(eps)")
    if eps > 1
        return Linear_prec(eps, 1/eps)
    else
        return Linear_prec(1/eps, eps)
    end
end


function eval_down(prec::Linear_prec, r::fT) where {fT}
    return fT(r*prec.epsi)
end




struct Affine_cutoff{fT, precT<:AbstractPrecision} <: AbstractPrecision
    eps_abs::fT
    Rmax::fT
    prec::precT
end

function eval_down(prec::Affine_cutoff, r::fT) where {fT}
    return fT( min(r - prec.eps_abs, eval_down(prec.prec, r)))
end



struct Q_cmp{fT, precT<:AbstractPrecision}
    r::fT
    prec::precT
end


function eval_Q(prec::AbstractPrecision, r::fT) where {fT}
    return Q_cmp(r, prec)
end

#assumes: t-> t-eval_down(t) is positive and monotone
function Base.:<(qcmp::Q_cmp, s)::Bool
    #true iff eval_up(s-2*qcmp.r) > s
    #i.e. for eps_rel: (1+eps_rel)(s-2r) > s , i.e. s > 2*r*(1+1/eps_rel)
    # for eps_abs, large s: true if s - 2*qcmp.r + eps_abs > s, i.e. eps_abs > 2*r
    # resp. small s < 2*r: if eps_abs > s: t(r)<s if s
   # eu = eval_up(qcmp.prec, s-2*qcmp.r)
    #return eu > s && eu > qcmp.r
    
    #alt: s>2*qcmp.r && eval_up(qcmp.prec, s-2*qcmp.r) > s 
    #
    2*qcmp.r < s - eval_down(qcmp.prec, s)
end

