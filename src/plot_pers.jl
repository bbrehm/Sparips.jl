using RecipesBase: plot, plot! 

_logspace(start, stop, num=50) = exp.(range(log(start), stop=log(stop), length=num));



function _filtbd(prec1, prec2, dvals, cutmin)
    bv = Float64[]
    dv = Float64[]
    for d in dvals
      b = eval_down(prec2, eval_down(prec1, d))
      if b >= cutmin
        push!(dv, d)
        push!(bv, b)
      end
    end
    (bv, dv)
end

function plotpers(pers::pers_diagram; kwargs...)
    kwdict = Dict{Symbol, Any}(kwargs)
    logscale::Bool = pop!(kwdict, :logscale, false)
    filter_arg = pop!(kwdict, :filter, 1.0)
    nofilter = false
    real_markersize = pop!(kwdict, :real_markersize, 2.0)
    maybe_markersize = pop!(kwdict, :maybe_markersize, 1.0)

    if filter_arg isa AbstractPrecision
        filter_ = filter_arg 
    elseif filter_arg === nothing
        nofilter = true
        filter_ = Linear_prec(1.0)
    elseif filter_arg isa Real
        filter_ = Linear_prec(Float64(filter_arg))
    else
        throw("unclear filter $(filter_arg) of type $(typeof(filter_arg))")
    end
    cutmax = Float64(pers.prec.Rmax)*1.2
    if logscale
        if pers.prec.eps_abs >0
            cutmin = Float64(pers.prec.eps_abs/4)
        else
            cutmin = minimum(b for pp in pers.bars for (b,d) in pp if b>0)/1.5
        end
    else
        cutmin = 0.0
    end


    idfun = (logscale ? _logspace(cutmin, cutmax, 200) : collect(range(cutmin; stop=cutmax, length=200) ))
    psifunb, psifund = _filtbd(filter_ , pers.prec , idfun, cutmin)
    idfun .= max.(cutmin, min.(cutmax, idfun))
    psifunb .= max.(cutmin, min.(cutmax, psifunb))
    psifund .= max.(cutmin, min.(cutmax, psifund))

    plts = []
    for i=1:length(pers.bars)
        pts_b_real = Float64[]
        pts_d_real = Float64[]
        rec_b_real = Float64[]
        rec_d_real = Float64[]
        pts_b = Float64[]
        pts_d = Float64[]
        rec_b = Float64[]
        rec_d = Float64[]
        for ((_b,b), (_d,d)) in pers.errbars[i]
            if _d > b
                push!(pts_b_real, b)
                push!(pts_d_real, d)
                append!(rec_b_real, (b, _b, _b, b, NaN))
                append!(rec_d_real, (d, d, _d, _d, NaN))
            elseif nofilter || (_d > 0 ) && (_tmp = eval_down(filter_, d); (_tmp > b) &&  eval_down(filter_, _tmp) >0 )
                push!(pts_b, b)
                push!(pts_d, d)
                append!(rec_b, (b, _b, _b, b, NaN))
                append!(rec_d, (d, d, _d, _d, NaN))
            end
        end
 
        for arr in (pts_b_real, pts_d_real, rec_b_real, rec_d_real, pts_b, pts_d, rec_b, rec_d)
            arr .= max.(cutmin, arr)
            arr .= min.(cutmax, arr)
        end

        p = plot(; legend=false, title="Dim $(i-1)", xlabel="birth", ylabel="death")
        if logscale
             plot!(p;xaxis=:log, yaxis=:log)
        end
        #first plot rectangles. order important because of occlusions.
        plot!(p, rec_b, rec_d; linewidth=0, alpha=0.4, color=:yellow, seriestype=:shape)
        plot!(p, rec_b_real, rec_d_real; linewidth=0, alpha=0.4, color=:red, seriestype=:shape)
        real_markersize > 0 &&
            plot!(p, pts_b, pts_d; color=:black, seriestype=:scatter, markersize=real_markersize)
        maybe_markersize > 0 &&            
            plot!(p, pts_b_real, pts_d_real; color=:black, seriestype=:scatter, markersize=maybe_markersize)

        plot!(p, idfun, idfun; color=:black, linewidth=1.0)
        plot!(p, psifunb, psifund; linewidth=1.0, color=:green)
        #also plot the cutoff-grid:
        if pers.prec.eps_abs > 0
            plot!(p, [cutmin, cutmax], [pers.prec.eps_abs, pers.prec.eps_abs]; seriestype=:path, style=:dash, color=:black)
            plot!(p, [pers.prec.eps_abs, pers.prec.eps_abs], [cutmin, cutmax]; seriestype=:path, style=:dash, color=:black)
        end

        push!(plts, p)
    end
    plts
end







#=
function _plotpers(pers::pers_diagram, logscale::Bool, filter::AbstractPrecision)
    #find out plotmin

    if logscale
        adjust = Cutoff(Dummy_prec(),Float64(pers.prec.eps_abs)/4, Float64(pers.prec.Rmax))
        idfun = _logspace(adjust.eps_abs, adjust.Rmax*1.1, 200);
        
    else
        adjust = Cutoff(Dummy_prec(), 0.0, Float64(pers.prec.Rmax))
        idfun = range(0.0; stop=adjust.Rmax*1.1, length=200);
    end
    psifun = broadcast(t->eval_up(filter, eval_up(pers.prec, t)),idfun)
    psifun_pre = broadcast(t-> eval_up(pers.prec, t),idfun)
    plts=[]
    for i=1:pers.dimscomplete
        #pts = Vector{Tuple{Float64, Float64}}[]
        pts_b_real = Float64[]
        pts_d_real = Float64[]
        rec_b_real=Float64[]
        rec_d_real=Float64[]
        pts_b = Float64[]
        pts_d = Float64[]
        rec_b=Float64[]
        rec_d=Float64[]
        for (b,d) in pers.bars[i]
            b_ = eval_up(pers.prec, b)
            if b_ < d
                _b = _eval_down(idfun, psifun_pre, b)
                _d = _eval_down(idfun, psifun_pre, d)
                #cutoff
                b=eval_up(adjust, b)
                d=eval_up(adjust, d)
                _b=eval_up(adjust, _b)
                _d=eval_up(adjust, _d)

                push!(pts_b_real, b)
                push!(pts_d_real, d)

                push!(rec_b_real, b)
                push!(rec_d_real, d)

                push!(rec_b_real, _b)
                push!(rec_d_real, d)

                push!(rec_b_real, _b)
                push!(rec_d_real, _d)

                push!(rec_b_real, b)
                push!(rec_d_real, _d)

                push!(rec_b_real, NaN)
                push!(rec_d_real, NaN)
                continue
            end
            if eval_up(filter, b) < d
                _b = _eval_down(idfun, psifun_pre, b)
                _d = _eval_down(idfun, psifun_pre, d)
                #cutoff
                b=eval_up(adjust, b)
                d=eval_up(adjust, d)
                _b=eval_up(adjust, _b)
                _d=eval_up(adjust, _d)

                push!(pts_b, b)
                push!(pts_d, d)

                push!(rec_b, b)
                push!(rec_d, d)

                push!(rec_b, _b)
                push!(rec_d, d)

                push!(rec_b, _b)
                push!(rec_d, _d)

                push!(rec_b, b)
                push!(rec_d, _d)

                push!(rec_b, NaN)
                push!(rec_d, NaN)
                continue
            end
        end
        if logscale
            p=plot(;xaxis=:log, yaxis=:log, legend=false, title="Dim $(i-1)")
        else
            p=plot(; legend=false, title="Dim $(i-1)")
        end
        plot!(p, Shape(rec_b, rec_d); linewidth=0, alpha=0.4, color=:blue)
        plot!(p, Shape(rec_b_real, rec_d_real); linewidth=0, alpha=0.4, color=:red)
        plot!(p, pts_b, pts_d; color=:black, seriestype=:scatter, markersize=2.0)
        plot!(p, pts_b_real, pts_d_real; color=:black, seriestype=:scatter, markersize=2.0)

        plot!(p, idfun, idfun; color=:black,linewidth=1.0)
        plot!(p, idfun, psifun;linewidth=1.0, color=:green)
        push!(plts, p)
    end

    return plts
end

=#