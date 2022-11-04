using StatsFuns: softmax, softmax!


@rule NormalMixture{N}(:switch, Marginalisation) (
    q_out::Any,
    q_m::ManyOf{N, Any},
    q_p::ManyOf{N, Any}
) where {N} = begin
    U = map(zip(q_m, q_p)) do (m, p)
        return -score(
            AverageEnergy(),
            NormalMeanPrecision,
            Val{(:out, :μ, :τ)},
            map((q) -> Marginal(q, false, false), (q_out, m, p)),
            nothing
        )
    end
    return Categorical(clamp!(softmax!(U), tiny, one(eltype(U)) - tiny))
end

@rule NormalMixture{N}((:m, k), Marginalisation) (q_out::Any, q_switch::Bernoulli, q_p::Any, ) where {N} = begin 
    pv    = probvec(q_switch)
    T     = eltype(pv)
    z_bar = clamp.(pv, tiny, one(T) - tiny)
    return NormalMeanVariance(mean(q_out), inv(z_bar[k] * mean(q_p)))
end

@rule NormalMixture{2}((:m, k), Marginalisation) (q_out::NormalMeanVariance, q_switch::Bernoulli, q_p::GammaShapeRate, ) = begin 
    pv    = probvec(q_switch)
    T     = eltype(pv)
    z_bar = clamp.(pv, tiny, one(T) - tiny)
    return NormalMeanVariance(mean(q_out), inv(z_bar[k] * mean(q_p)))
end

@rule NormalMixture{2}((:p, k), Marginalisation) (q_out::NormalWeightedMeanPrecision, q_switch::Bernoulli, q_m::SampleList, ) = begin 
    m_mean_k, v_mean_k = mean_cov(q_m)
    m_out, v_out       = mean_cov(q_out)
    z_bar              = probvec(q_switch)
    return GammaShapeRate(one(eltype(z_bar)) + z_bar[k] / 2, z_bar[k] * (v_out + v_mean_k + abs2(m_out - m_mean_k)) / 2)
end

@average_energy NormalMixture (
    q_out::Any,
    q_switch::Any,
    q_m::ManyOf{N, Any},
    q_p::ManyOf{N, Any}
) where {N} = begin
    z_bar = probvec(q_switch)
    return mapreduce(+, 1:N, init = 0.0) do i
        return z_bar[i] * score(
            AverageEnergy(),
            NormalMeanPrecision,
            Val{(:out, :μ, :τ)},
            map((q) -> Marginal(q, false, false), (q_out, q_m[i], q_p[i])),
            nothing
        )
    end
end


@rule NormalMixture{N}(:out, Marginalisation) (
    q_switch::Any,
    q_m::ManyOf{N, Any},
    q_p::ManyOf{N, Any}
) where {N} = begin
    πs = probvec(q_switch)
    return NormalMeanPrecision(sum(πs .* mean.(q_m)), sum(πs .* mean.(q_p)))
end