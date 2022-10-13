using Rocket, ReactiveMP, GraphPPL
using LinearAlgebra
using Random, Distributions
using Flux, ForwardDiff

include("helpers.jl")

function uniform_sum(x, m)
    aₓ, bₓ = -5.0, 5.0
    aₘ, bₘ = 0.0, 3.0
    x_ = x in aₓ:bₓ ? 1/(b-a) : 0
    m_ = m in aₘ:bₘ ? 1/(b-a) : 0 
    return x_ + m_
end

function D(x)
    aₓ, bₓ = -5.0, 5.0
    x_ = x in aₓ:bₓ ? 1/(b-a) : 0
    x_ <= 0 ? pdf(NormalMeanVariance(x_ + 1, 0.1), x_) : pdf(NormalMeanVariance(0.0, 100.0), x_)
end

function constraint(y₁, y₂)
    return y₁ == y₂ ? 1.0 : 0.0
end


@model [ default_factorisation = MeanField() ] function model2()
    τsoft = datavar(Float64)

    α ~ Beta(1.0, 1.0)
    x ~ NormalMeanVariance(10.0, 10.0)
    m ~ NormalMeanVariance(-10.0, 10.0)

    xxmm ~ uniform_sum(x, m) where {meta=CVIApproximation(2000, 1000, Descent(1.0))}

    d ~ D(x) where {meta=CVIApproximation(2000, 1000, Descent(0.1))}

    zα ~ Bernoulli(α)
    
    y  ~ NormalMixture(zα, (d, xxmm), (10000.0, 10))
    ŷ  ~ NormalMixture(zα, (d, xxmm), (10000.0, 10))

    τ ~ constraint(y, ŷ) where {meta=CVIApproximation(2000, 1000, Descent(0.1))}

    τsoft ~ NormalMeanVariance(τ, 1e-5)

end

init_marginals = (xxmm=NormalMeanPrecision(), d = NormalMeanPrecision(), m = NormalMeanPrecision(), x = NormalMeanPrecision(), α  = vague(Beta), τ = NormalMeanPrecision(), ŷ=NormalMeanPrecision(), y=NormalMeanPrecision())
res = inference(model = Model(model2), data=(τsoft = 1.0,), free_energy=false, initmarginals = init_marginals, initmessages = init_marginals, iterations=10, showprogress=true)