using RxInfer
using LinearAlgebra
using Random, Distributions
using Flux, ForwardDiff
using StableRNGs

include("helpers.jl")

function uniform_m(m)
    aₘ, bₘ = 0.0, 3.0
    m_ = m in aₘ:bₘ ? 1/(bₘ-aₘ) : 0 
end

function uniform_x(x)
    aₓ, bₓ = -5.0, 5.0
    x_ = x in aₓ:bₓ ? 1/(bₓ-aₓ) : 0
end

function D(x)
    # aₓ, bₓ = -5.0, 5.0
    # x_ = x in aₓ:bₓ ? 1/(bₓ-aₓ) : 0
    x_ = x
    x_ <= 0 ? pdf(NormalMeanVariance(x_ + 1, 0.1), x_) : pdf(NormalMeanVariance(0.0, 100.0), x_)
end

# idea, linearization + cvi
@model function model2()
    τ = datavar(Float64)

    α ~ Beta(1.0, 1.0)

    # workaround for uniform

    x ~ NormalMeanVariance(0.0, 1.0)
    m ~ NormalMeanVariance(3.0, 1.0)

    x_ ~ uniform_x(x) where {meta=Unscented()}
    m_ ~ uniform_m(m) where {meta=Unscented()}

    x_m ~ x_ + m_ 

    # d ~ D(x_) where {meta=CVIApproximation(StableRNG(42), 1000, 1000, Descent(0.1))}
    d ~ D(x_) where {meta=Unscented()}

    zα ~ Bernoulli(α)
    
    y  ~ NormalMixture(zα, (d, x_m), (1e4, 10.0))
    ŷ  ~ NormalMixture(zα, (d, x_m), (1e4, 10.0))

    τ ~ NormalMeanPrecision(y-ŷ, 1e4)

end

init_marginals = (x_m=NormalMeanPrecision(), d = NormalMeanPrecision(), m = NormalMeanPrecision(), x = NormalMeanPrecision(), 
                  m_ = NormalMeanPrecision(), x_ = NormalMeanPrecision(), 
                  α  = vague(Beta), ŷ=NormalMeanPrecision(), y=NormalMeanPrecision())

res = inference(model = model2(), data=(τ = 0.0,), free_energy=false, initmarginals = init_marginals, initmessages = init_marginals, iterations=10, showprogress=true, constraints=MeanField())

res.posteriors[:α][end]
@. mean(res.posteriors[:x_])[end]
@. var(res.posteriors[:x_])[end]
@. mean(res.posteriors[:m_])[end]
@. var(res.posteriors[:m_])[end]

@. mean(res.posteriors[:x])[end]
@. var(res.posteriors[:x])[end]
@. mean(res.posteriors[:m])[end]
@. var(res.posteriors[:m])[end]


@. mean(res.posteriors[:y])[end]
@. var(res.posteriors[:y])[end]
@. mean(res.posteriors[:ŷ])[end]
@. var(res.posteriors[:ŷ])[end]
