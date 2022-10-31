using Rocket, ReactiveMP, GraphPPL
using LinearAlgebra
using Random, Distributions
using Flux, ForwardDiff

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
    aₓ, bₓ = -5.0, 5.0
    x_ = x in aₓ:bₓ ? 1/(bₓ-aₓ) : 0
    x_ <= 0 ? pdf(NormalMeanVariance(x_ + 1, 0.1), x_) : pdf(NormalMeanVariance(0.0, 100.0), x_)
end


@model [ default_factorisation = MeanField() ] function model2()
    τ = datavar(Float64)

    α ~ Beta(1.0, 1.0)

    # workaround for uniform

    # x ~ NormalMeanVariance(0, 1.0)
    # m ~ NormalMeanVariance(-4.0, 1.0)

    # x_ ~ uniform_x(x) where {meta=CVIApproximation(2000, 1000, Descent(0.1))}
    # m_ ~ uniform_m(m) where {meta=CVIApproximation(2000, 1000, Descent(0.1))}

    # x__ ~ NormalMeanVariance(x_, 1.0)
    # m__ ~ NormalMeanVariance(m_, 1.0)


    x__ ~ NormalMeanVariance(0.0, 1.0)
    m__ ~ NormalMeanVariance(1.0, 1.0)

    x_m ~ x__ + m__ 

    d ~ D(x__) where {meta=CVIApproximation(2000, 1000, Descent(1.0))}

    zα ~ Bernoulli(α)
    
    y  ~ NormalMixture(zα, (d, x_m), (100.0, 10.0))
    ŷ  ~ NormalMixture(zα, (d, x_m), (100.0, 10.0))

    τ ~ NormalMeanVariance(y-ŷ, 1e-4)

end

init_marginals = (x_m=NormalMeanPrecision(), d = NormalMeanPrecision(), m = NormalMeanPrecision(), x = NormalMeanPrecision(), 
                  m_ = NormalMeanPrecision(), x_ = NormalMeanPrecision(), 
                  α  = vague(Beta), ŷ=NormalMeanPrecision(), y=NormalMeanPrecision())

res = inference(model = Model(model2), data=(τ = 0.0,), free_energy=false, initmarginals = init_marginals, initmessages = init_marginals, iterations=100, showprogress=true)

res.posteriors[:α][end]
@. mean(res.posteriors[:x__])[end]
@. var(res.posteriors[:x__])[end]
@. mean(res.posteriors[:m__])[end]
@. var(res.posteriors[:m__])[end]
