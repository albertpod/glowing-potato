using RxInfer
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

# idea, linearization + cvi
@model [ default_factorisation = MeanField() ] function model2()
    τ = datavar(Float64)

    α ~ Beta(1.0, 1.0)

    # workaround for uniform

    x ~ NormalMeanVariance(0, 1.0)
    m ~ NormalMeanVariance(-4.0, 1.0)

    x_ ~ uniform_x(x) where {meta=Linearization()}
    m_ ~ uniform_m(m) where {meta=Linearization()}

    x_m ~ x_ + m_ 

    d ~ D(x_) where {meta=UT()}

    zα ~ Bernoulli(α)
    
    y  ~ NormalMixture(zα, (d, x_m), (1000.0, 10.0))
    ŷ  ~ NormalMixture(zα, (d, x_m), (1000.0, 10.0))

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
