using RxInfer
using LinearAlgebra
using Random, Distributions
using Flux, ForwardDiff
using StableRNGs
using Bijectors

include("helpers.jl")

dist_m = Uniform(0, 3)
f_m(x) = inv(bijector(dist))(x)
finv_m(x) = bijector(dist)(x)

dist_x = Uniform(-5, 5)
f_x(x) = inv(bijector(dist))(x)
finv_x(x) = bijector(dist)(x)


# 
Dm(x) = x <= 0 ? x + 1 : 0.0
Dp(x) = x <= 0 ? inv(0.1) : inv(100.0)

# idea, linearization + cvi
@model function model2(n)
    τ = datavar(Float64, n)
    zα = randomvar(n)

    α ~ Beta(1.0, 1.0)

    x ~ Normal(μ=0.0, v=3.0)
    m ~ Normal(μ=1.5, v=3.0)

    x_ ~ f_x(x)
    m_ ~ f_m(m)

    dm ~ Dm(x_)
    dp ~ Dp(x_)

    zα .~ Bernoulli(α)
    
    x_m ~ x_ + m_ 

    for i in 1:n
        y  ~ NormalMixture(zα[i], (dm, x_m), (dp, 10.0))
        ŷ  ~ NormalMixture(zα[i], (dm, x_m), (dp, 10.0))

        τ[i] ~ Normal(μ = y - ŷ, v=1e-4)
    end

end

init_marginals = (x_m=NormalMeanVariance(), dm = NormalMeanVariance(), dp = GammaShapeRate(), m = NormalMeanVariance(), x = NormalMeanVariance(), 
                  m_ = NormalMeanVariance(), x_ = NormalMeanVariance(), 
                  α  = vague(Beta), ŷ=NormalMeanVariance(), y=NormalMeanVariance())

@meta function model2_meta(seed, n_samples, itrs, optimizer=Descent(0.01))
    Dm() -> CVI(StableRNG(seed), n_samples, itrs, optimizer, true)
    Dp() -> CVI(StableRNG(seed), n_samples, itrs, optimizer, true)
    f_x() -> DeltaMeta(method=Linearization(), inverse=finv_x)
    f_m() -> DeltaMeta(method=Linearization(), inverse=finv_m)
end

n = 1
res = inference(model = model2(n), data=(τ = [ 0.0 for i in 1:n ],), free_energy=true, initmarginals = init_marginals, initmessages = init_marginals, iterations=100, showprogress=true, constraints=MeanField(), meta=model2_meta(42, 1000, 100)
)

mean(res.posteriors[:α][end])

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