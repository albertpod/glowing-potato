using Rocket, ReactiveMP, GraphPPL
using LinearAlgebra
using Random, Distributions

include("helpers.jl")

@model [ default_factorisation = MeanField() ] function model1(n)
    y = randomvar(n)
    x = datavar(Float64, n)
    β ~ Beta(1.0, 1.0)
    α ~ Beta(1.0, 1.0)

    zβ = randomvar(n)
    zα = randomvar(n)
    
    m ~ NormalMeanVariance(0.0, 100.0)

    for i in 1:n
        zβ[i] ~ Bernoulli(β)
        zα[i] ~ Bernoulli(α)
        y[i] ~ NormalMixture(zβ[i], (0.0, 1.0), (1.0, 1.0))
        x[i] ~ NormalMixture(zα[i], (m, y[i]), (1.0, 1.0))
    end

end

function generate_dataset(n, priors)
    βdist = priors[:β]
    αdist = priors[:α]
    mdist = priors[:m]

    β = mean(βdist)
    α = mean(αdist)
    m = mean(mdist)

    obs, lat = [], []
    for _ in 1:n
        # change m, β, α to rand(mdist), rand(βdist) ...
        push!(lat, rand(MixtureModel(Normal[Normal(0.0, 1.0),Normal(1.0, 1.0)], [β, 1-β])))
        push!(obs, rand(MixtureModel(Normal[Normal(m, 1.0), Normal(lat[end], 1.0)], [α, 1-α])))
    end
    return αdist, βdist, mdist, lat, obs
end

n = 1000

priors = Dict(:m => Normal(5.0, 1e-1), :α => Beta(10.0, 1.0), :β => Beta(1.0, 1.0))
real_α, real_β, real_m, y_lat, x_obs = generate_dataset(n, priors)

model = Model(model1, length(x_obs))
data  = (x = x_obs,)

initmarginals = (
    β  = vague(Beta), 
    α  = vague(Beta), 
    y = NormalMeanVariance(0.0, 1e2),
    m = NormalMeanVariance(0.0, 1e2),
)


result = inference(
    model = model, 
    data  = data, 
    initmarginals = initmarginals, 
    iterations  = 100, 
    free_energy = true,
    showprogress = true,
)

# switches
za = @. mean(result.posteriors[:zα][end])
zb = @. mean(result.posteriors[:zβ][end])

inf_α = result.posteriors[:α][end]
inf_β = result.posteriors[:β][end]
inf_m = result.posteriors[:m][end]

y_inf = result.posteriors[:y][end]



using StatsPlots

plot(inf_α, label="infered mean $(mean(inf_α))", legend=:left)
plot!(real_α, label="infered mean $(mean(real_α))", legend=:left)

plot(inf_β, label="infered mean $(mean(inf_β))", legend=:left)
plot!(real_β, label="infered mean $(mean(real_β))", legend=:left)

plot(Normal(mean(inf_m), var(inf_m)))
plot!(real_m)

plot(x_obs, xlims=(1, 100))
plot!(y_lat)
plot!(mean.(y_inf), ribbon=sqrt.(var.(y_inf)))
