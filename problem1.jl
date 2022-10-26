using Rocket, ReactiveMP, GraphPPL
using LinearAlgebra
using Random, Distributions

include("helpers.jl")

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

n_samples = 1000

gen_priors = Dict(:m => Normal(3.0, 1.0), :α => Beta(10.0, 1.0), :β => Beta(2.0, 1.0)) # distributions for data generation
real_α, real_β, real_m, y_lat, x_obs = generate_dataset(n_samples, gen_priors)


@model [ default_factorisation = MeanField() ] function model1(n)
    y = randomvar(n)
    x = datavar(Float64, n)
    β ~ Beta(1.0, 1.0)
    α ~ Beta(1.0, 1.0)

    zβ = randomvar(n)
    zα = randomvar(n)
    
    m ~ NormalMeanVariance(0.0, 1e2)

    for i in 1:n
        zβ[i] ~ Bernoulli(β)
        zα[i] ~ Bernoulli(α)
        y[i] ~ NormalMixture(zβ[i], (0.0, 1.0), (1.0, 1.0))
        x[i] ~ NormalMixture(zα[i], (m, y[i]), (1.0, 1.0))
    end

end

model = Model(model1, length(x_obs))
data  = (x = x_obs,)

# initial marginal distributions
initmarginals = (
    β  = Beta(3.0, 1.0), 
    α  = vague(Beta), 
    y = NormalMeanVariance(0.0, 1e2),
    m = NormalMeanVariance(0.0, 1e2),
)


result = inference(
    model = model, 
    data  = data, 
    initmarginals = initmarginals, 
    iterations  = 50, 
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
vline!([mean(real_α)], label="real mean $(mean(real_α))", legend=:left)

plot(inf_β, label="infered mean $(mean(inf_β))", legend=:right)
vline!([mean(real_β)], label="real mean $(mean(real_β))", legend=:left)

plot(Normal(mean(inf_m), var(inf_m)), label="infered mean $(mean(inf_m))")
vline!([mean(real_m)], label="real mean $(mean(real_m))", legend=:top)

plot(x_obs, xlims=(1, 100))
plot!(y_lat)
plot!(mean.(y_inf), ribbon=sqrt.(var.(y_inf)))

plot(result.free_energy, xlabel="iteration", ylabel="free energy")