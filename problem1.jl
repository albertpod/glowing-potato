using RxInfer
using LinearAlgebra
using Random, Distributions
using StatsPlots

# Function for data generation 
function generate_dataset(n, parameters)
    βdist = parameters[:β]
    αdist = parameters[:α]
    mdist = parameters[:m]

    β = mean(βdist)
    α = mean(αdist)
    m = mean(mdist)

    obs, lat = [], []
    for _ in 1:n
        #  or rand(mdist), rand(βdist) ...
        push!(lat, rand(MixtureModel(Normal[Normal(0.0, 1.0),Normal(1.0, 1.0)], [β, 1-β])))
        push!(obs, rand(MixtureModel(Normal[Normal(m, 1.0), Normal(lat[end], 1.0)], [α, 1-α])))
    end
    return αdist, βdist, mdist, lat, obs
end

n_samples = 1000

parameters = Dict(:m => Normal(10.0, 1.0), :α => Beta(2.0, 1.0), :β => Beta(1.0, 1.0)) # distributions for data generation
real_α, real_β, real_m, y_lat, x_obs = generate_dataset(n_samples, parameters)


@model function model1(n, priors)
    y = randomvar(n) # latent variables
    x = datavar(Float64, n) # observations
    β ~ Beta(priors[:α].α, priors[:α].β) # prior for the selector variable of y mixture
    α ~ Beta(priors[:β].α, priors[:β].β) # prior for the selector variable of x mixture

    # selector variable (mixture responsiblity)
    zβ = randomvar(n) 
    zα = randomvar(n)
    
    m ~ Normal(m=mean(priors[:m]), v=var(priors[:m]))

    for i in 1:n
        zβ[i] ~ Bernoulli(β) # selector is a binary variable, hence Bernoulli
        zα[i] ~ Bernoulli(α)
        y[i] ~ NormalMixture(zβ[i], (0.0, 1.0), (1.0, 1.0)) # order: selector, tuple of means, tuple of variances
        x[i] ~ NormalMixture(zα[i], (m, y[i]), (1.0, 1.0))
    end

end

# we initialize priors
priors = Dict(:β => vague(Beta), :α => vague(Beta), :m => Normal(0.0, 1e2))
data  = (x = x_obs,)

# initial marginal distributions due to mean-filed assumption
# To recover β, we need a somewhat good initial marginal
initmarginals = (
    β  = Beta(4.0, 1.0), 
    α  = vague(Beta), 
    y = NormalMeanVariance(0.0, 1e2),
    m = NormalMeanVariance(0.0, 1e2),
)

result = inference(
    model = model1(length(x_obs), priors), 
    data  = data,
    constraints = MeanField(),
    initmarginals = initmarginals, 
    iterations  = 10, 
    free_energy = true,
    showprogress = true,
)

# Retrieve posteriors
# switches
za = @. mean(result.posteriors[:zα][end])
zb = @. mean(result.posteriors[:zβ][end])

inf_α = result.posteriors[:α][end]
inf_β = result.posteriors[:β][end]
inf_m = result.posteriors[:m][end]

y_inf = result.posteriors[:y][end]


plot(inf_α, label="infered mean $(mean(inf_α))", legend=:left, fillalpha=0.3, fillrange = 0)
vline!([mean(real_α)], label="real mean $(mean(real_α))", legend=:left)

plot(inf_β, label="infered mean $(mean(inf_β))", legend=:right, fillalpha=0.3, fillrange = 0)
vline!([mean(real_β)], label="real mean $(mean(real_β))", legend=:left)

plot(Normal(mean(inf_m), var(inf_m)), label="infered mean $(mean(inf_m))", fillalpha=0.3, fillrange = 0)
vline!([mean(real_m)], label="real mean $(mean(real_m))", legend=:top)

scatter(x_obs, xlims=(1, 100), label="observations")
plot!(y_lat, label="latent state y")
plot!(mean.(y_inf), ribbon=sqrt.(var.(y_inf)), label="inferred")

plot(result.free_energy, xlabel="iteration", ylabel="free energy", legend=false)
