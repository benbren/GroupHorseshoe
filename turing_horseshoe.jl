using Turing 
using LinearAlgebra
using FillArrays 
using ReverseDiff
using Memoization


struct HP{T}
    X::T
end 

Turing.setadbackend(:reversediff)
Turing.setrdcache(true)



@model function(m::HP)(::Val{:Ungrouped})

    X = Matrix(m.X) 
    q = size(X,2)

    # Latent parameters 
    v ~ filldist(InverseGamma(0.5, 1), q) 
    eps ~ InverseGamma(0.5,1)

    # Global and local shrinkage parameters 
    τ2 ~ InverseGamma(0.5, eps^(-1))
    λ2 = Vector{Float64}(undef, q)

    for i in 1:q 
        λ2[i] ~ InverseGamma(0.5, v[i]^(-1))
    end   

    # Random Noise 
    σ ~ Exponential(1) # Good ole jeffs prior on σ

    # Regression parameters 
    β ~ MvNormal(zeros(q), Diagonal(λ2 .* (τ2*(σ^2))))

    # Modeling
    y ~ MvNormal(X*β, σ^2 * I)

    return(; τ2, λ2, β, σ, y)

end 

@model function(m::HP)(::Val{:Grouped}, group = NaN) # TODO: How do I assign this group? 
        # TODO: Update to conditional InverseGamma model 

    X = m.X 
    q = size(X,2)
    X = Matrix(X)
    group = Matrix(group)
    g = size(group)[1]
    g_sizes = [sum(group .== i) for i in 1:maximum(group)]; 
    halfcauchy = truncated(Cauchy(0,1); lower = 0)
    τ ~ halfcauchy
    λ ~ filldist(halfcauchy, q)
    γ ~ filldist(halfcauchy, g)
    σ ~ Exponential(1)
    β  = Vector{Float64}(undef, q)
    k = 1 
    lam = 1
    gam = 1
    for size in g_sizes  
        m = size + (k-1);
        for j in k:m
            β[lam] ~ Normal(0,(λ[lam]*γ[gam]*τ*σ)^2)
            lam +=1 
        end 
        gam += 1 
        k += size 
    end
    y ~ MvNormal(X*β, σ^2 * I)
end 


model_un = HP(X)(Val(:Ungrouped)); 

model_y_un = model_un| (; y)

samps_un = sample(model_un, NUTS(), MCMCThreads(), 2_000, 2)

model = HP(X)(Val(:Grouped), g); 

model_y = model | (; y)

samps = sample(model_y, NUTS(), MCMCThreads(), 2_000, 2)