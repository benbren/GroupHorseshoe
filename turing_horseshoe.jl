using Turing 
using LinearAlgebra
using FillArrays 

struct HP{T}
    X::T
end 

@model function(m::HP)(::Val{:Ungrouped})
    X = m.X 
    q = size(X,2)
    halfcauchy = truncated(Cauchy(0,1); lower = 0)
    τ ~ halfcauchy
    λ ~ filldist(halfcauchy, q)
    σ ~ Exponential(1)
    β ~ MvNormal(zeros(q), diagm(λ .* (τ*σ)).^2)

    y ~ MvNormal(X*β, σ^2 * I)

    return(; τ , λ, β, σ, y)
end 

@model function(m::HP)(::Val{Grouped}, group = NaN ) # TODO: How do I assign this group? 

    X = m.X 
    q = size(X,2)
    g = size(group)
    g_sizes = [sum(group .== i) for i in 1:maximum(group)]; 
    halfcauchy = truncated(Cauchy(0,1); lower = 0)
    τ ~ halfcauchy
    λ ~ filldist(halfcauchy, q)
    γ ~ filldist(halfcauchy, g)
    k = 1 
    lam = 1
    gam = 1
    for size in g_sizes  
        m = size + (k-1);
        for j in k:m
            β ~ (0,λ[lam]*γ[gam]*τ*σ)
            lam +=1 
        end 
        gma += 1 
        k += size 
    y ~ MvNormal(X*β, σ^2 * I)
end 


model = HP(X)(Val(:Ungrouped)); 

model_y = model | (; y)

samps = sample(model_y, NUTS(), MCMCThreads(), 2_000, 4)