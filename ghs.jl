# Function to perform the horseshoe Gibbs Sampler with grouped data (from a network)

using Distributions
using LinearAlgebra

function ghs(X, y, group, burnins, draws) 

"""
ghs(X, y, group, burnins, draws)

TBW
"""
    n,q = size(X);
    y = Matrix(y)
    X = Matrix(X)
    group = Matrix(group)
    g = size(group)[1];
    lam = Vector{Float64}(ones(q));
    alam = Vector{Float64}(ones(q));
    gam = Vector{Float64}(ones(q));
    agam = Vector{Float64}(ones(q));
    beta = Vector{Float64}(zeros(q) .+ 5);
    tau2 = 1 ;
    atau = 1 ;
    sigma2 = 1; 
    iterations = burnins + draws ;
    g_sizes = [sum(group .== i) for i in 1:maximum(group)]; 
    posteriors = zeros(q)'; 
    it = 1;

    while it < iterations

        # Update λ and a_λ #

        alam = [rand(InverseGamma(1, 1 + 1/lam[i]),1) for i in 1:q]; 

        alam = reduce(vcat,alam)

        lam = [rand(InverseGamma(1, 1 / alam[i] + (beta[i]^2)/(2*sigma2*tau2*gam[i])),1) for i in 1:q]; 
        lam = reduce(vcat,lam)

        # γ updates 
     
        k = 1 
        agam_next = []
        gam_next = []
        beta_over_lam = beta .^2 ./ lam
       for i in g_sizes  
           m = i + (k-1);
           beta_sum = sum(beta_over_lam[k:m]);
           append!(gam_next, repeat(rand(InverseGamma((i + 1)/2 , 1/agam[k] + beta_sum / (2*sigma2*tau2)),1),i));
           append!(agam_next, repeat(rand(InverseGamma(1, 1 + 1/gam[k]),1), i));
          k = k + i ;
       end 

        agam = reduce(vcat,agam_next)
        gam = reduce(vcat,gam_next)
        lamgam = (lam .* gam)
        divsum = beta.^2 ./ lamgam
        atau = reduce(vcat,rand(InverseGamma(1, 1 + 1 / tau2),1)); 
        tau2 = rand(InverseGamma((q+1) / 2 , 1 / atau + sum(divsum)/(2*sigma2)),1)
        tau2 = reduce(vcat, tau2)
       
    
        Λ_inv = diagm(lamgam) / tau2
        A = transpose(X)*X + Λ_inv;
        R = cholesky(A);
        mu = transpose(X)*y; 
        b = R.L \ mu;
        z = rand(Normal(0,sqrt(sigma2)),q);
        beta = R \ (z + b);


        err = y - X*beta;

        sigma2 = reduce(vcat,rand(InverseGamma(0.5*(n + q), 0.5*(dot(err,err) + (transpose(beta)*Λ_inv*beta)[1])), 1));
        if it > burnins
            posteriors = vcat(posteriors,beta'); 
        end 


        if it % 500 == 0
            print("Iteration: $it");
            print("\n") 
        end 
        it += 1 ;

    end 

    posterior_pe  = [mean(posteriors[:,i]) for i in 1:q]

    return posterior_pe
end



        

