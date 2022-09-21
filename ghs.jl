# Function to perform the horseshoe Gibbs Sampler with grouped data (from a network)

using Distributions
using LinearAlgebra

function ghs(X, y, group, burnins, draws) 
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
    
    it = 1;

    while it < iterations

        # Update λ and a_λ #
        
        alam = [rand(InverseGamma(1, 1 + 1/lam[i]),1) for i in 1:q]; 
        alam = reduce(vcat,alam)

        lam = [rand(InverseGamma(1, 1 / alam[i] + (beta[i]^2)/(2*sigma2*tau2*gam[i])),1) for i in 1:q]; 
        lam = reduce(vcat,lam)
        
        # Update γ and a_γ # 
        k = 1 
        agam_next = []
        gam_next = []
        beta_over_lam = beta .^2 ./ lam
        for i in g_sizes  
           m = i + (k-1);
           beta_sum = sum(beta_over_lam[k:m]);
           append!(gam_next, repeat(rand(InverseGamma(0.5*(i + 1),1/agam[k] + beta_sum / (2*sigma2*tau2)),1),i));
           append!(agam_next, repeat(rand(InverseGamma(1, 1 + 1/gam_next[k]),1), i));
           k = k + i ;
        end 

        agam = reduce(vcat,agam_next)
        gam = reduce(vcat,gam_next)

        atau = reduce(vcat,rand(InverseGamma(1, 1 + 1/tau2),1)); 
        tau2 = rand(InverseGamma(0.5*(q+1), 1 / atau + 0.5*sum((beta.^2) ./(lam.*gam))/sigma2),1)
        tau2 = reduce(vcat, tau2)
       
       
        lamdiag = []

        for i in 1:q
            append!(lamdiag, (lam .* gam)[i]); 
        end 

        Λ = diagm(convert(Vector{Float64},lamdiag));

        A = transpose(X)*X + inv(Λ.*tau2);
        R = cholesky(A);
        b = inv(R)*transpose(X)*y;
        z = rand(Normal(0,sqrt(sigma2)),q);
        beta = inv(R)*(z + b);
        err = y - X*beta;

        sigma2 = reduce(vcat,rand(InverseGamma(0.5*(n + q), 0.5*(dot(err,err) + (transpose(beta)*inv(Λ.*tau2)*beta)[1])), 1));
        print(it);
        print("\n")
        it += 1 ;
    end 

    return beta
end



        


