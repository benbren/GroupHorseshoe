

using Distributions
using LinearAlgebra

function ghs(X, y, group, burnins, draws) 

"""
posterior_estimates = ghs(X, y, group, burnins, draws)

Function to perform the horseshoe Gibbs Sampler with grouped data (from a network)

    X = sorted design matrix (columns sorted according to "group")
    y = continuous outcome 
    group = sorted group membership (of the parameters) 
    burnins = obvious 
    draws = recorded draws from the posterior 

"""
    n,q = size(X);
    y = Matrix(y);
    X = Matrix(X);
    group = Matrix(group);
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
    it = 1 ;
    while it < iterations

        # Update λ and a_λ ###### 
        sigtau = sigma2 * tau2 
        #alam = [rand(InverseGamma(1, 1 + 1 / lam[i]),1) for i in 1:q]; 
        alam_scale = 1 .+ 1 ./ lam ; 
        alam = rand.(InverseGamma.(1, alam_scale)) ; 
        alam = reduce(vcat,alam) ; 

        #lam = [rand(InverseGamma(1, 1 / alam[i] + 0.5*(beta[i]^2)/(sigma2*tau2*gam[i])),1) for i in 1:q]; 
        lam_scale = (1 ./ alam) .+ (0.5 * (beta.^2) ./ (sigtau .* gam))
        lam =  rand.(InverseGamma.(1, lam_scale)) ;
        lam = reduce(vcat,lam)

        # γ updates ####
     
        k = 1 
        agam_next = []
        gam_next = []
        beta_over_lam = beta .^2 ./ lam
       for i in g_sizes  
           m = i + (k-1);
           beta_sum = sum(beta_over_lam[k:m])
           gam_scale = 1/agam[k] + 0.5*beta_sum / sigtau
           agam_scale = 1 + 1/gam[k]
           gam[k:m] .= reduce(vcat,rand(InverseGamma((i + 1)/2 , gam_scale)))
           agam[k:m] .= reduce(vcat,rand(InverseGamma(1, agam_scale)))
           k = k + i ;
       end 
       
        # Update τ  ##### 

        lamgam = lam .* gam
        divsum = beta.^2 ./ lamgam
        atau_scale = 1 + 1/tau2
        atau = reduce(vcat,rand(InverseGamma(1, atau_scale))); 

        tau_scale = 1 / atau + sum(divsum)/(2*sigma2) 
        tau2 = rand(InverseGamma((q+1) / 2 , tau_scale))
        tau2 = reduce(vcat, tau2)
       
       # Update β  ##### 

        Λ_inv = diagm(lamgam.^(-1)) * tau2^(-1)
        A = X'*X + Λ_inv;
        R = cholesky(A);
        mu = X'*y; 
        b = R.L \ mu;
        z = rand(Normal(0,sqrt(sigma2)),q);
        beta = R.U \ (z .+ b);

        # Update σ #####

        err = y - X*beta;

        sigmascale = dot(err,err)/2 + (beta'*Λ_inv*beta)[1]/2

        sigma2 = reduce(vcat,rand(InverseGamma(0.5*(n + q), sigmascale)));
        if it > burnins
            posteriors = vcat(posteriors,beta'); 
        end 


        if it % 25 == 0
            print("Iteration: $it");
            print("\n"); 
        end 
        it += 1;

    end 

    posterior_pe  = mean(posteriors, dims = 1)

    return posterior_pe
end




        

