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
    atua = 1 ;
    sigma2 = 1; 
    iterations = burnins + draws ;
    it = 1;
    while it < iterations
        alam = [rand(InverseGamma(1, 1 + 1/lam[i][1]),1) for i in 1:q]; 
        lam = [rand(InverseGamma(1, 1 / alam[i][1] + beta[i]^2/(2*sigma2[1]*tau2[1]*gam[i][1])),1) for i in 1:q]; 
        g_sizes = [sum(group .== i) for i in 1:maximum(group)]; 
        k = 1 
        agam_next = []
        gam_next = []
        for i in g_sizes  
           m = i + (k-1);
           beta_sum = sum(beta[k:m].^2 ./ lam[k:m]);
           append!(gam_next, repeat(rand(InverseGamma(0.5*(i + 1),1/agam[k][1] + beta_sum[1] / (2*sigma2[1]*tau2[1])),1),i));
           append!(agam_next, repeat(rand(InverseGamma(1, 1 + 1/gam[k][1]),1), i));
           k = k + i ;
        end 
        agam = agam_next 
        gam = gam_next
        atau = rand(InverseGamma(1, 1 + 1/tau2[1]),1); 
        tau2 = rand(InverseGamma(0.5*(q+1), 1 / atau[1] + sum((beta.^2) ./ 2*sigma2[1].*(lam.*gam))[1]),1)
        lamdiag = []
        for i in 1:q
            append!(lamdiag, (lam .* gam)[i]); 
        end 
        Λ = diagm(convert(Vector{Float64},lamdiag));

        A = transpose(X)*X + inv(Λ.*tau2);
        R = cholesky(A);
        b = inv(R)*transpose(X)*y;
        z = rand(Normal(0,1),q);
        beta = inv(R)*(z + b);
        err = y - X*beta;

        sigma2 = rand(InverseGamma(0.5*(n + q), 0.5*(dot(err,err) + (transpose(beta)*inv(Λ.*tau2)*beta)[1])), 1);
        print(it);
        print("\n")
        it += 1 ;
    end 

    return beta
end



        


