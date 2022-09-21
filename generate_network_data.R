generate_network_data <- function(p, 
                                  n,
                                  r2,
                                  prop_zero = 0.8,
                                  beta_mean = 5,
                                  beta_sd = 0.01,
                                  x_cor = 0.5,
                                  x_sigma = 1,
                                  beta_hubs = F,
                                  hub_nodes = NULL,
                                  hub_degrees = NULL,
                                  groups = F,
                                  num_groups = NULL,
                                  group_sizes = NULL,
                                  num_signals = NULL){
  
  #profvis({
  ## Create a random Adjacency Matrix that represents the network 
  q = p*(p-1)/2
  node_specific_shrinkage <- T
  if(!beta_hubs & !groups){
    adjm <- matrix(NA,p,p)
    true_signals <- matrix(sample(0:1,q , replace = T, prob = c(prop_zero,1-prop_zero)))
    adjm[lower.tri(adjm)] <- true_signals
    adjm[upper.tri(adjm)] <- t(adjm)[upper.tri(adjm)]
    diag(adjm) <- 0
    
    ntwrk <- igraph::graph_from_adjacency_matrix(adjm,
                                                 mode= "undirected")
  } else if (beta_hubs){
    if(is.null(hub_nodes | is.null(hub_degrees))){
      stop("Please specify nodes AND degress for the hubs")
    }
    hub_connects <- list()
    if(p - length(hub_nodes) - sum(hub_degrees) < 2){
      stop("Pick a larger p - your hub and connections are greater than the number of nodes ")
    }
    num_nodes <- p
    adjm <- matrix(0,nrow=num_nodes, ncol=num_nodes)
    for(i in 1:length(hub_nodes)){
      if(i == 1)
        hub_connects[[i]] <- length(hub_nodes)+1:hub_degrees[1]
      else
        hub_connects[[i]] <- max(hub_connects[[i-1]]) + 1:hub_degrees[i]
      adjm[hub_nodes[i],hub_connects[[i]]] <- 1
    }
    adjm <- t(adjm) + adjm
    
    true_signals <-  adjm[lower.tri(adjm)]
    diag(adjm) <- 0 
    
    ntwrk <- igraph::graph_from_adjacency_matrix(adjm,
                                                 mode="undirected")
  } else {
    if(num_groups != length(group_sizes)){
      stop("Need to enter group size for each group")
    }
    if(sum(group_sizes) != p){
      stop("Sum of group sizes not equal to the size of the network")
    }
    for(j in 1:length(group_sizes)){
      if(num_signals[j] > choose(group_sizes[j],2)){
        stop(paste("Too many signals in group", j))
      }
    }
    
    group_membership <- NULL
    
    for (i in 1:num_groups){
      group_membership <- c(group_membership, rep(i, group_sizes[i]))
  }
    group_membership_dat <- cbind.data.frame(j = 1:p, k = 1:p, group_membership)
    
    i <- 2
    j <- NULL # rows 
    k <- NULL # columns 
    
    while (i < p+1){
      addonj <- i:p
      addonk <- rep(i-1,p+1-i) 
      j <- c(j,addonj)
      k <- c(k,addonk)
      i <- i + 1
    }
    jk <- data.frame(j = j, k = k)
    
    group_membership_idx <- group_membership_dat %>% select(-k) %>% right_join(jk, by = "j") %>% 
      rename(j_group = group_membership) %>% rename(j_node = j) %>% left_join(group_membership_dat, by = "k") %>% 
      rename(k_group = group_membership) %>% select(-j) %>% rename(j = j_node)%>% mutate(possible_signal = ifelse(k_group == j_group, T,F)) 
    
    signals <- NULL
    for (i in 1:num_groups){
      tmp <- group_membership_idx %>% filter(possible_signal & k_group == i)
      sigs_in_grp <- num_signals[i]
      non_permuted_signals <- c(rep(1,sigs_in_grp), rep(0, choose(group_sizes[i],2) - sigs_in_grp))
      permuted_signals <- sample(non_permuted_signals)
      dat <- data.frame(j = tmp$j, k = tmp$k, signal = permuted_signals)
      signals <- rbind(dat,signals)
    }
    
    group_membership_final <- signals %>% right_join(group_membership_idx, by = c("j","k")) %>% 
      mutate(signal = ifelse(is.na(signal), 0, signal)) %>% arrange(k)
    adjm <- matrix(NA,p,p)
    true_signals <- group_membership_final$signal
    adjm[lower.tri(adjm)] <- true_signals
    adjm[upper.tri(adjm)] <- t(adjm)[upper.tri(adjm)]
    diag(adjm) <- 0
    ntwrk <- igraph::graph_from_adjacency_matrix(adjm,
                                                 mode= "undirected")
    node_specific_shrinkage <- F
    
  }
  
  i <- 2
  j <- NULL # rows 
  k <- NULL # columns 
  
  while (i < p+1){
    addonj <- i:p
    addonk <- rep(i-1,p+1-i) 
    j <- c(j,addonj)
    k <- c(k,addonk)
    i <- i + 1
  }
  
  beta_with_indices = data.frame(signal = true_signals, j = j, k = k)
  beta_with_indices$beta <- beta_with_indices$signal * rnorm(dim(beta_with_indices)[1],beta_mean, beta_sd)
  # 


  #cov_mat <- diag(q)*x_sigma
  cov_mat <- toeplitz(c(x_sigma, rep(x_cor,q-1)))
  #cov_mat[upper.tri(cov_mat)] <- x_cor
  #cov_mat[lower.tri(cov_mat)] <- x_cor
  observed_networks <- list()
  observed_networkslt <- list()
  
  y <- NULL
  x <- mvnfast::rmvn(n,rep(0,q), cov_mat, ncores = 2)
  for (i in 1:n){
    x_i <- x[i,]
    ob_ntwrk <- matrix(NA,p,p)
    ob_ntwrk[lower.tri(ob_ntwrk)] <- x_i
    ob_ntwrk[upper.tri(ob_ntwrk)] <- t(ob_ntwrk)[upper.tri(ob_ntwrk)]
    diag(ob_ntwrk) <- 0
    observed_networkslt[[i]] <- ob_ntwrk[lower.tri(ob_ntwrk)] # make sure this matches indices!! 
    observed_networks[[i]] <- ob_ntwrk # make sure this matches indices!! 
  }
  
  X <- NULL   
  for (i in 1:n) {
    X <- rbind(X, observed_networkslt[[i]])
  }
  xb <- X %*% beta_with_indices$beta
 
  xb = xb + rnorm(n,0,0.001)

  vxb <- var(X %*% beta_with_indices$beta)
  noise <- ((1-r2) / r2)*vxb
  
  y <- xb + rnorm(n,0,sqrt(noise))
  
  if(groups){
    beta_with_indices$k_group = group_membership_final$k_group
    beta_with_indices$j_group = group_membership_final$j_group
    beta_with_indices$group = ifelse(group_membership_final$k_group == group_membership_final$j_group,
                                     group_membership_final$j_group, num_groups +1)
  } else{
    beta_with_indices$k_group = k
    beta_with_indices$j_group = j
  }
  
  X <- t(matrix(unlist(observed_networkslt), choose(p,2), n))
  sum_gs <- rep(NA, length(group_sizes)+1)
  sum_gs[1] <- 0 
  for(i in 2:length(sum_gs)){
    sum_gs[i] <- choose(group_sizes[i-1],2) + sum_gs[i-1]
  }
  stan_dat <- list(N = n, 
                        Q = choose(p, 2),
                        G = length(unique(beta_with_indices$group)), 
                        group_sizes = c(choose(group_sizes,2), choose(p,2) - sum(choose(group_sizes,2))),
                        sum_gs = sum_gs,
                        y = as.vector(y), 
                        X = X[,order(beta_with_indices$group)])
  
  
  rtrn <- list(outcomes = y, X = X, signal = beta_with_indices, ntworkslt = observed_networkslt,
               p = p, n = n, full_ntwork = ntwrk, adjm = adjm, noise = noise,
               shrinking_nodes = node_specific_shrinkage, stan_dat = stan_dat)
  #})
  return(rtrn)
  
}

  

