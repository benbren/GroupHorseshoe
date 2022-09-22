lisbrary(tidyverse)

rinvgamma <- function(n, a, b) {
  return(1 / rgamma(n, a, b))
}


group_horseshoe_gibs_nodes <- function(burn_ins,
                                    posterior_draws ,
                                    dat = NULL,
                                    p = 5,
                                    n = 1000,
                                    debug = F,
                                    update = "all",
                                    show = F,
                                    show_params = F,
                                    generate = T,
                                    sigma2_beta = 2,
                                    prop_zero = 0.8,
                                    beta_mean = 5,
                                    beta_sd = 0.01,
                                    x_cor = 0.5,
                                    x_sigma = 1,
                                    init_beta = 0,
                                    ungroup = F,
                                    beta_hubs = F,
                                    hub_nodes = NULL,
                                    hub_degrees = NULL) {
  if (!is.null(dat)) {
    generate <- F
  }
  
  if (generate) {
    dat <- generate_network_data(
      p = p,
      n = n,
      sigma2_beta = sigma2_beta,
      prop_zero = prop_zero,
      beta_mean = beta_mean,
      beta_sd = beta_sd,
      x_cor = x_cor,
      x_sigma = x_sigma,
      beta_hubs = beta_hubs,
      hub_nodes = hub_nodes,
      hub_degrees = hub_degrees
    )
  }
  if(all(dat$signal == 0)){
    error("No Signal")
  }
  
  n <- dat$n
  p <- dat$p
  
  
  q = p * (p - 1) / 2
  
  #system.time({
  gam <-
    data.frame(
      gam_value = rep(1, p),
      j = 1:p,
      a_gam = 1,
      k = 1:p
    )
  lambda <-
    data.frame(
      lam_value = rep(1, q),
      j = dat$signal$j,
      k = dat$signal$k,
      a_lam = 1
    )
  sigma2 <- 1
  betas <-
    data.frame(
      values  = rep(init_beta, q),
      j = dat$signal$j,
      k = dat$signal$k
    )
  if (debug) {
    betas$values = dat$signal$beta
  }
  tau2 <- 1
  iterations <- burn_ins + posterior_draws
  it <- 0
  beta_mat <- NULL
  sigma_vec <- NULL
  gam_j_vec <- NULL
  gam_k_vec <- NULL
  tau_vec <- NULL
  lam_vec <-
  X <- NULL
  for (i in 1:n) {
    X <- rbind(X, dat$ntworkslt[[i]])
  }
  
  
  XtX <- crossprod(X)  
  
  while (it < iterations + 1) {
    ### Updating Hyperparameters ####
    if ("atau" %in% update | update == "all") {
      # posterior of a_tau is invgamma(1,1 + 1/tau^2)
      a_tau <- rinvgamma(1, 1, 1 + 1 / tau2)
      if (show) {
        print(a_tau)
      }
      
    }
    
    if ("agam" %in% update | update == "all") {
      # if(ungroup){
      #   next
      # }
      
      for (node in 1:p) {
        # posterior of a_gamma_j is invgamma(1, 1 + 1/gamma_j^2)
        gam$a_gam[node] <- rinvgamma(1, 1, 1 + 1 / gam$gam_value[node])
        if (ungroup) {
          gam$a_gam[node] <- 1
        }
        
      }
    }
    
    if ("alam" %in% update | update == "all") {
      for (node_pair in 1:q) {
        # posterior of a_lam_jk is invgamma(1, 1 + 1/lambda_jk^2)
        lambda$a_lam[node_pair] <-
          rinvgamma(1, 1, 1 + 1 / lambda$lam_value[node_pair])
      }
    }
    
    gam_k <-
      gam %>% select(-j) %>% rename(gam_k_value = gam_value, a_k_gam = a_gam)
    gam_j <-
      gam %>% select(-k) %>% rename(gam_j_value = gam_value, a_j_gam = a_gam)
    
    
    params <- lambda %>%
      left_join(gam_k, b = "k") %>%
      left_join(gam_j, by = "j") %>%
      left_join(betas, by = c("j", "k")) %>%
      rowwise() %>%
      mutate(div_sum = values ^ 2 / (gam_j_value * gam_k_value * lam_value))
    
    if (show_params) {
      print(params)
    }
    
    
    if ("tau" %in% update | update == "all") {
      # posterior of tau^2 is invgamma( (q + 1) / 2, 1 / a_tau + 1/2sigma^2 * sum(beta_jk^2 / lam_jk ^2 gam_j ^2 gam_k ^2))
      tau2 <-
        rinvgamma(1, (q + 1) / 2, 1 / a_tau + sum(params$div_sum) / (2 * sigma2))
      if (show) {
        print(tau2)
      }
    }
    
    if ("gamma" %in% update | update == "all") {
        
      for (node in 1:p) {
        k_dat <- params %>% filter(k == node | j == node) %>%
          rowwise() %>%
          mutate(denom_piece  = ifelse(
            j == node,
            (values ^ 2) / (gam_k_value * lam_value) ,
            (values ^ 2) / (gam_j_value * lam_value)
          ))
        
        a_k <-
          gam %>% filter(k == node) %>% select(a_gam) %>% unlist()
        
        denom <- sum(k_dat$denom_piece)
        
        # posterior for gamma_k is invgamma(m_k + 1 / 2, 1/a_k + 1/2sigma^2tau^2)
        gam$gam_value[gam$k == node] <-
          rinvgamma(1, p / 2, (1 / a_k) + denom / (2 * sigma2 * tau2))
        
        if (ungroup) {
          gam$gam_value[gam$k == node] <- 1
        }
      } 
    }
 
    if ("lambda" %in% update | update == "all") {

      lambda$lam_value <- rinvgamma(q,rep(1,q), 1 / params$a_lam + params$values ^ 2 / (
        2 * (sigma2 * tau2 * params$gam_k_value * params$gam_j_value)))
    }
    
    
    gam_k <-
      gam %>% select(-j) %>% rename(gam_k_value = gam_value, a_k_gam = a_gam)
    gam_j <-
      gam %>% select(-k) %>% rename(gam_j_value = gam_value, a_j_gam = a_gam)
    
    params <- lambda %>%
      left_join(gam_k, b = "k") %>%
      left_join(gam_j, by = "j") %>%
      left_join(betas, by = c("j", "k"))
    
    
    # Updating Model Parameters #### 
    
    xb <- X %*% betas$values
    
    LAMBDA_diag <-
      params %>% rowwise() %>% mutate(diag_el = gam_k_value * gam_j_value * lam_value) %>% select(diag_el) %>% unlist()
    lam <- diag(1, q)
    diag(lam) <- LAMBDA_diag
    inv_lam <- solve(tau2 * lam, tol = 1e-26)
    
    
    err = dat$outcomes - xb
    if ("sigma" %in% update | update == "all") {
      sigma2 <-
        rinvgamma(1,
                  (n + q) / 2,
                  crossprod(err)/ 2 + t(betas$values) %*% (inv_lam) %*% betas$values / 2)
      if (show) {
        print(sigma2)
        print(err)
      }
    }
    
    A <- XtX + inv_lam   
    
    if ("betas" %in% update | update == "all") {
      R <- chol(A)
      #mu <- t(X)%*%dat$outcomes
      mu <- crossprod(X,dat$outcomes)
      #b <- solve(t(R))%*%mu
      b <- backsolve(R,mu,transpose=TRUE)
      z <- rnorm(q,0,sqrt(sigma2))

      betas$values <- backsolve(R,z + b)
      
    }
    print(paste("iteration:", it))
    
    if (it > burn_ins) {
      beta_mat <- cbind(beta_mat, betas$values)
      lam_vec <- cbind(lam_vec, params$lam_value)
      gam_j_vec <- cbind(gam_j_vec, params$gam_j_value)
      gam_k_vec <- cbind(gam_k_vec, params$gam_k_value)
      tau_vec <- c(tau_vec, tau2)
      sigma_vec <- c(sigma_vec, sigma2)
    }
    
    it <- it + 1
    
    
    
  }
  
  posterior_pe <- rowMeans(beta_mat)
  
  posterior_se <- apply(beta_mat, 1, sd)
  
  preds <- X %*% posterior_pe
  
  return(
    list(
      real_dat = dat,
      posterior_draws_beta = beta_mat,
      posterior_pe = posterior_pe,
      posterior_se = posterior_se,
      posterior_draws_lam = lam_vec,
      posterior_draws_gam_j = gam_j_vec,
      posterior_draws_gam_k = gam_k_vec,
      posterior_draws_tau = tau_vec,
      posterior_draws_sigma = sigma_vec,
      predictions = preds,
      burn_ins = burn_ins, 
      posterior_draws = posterior_draws
    )
  )
}


group_horseshoe_gibs_groups <- function(burn_ins,
                                 posterior_draws,
                                 dat,
                                 p = 5,
                                 n = 1000,
                                 debug = F,
                                 update = "all",
                                 show = F,
                                 show_params = F,
                                 init_beta = 0, 
                                 ungroup = F, 
                                 empirical = F) {
  
  library(tidyverse)

  
  n <- dat$n
  p <- dat$p
  
  
  q = p * (p - 1) / 2
  g <- max(dat$signal$group)
  
  #system.time({
  gam <-
    data.frame(
      gam_value = rep(1,g),
      group = 1:g,
      a_gam = 1
    )

  lambda <-
    data.frame(
      lam_value = rep(1, q),
      j = dat$signal$j,
      k = dat$signal$k,
      group = dat$signal$group,
      a_lam = 1
    )
  sigma2 <- 1
  betas <-
    data.frame(
      values  = rep(init_beta, q),
      j = dat$signal$j,
      k = dat$signal$k
    )
  if (debug) {
    betas$values = dat$signal$beta
  }
  if(empirical){
    tau2 = sum(dat$signal$signal == 1)/q
  } else {
    tau2 <- 1
  }
  
  iterations <- burn_ins + posterior_draws
  it <- 0
  beta_mat <- NULL
  sigma_vec <- NULL
  gam_vec <- NULL
  tau_vec <- NULL
  lam_vec <- NULL
  X <- NULL
  for (i in 1:n) {
    X <- rbind(X, dat$ntworkslt[[i]])
  }
  
  
  XtX <- crossprod(X)  
  
  while (it < iterations + 1) {
    ### Updating Hyperparameters ####
    if (("atau" %in% update | update == "all") & !empirical) {
      # posterior of a_tau is invgamma(1,1 + 1/tau^2)
      a_tau <- rinvgamma(1, 1, 1 + 1 / tau2)
      if (show) {
        print(a_tau)
      }
      
    }
    
    if ("agam" %in% update | update == "all") {
      # if(ungroup){
      #   next
      # }
      
      for (group in 1:g) {
        # posterior of a_gamma_j is invgamma(1, 1 + 1/gamma_j^2)
        gam$a_gam[group] <- rinvgamma(1, 1, 1 + 1 / gam$gam_value[group])
        if (ungroup) {
          gam$a_gam[group] <- 1
        }

      }
    }
    
    if ("alam" %in% update | update == "all") {
      for (node_pair in 1:q) {
        # posterior of a_lam_jk is invgamma(1, 1 + 1/lambda_jk^2)
        lambda$a_lam[node_pair] <-
          rinvgamma(1, 1, 1 + 1 / lambda$lam_value[node_pair])
      }
    }
    
    
    params <- lambda %>%
      left_join(gam, by = "group")%>%
      left_join(betas, by = c("j", "k")) %>%
      rowwise() %>%
      mutate(div_sum = values ^ 2 / (gam_value * lam_value))
    
    if (show_params) {
      print(params)
    }
    
    
    if (("tau" %in% update | update == "all") & !empirical) {
      # posterior of tau^2 is invgamma( (q + 1) / 2, 1 / a_tau + 1/2sigma^2 * sum(beta_jk^2 / lam_jk ^2 gam_j ^2 gam_k ^2))
      tau2 <-
        rinvgamma(1, (q + 1) / 2, 1 / a_tau + sum(params$div_sum) / (2 * sigma2))
      if (show) {
        print(tau2)
      }
    }
    
    if ("gamma" %in% update | update == "all") {
      
      for (i in 1:g) {
        k_dat <- params %>% filter(group == i) %>%
          rowwise() %>%
          mutate(denom_piece = (values ^ 2) / (lam_value))
        
        a_k <-
          gam %>% filter(group == i) %>% select(a_gam) %>% unlist()
        
        denom <- sum(k_dat$denom_piece)
        
        # posterior for gamma_k is invgamma(m_k + 1 / 2, 1/a_k + 1/2sigma^2tau^2)
        gam$gam_value[gam$group == i] <-
          rinvgamma(1,
                    (nrow(k_dat) + 1)/ 2, (1 / a_k) + denom / (2 * sigma2 * tau2))
        
        if (ungroup) {
          gam$gam_value[gam$group == i] <- 1
        }
      } 
    }
    
    if ("lambda" %in% update | update == "all") {
      
      lambda$lam_value <- rinvgamma(q,rep(1,q), 1 / params$a_lam + params$values ^ 2 / (
        2 * (sigma2 * tau2 * params$gam_value)))
    }
    
    

    
    params <- lambda %>%
      left_join(gam, b = "group") %>%
      left_join(betas, by = c("j", "k"))
    
    
    # Updating Model Parameters #### 
    
    xb <- X %*% betas$values
    
    LAMBDA_diag <-
      params %>% rowwise() %>% mutate(diag_el = gam_value * lam_value) %>% select(diag_el) %>% unlist()
    inv_lam <- diag(1, q)
    diag(inv_lam) <- 1/(tau2*LAMBDA_diag)

    
     
    err = dat$outcomes - xb
    if ("sigma" %in% update | update == "all") {
      sigma2 <-
        rinvgamma(1,
                  (n + q) / 2,
                  t(err) %*% err / 2 + (t(betas$values) %*% (inv_lam) %*% betas$values) / 2)
      if (show) {
        print(sigma2)
        print(mean(err))
      }
    }
    
    A <- XtX + inv_lam   
    
    if ("betas" %in% update | update == "all") {
      
      R <- chol(A) 
  
      mu <- crossprod(X,dat$outcomes)
  
      b <- backsolve(R,mu,transpose=TRUE)
      z <- rnorm(q,0,sqrt(sigma2))
      
      betas$values <- backsolve(R,z + b)
      
    }
    
    if (it > burn_ins) {
      beta_mat <- cbind(beta_mat, betas$values)
      lam_vec <- cbind(lam_vec, params$lam_value)
      gam_vec <- cbind(gam_vec, params$gam_value)
      tau_vec <- c(tau_vec, tau2)
      sigma_vec <- c(sigma_vec, sigma2)
    }

    
    it <- it + 1
    print(it)
    
    
    
  }
  
  posterior_pe <- rowMeans(beta_mat)
  
  posterior_se <- apply(beta_mat, 1, sd)
  
  preds <- X %*% posterior_pe
  
  K = mean(tau_vec)*diag(LAMBDA_diag) %*% solve(MASS::ginv(XtX) + mean(tau_vec)*diag(LAMBDA_diag))
  ev = eigen(K)
  
  return(
    list(
      real_dat = dat,
      posterior_draws_beta = beta_mat,
      posterior_pe = posterior_pe,
      posterior_se = posterior_se,
      posterior_draws_lam = lam_vec,
      posterior_draws_gam = gam_vec,
      posterior_draws_tau = tau_vec,
      posterior_draws_sigma = sigma_vec,
      predictions = preds,
      burn_ins = burn_ins, 
      posterior_draws = posterior_draws,
      params = params, 
      shrinkage_matrix = K, 
      eigenbasis = ev$values
    )
  )
}






