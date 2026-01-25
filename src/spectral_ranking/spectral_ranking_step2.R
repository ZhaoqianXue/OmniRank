#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  requireNamespace("readr", quietly = TRUE)
  requireNamespace("dplyr", quietly = TRUE)
  requireNamespace("jsonlite", quietly = TRUE)
})

# ==============================================================================
# Helper functions adapted from ranking_cli.R to ensure consistency
# ==============================================================================

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  kv <- list()
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (startsWith(key, "--")) {
      if (i + 1 <= length(args)) {
        kv[[substring(key, 3)]] <- args[[i + 1]]
        i <- i + 2
      } else {
        stop("Missing value for ", key)
      }
    } else {
      i <- i + 1
    }
  }
  required <- c("csv", "json_step1", "out")
  # B, seed, bigbetter can be inferred from json_step1 if present, or overridden
  
  for (r in required) {
    if (is.null(kv[[r]])) stop(sprintf("Missing required argument: --%s", r))
  }
  
  # Optional: B, seed, bigbetter (defaults taken from JSON if missing)
  kv$B <- if (!is.null(kv$B)) as.integer(kv$B) else NULL
  kv$seed <- if (!is.null(kv$seed)) as.integer(kv$seed) else NULL
  kv$bigbetter <- if (!is.null(kv$bigbetter)) as.integer(kv$bigbetter) else NULL
  
  kv
}

safe_dir_create <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

process_data <- function(data, bigbetter = FALSE) {
  Idx <- colnames(data)
  numidx <- length(Idx)
  xx <- matrix(0, 0, numidx)
  ww <- matrix(0, 0, numidx)

  for (ii in 1:nrow(data)) {
    target_row <- data[ii, ]
    pairs <- t(combn(seq_along(target_row), 2))
    valid_idx <- !is.na(target_row[pairs[, 1]]) & !is.na(target_row[pairs[, 2]])
    if (!any(valid_idx)) next
    pairs <- pairs[valid_idx, , drop = FALSE]

    if (bigbetter) {
      v1 <- ifelse(target_row[pairs[, 1]] > target_row[pairs[, 2]], Idx[pairs[, 2]], Idx[pairs[, 1]])
      v2 <- ifelse(target_row[pairs[, 1]] > target_row[pairs[, 2]], Idx[pairs[, 1]], Idx[pairs[, 2]])
    } else {
      v1 <- ifelse(target_row[pairs[, 1]] > target_row[pairs[, 2]], Idx[pairs[, 1]], Idx[pairs[, 2]])
      v2 <- ifelse(target_row[pairs[, 1]] > target_row[pairs[, 2]], Idx[pairs[, 2]], Idx[pairs[, 1]])
    }

    tmp.xx <- matrix(0, nrow = length(v1), ncol = numidx)
    tmp.ww <- matrix(0, nrow = length(v1), ncol = numidx)
    for (jj in seq_along(v1)) {
      tmp.xx[jj, Idx == v1[jj]] <- 1
      tmp.xx[jj, Idx == v2[jj]] <- 1
      tmp.ww[jj, Idx == v2[jj]] <- 1
    }
    xx <- rbind(xx, tmp.xx)
    ww <- rbind(ww, tmp.ww)
  }

  yy <- matrix(c(xx), ncol = numidx)
  zz <- matrix(c(ww), ncol = numidx)
  list(aa = as.matrix(yy), ww = as.matrix(zz), idx = Idx)
}

solve_spectral <- function(AA, WW, weights, n, dval) {
  P <- matrix(0, n, n)
  for (i in 1:n) {
    for (j in 1:n) {
      if (j != i) {
        P[i, j] <- sum(AA[, i] * AA[, j] * WW[, j] / weights) / dval
      }
    }
    P[i, i] <- 1 - sum(P[i, ])
  }

  tmp.P <- t(t(P) - diag(n)) %*% (t(P) - diag(n))
  tmp.svd <- svd(tmp.P)
  pihat <- abs(tmp.svd$v[, n])
  pihat <- pmax(pihat, .Machine$double.eps)
  
  log_pihat <- log(pihat)
  thetahat <- log_pihat - mean(log_pihat, na.rm = TRUE)
  
  list(theta = thetahat, pi = pihat, P = P)
}

# ==============================================================================
# Step 2 Only Logic
# ==============================================================================

run_step2_only <- function(AA, WW, Idx, theta1_named, B = 2000) {
  n <- ncol(AA)
  L <- nrow(AA)
  dval <- 2 * max(colSums(AA))
  
  # Align theta1 with Idx
  # theta1_named should be a named vector where names match Idx
  # verify alignment
  if (!all(Idx %in% names(theta1_named))) {
    stop("Mismatch between CSV columns and Step 1 JSON method names")
  }
  theta1 <- theta1_named[Idx]
  
  # --------------------------------------------------------------------------
  # Step 2: Weighted Spectral Method (Optimal Weights)
  # Uses f(A_l) = sum_{u in A_l} exp(theta_u)
  # --------------------------------------------------------------------------
  exp_theta_mat <- t(matrix(exp(theta1), n, L)) 
  weights_step2 <- rowSums(AA * exp_theta_mat)
  
  res2 <- solve_spectral(AA, WW, weights_step2, n, dval)
  theta2 <- res2$theta
  pi2 <- res2$pi
  
  # --------------------------------------------------------------------------
  # Inference
  # --------------------------------------------------------------------------
  RR <- matrix(0, 6, n)
  colnames(RR) <- Idx
  RR[1, ] <- theta2
  RR[2, ] <- n + 1 - rank(theta2)
  
  # Variance estimation
  Vmatrix <- matrix(0, L, n)
  tauhatvec <- numeric(n)
  
  tmp.pimatrix <- t(AA) * pi2
  tmp.pivec <- colSums(tmp.pimatrix)
  tmp.var <- numeric(n)
  
  for (oo in 1:n) {
    tauhatvec[oo] <- sum(AA[, oo] * (1 - pi2[oo] / tmp.pivec) * pi2[oo] / weights_step2, na.rm = TRUE) / dval
    tmp.var[oo] <- sum(AA[, oo] * (tmp.pivec - pi2[oo]) / weights_step2^2) * pi2[oo] / dval^2 / tauhatvec[oo]^2
    Vmatrix[, oo] <- (AA[, oo] * WW[, oo] * tmp.pivec - AA[, oo] * pi2[oo]) / weights_step2
  }
  sigmahatmatrix <- matrix(tmp.var, n, n) + t(matrix(tmp.var, n, n))
  
  # Bootstrap
  Wmatrix <- matrix(rnorm(L * B), L, B)
  tmp.Vtau <- (t(Vmatrix) / tauhatvec) %*% Wmatrix
  
  R.left.m <- numeric(n)
  R.right.m <- numeric(n)
  R.left.one.m <- numeric(n)
  
  for (ooo in 1:n) {
    tmpGMmatrix0 <- matrix(rep(tmp.Vtau[ooo, ], n) - c(t(tmp.Vtau)), B, n)
    tmpGMmatrix <- abs(t(t(tmpGMmatrix0) / sqrt(sigmahatmatrix[ooo, ])) / dval)
    tmpGMmatrixone <- t(t(tmpGMmatrix0) / sqrt(sigmahatmatrix[ooo, ])) / dval
    
    tmp.GMvecmax <- apply(tmpGMmatrix, 1, max)
    tmp.GMvecmaxone <- apply(tmpGMmatrixone, 1, max)
    
    cutval <- quantile(tmp.GMvecmax, 0.95)
    cutvalone <- quantile(tmp.GMvecmaxone, 0.95)
    
    tmp.theta.sd <- sqrt(sigmahatmatrix[ooo, ])
    tmp.theta.sd <- tmp.theta.sd[-ooo]
    
    diffs <- (theta2[-ooo] - theta2[ooo]) / tmp.theta.sd
    
    R.left.m[ooo] <- 1 + sum(1 * (diffs > cutval))
    R.right.m[ooo] <- n - sum(1 * (diffs < -cutval))
    R.left.one.m[ooo] <- 1 + sum(1 * (diffs > cutvalone))
  }
  
  # Uniform CI logic
  WmatrixB <- matrix(rnorm(L * B), L, B)
  tmp.VtauB <- (t(Vmatrix) / tauhatvec) %*% WmatrixB
  
  GMvecmaxone_list <- numeric(0)
  for (ooo in 1:n) {
     tmpGMmatrix0 <- matrix(rep(tmp.VtauB[ooo, ], n) - c(t(tmp.VtauB)), B, n)
     tmpGMmatrixone <- t(t(tmpGMmatrix0) / sqrt(sigmahatmatrix[ooo, ])) / dval
     GMvecmaxone_list <- c(GMvecmaxone_list, apply(tmpGMmatrixone, 1, max))
  }
  GMmaxmatrixone <- matrix(GMvecmaxone_list, B)
  GMmaxone <- apply(GMmaxmatrixone, 1, max)
  cutvalone_uniform <- quantile(GMmaxone, 0.95)
  
  R.left.one_uniform <- numeric(n)
  for (ooo in 1:n) {
    tmp.theta.sd <- sqrt(sigmahatmatrix[ooo, ])
    tmp.theta.sd <- tmp.theta.sd[-ooo]
    diffs <- (theta2[-ooo] - theta2[ooo]) / tmp.theta.sd
    R.left.one_uniform[ooo] <- 1 + sum(1 * (diffs > cutvalone_uniform))
  }
  
  RR[3, ] <- R.left.m
  RR[4, ] <- R.right.m
  RR[5, ] <- R.left.one.m
  RR[6, ] <- R.left.one_uniform
  
  rownames(RR) <- c("theta.hat", "Ranking", "two-sided CI", "two-sided CI", "left-sided CI", "uniform left-sided CI")
  
  list(RR = RR, theta2 = theta2)
}

main <- function() {
  start_time <- Sys.time()
  args <- parse_args()
  
  csv_path <- args$csv
  json_path <- args$json_step1
  out_dir <- args$out
  
  safe_dir_create(out_dir)
  
  # Load JSON Step 1 Results
  step1_data <- tryCatch({
    jsonlite::read_json(json_path)
  }, error = function(e) {
    stop("Failed to read JSON Step 1 file: ", e$message)
  })
  
  # Extract params if not overridden in args
  bigbetter_flag <- if (!is.null(args$bigbetter)) (as.integer(args$bigbetter) == 1) else step1_data$params$bigbetter
  B <- if (!is.null(args$B)) as.integer(args$B) else step1_data$params$B
  seed <- if (!is.null(args$seed)) as.integer(args$seed) else step1_data$params$seed
  
  set.seed(seed)
  
  # Extract Theta1
  # step1_data$methods is a list of objects {name, theta_hat, ...}
  methods_list <- step1_data$methods
  theta1_named <- numeric(length(methods_list))
  names(theta1_named) <- sapply(methods_list, function(x) x$name)
  for (i in seq_along(methods_list)) {
    theta1_named[i] <- methods_list[[i]]$theta_hat
  }
  
  # Read CSV
  df <- tryCatch({
    readr::read_csv(csv_path, show_col_types = FALSE)
  }, error = function(e) {
    message("Falling back to base::read.csv: ", e$message)
    utils::read.csv(csv_path, stringsAsFactors = FALSE, check.names = TRUE)
  })
  
  # Process CSV exactly like Step 1
  if (requireNamespace("dplyr", quietly = TRUE)) {
    df <- dplyr::select(df, -dplyr::any_of(c("case_num", "model", "description")))
    df <- dplyr::select(df, where(is.numeric))
  } else {
    keep <- vapply(df, is.numeric, logical(1))
    df <- df[, keep, drop = FALSE]
  }
  
  if (ncol(df) < 2) stop("At least two numeric method columns are required")
  
  pdata <- process_data(df, bigbetter = bigbetter_flag)
  
  # Execute Step 2
  res_obj <- run_step2_only(pdata$aa, pdata$ww, pdata$idx, theta1_named, B = B)
  RR2 <- res_obj$RR
  
  # Prepare Output
  methods <- colnames(RR2)
  theta_hat <- as.numeric(RR2[1, ])
  rank <- as.numeric(RR2[2, ])
  ci_left_two <- as.numeric(RR2[3, ])
  ci_right_two <- as.numeric(RR2[4, ])
  ci_left <- as.numeric(RR2[5, ])
  ci_uniform_left <- as.numeric(RR2[6, ])

  results_df <- data.frame(
    method = methods,
    theta_hat = as.numeric(theta_hat),
    rank = as.integer(rank),
    ci_two_left = as.integer(ci_left_two),
    ci_two_right = as.integer(ci_right_two),
    ci_left = as.integer(ci_left),
    ci_uniform_left = as.integer(ci_uniform_left),
    stringsAsFactors = FALSE
  )
  
  # Convergence: Comparison with Step 1 theta would need re-aligning theta1_named to methods order
  theta1_aligned <- theta1_named[methods]
  diff_norm <- sqrt(sum((theta1_aligned - theta_hat)^2))
  
  runtime_sec <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  
  payload <- list(
    job_id = basename(dirname(out_dir)),
    params = list(bigbetter = bigbetter_flag, B = B, seed = seed, method = "step2_spectral_refined"),
    methods = lapply(seq_len(nrow(results_df)), function(i) {
      list(
        name = results_df$method[i],
        theta_hat = results_df$theta_hat[i],
        rank = results_df$rank[i],
        ci_two_sided = list(results_df$ci_two_left[i], results_df$ci_two_right[i]),
        ci_left = results_df$ci_left[i],
        ci_uniform_left = results_df$ci_uniform_left[i]
      )
    }),
    metadata = list(
      n_samples = nrow(df),
      k_methods = ncol(df),
      runtime_sec = runtime_sec,
      convergence_diff_l2 = diff_norm,
      convergence_diff_l2 = diff_norm,
      parent_step1_heterogeneity = step1_data$metadata$heterogeneity_index,
      parent_step1_spectral_gap = step1_data$metadata$spectral_gap,
      parent_step1_sparsity_ratio = step1_data$metadata$sparsity_ratio,
      parent_step1_mean_ci_width_top_5 = step1_data$metadata$mean_ci_width_top_5
    )
  )

  jsonlite::write_json(
    payload,
    file.path(out_dir, "ranking_results_step2.json"),
    pretty = TRUE, auto_unbox = TRUE
  )

  utils::write.csv(results_df, file.path(out_dir, "ranking_results_step2.csv"), row.names = FALSE)
}

tryCatch({
  main()
}, error = function(e) {
  message("Error: ", e$message)
  quit(status = 1)
})
