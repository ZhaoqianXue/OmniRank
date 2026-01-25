#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  requireNamespace("readr", quietly = TRUE)
  requireNamespace("dplyr", quietly = TRUE)
  requireNamespace("jsonlite", quietly = TRUE)
})

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
  required <- c("csv", "bigbetter", "B", "seed", "out")
  for (r in required) {
    if (is.null(kv[[r]])) stop(sprintf("Missing required argument: --%s", r))
  }
  kv$B <- as.integer(kv$B)
  kv$seed <- as.integer(kv$seed)
  kv$bigbetter <- as.integer(kv$bigbetter)
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

vanilla_spectrum_method <- function(AA2, WW2, Idx, B = 2000) {
  n <- ncol(AA2)
  L2 <- nrow(AA2)
  fAvec2 <- numeric(L2) + 2

  dval2 <- 2 * max(colSums(AA2))
  P2 <- matrix(0, n, n)
  for (i in 1:n) {
    for (j in 1:n) {
      if (j != i) {
        P2[i, j] <- sum(AA2[, i] * AA2[, j] * WW2[, j] / fAvec2) / dval2
      }
    }
    P2[i, i] <- 1 - sum(P2[i, ])
  }

  tmp.P2 <- t(t(P2) - diag(n)) %*% (t(P2) - diag(n))
  tmp.svd2 <- svd(tmp.P2)
  pihat2 <- abs(tmp.svd2$v[, n])

  # Handle potential zero or very small values to avoid -Inf in log
  pihat2 <- pmax(pihat2, .Machine$double.eps)

  log_pihat2 <- log(pihat2)
  thetahat2 <- log_pihat2 - mean(log_pihat2, na.rm = TRUE)

  RR2 <- matrix(0, 6, n)
  colnames(RR2) <- Idx
  RR2[1, ] <- thetahat2
  RR2[2, ] <- n + 1 - rank(thetahat2)

  Vmatrix2 <- matrix(0, L2, n)
  tauhatvec2 <- numeric(n)
  tmp.pimatrix2 <- t(AA2) * pihat2
  tmp.pivec2 <- colSums(tmp.pimatrix2)
  tmp.var2 <- numeric(n)

  for (oo in 1:n) {
    tauhatvec2[oo] <- sum(AA2[, oo] * (1 - pihat2[oo] / tmp.pivec2) * pihat2[oo] / fAvec2, na.rm = TRUE) / dval2
    tmp.var2[oo] <- sum(AA2[, oo] * (tmp.pivec2 - pihat2[oo]) / fAvec2 / fAvec2) * pihat2[oo] / dval2 / dval2 / tauhatvec2[oo] / tauhatvec2[oo]
    Vmatrix2[, oo] <- (AA2[, oo] * WW2[, oo] * tmp.pivec2 - AA2[, oo] * pihat2[oo]) / fAvec2
  }
  sigmahatmatrix2 <- matrix(tmp.var2, n, n) + t(matrix(tmp.var2, n, n))

  Wmatrix2 <- matrix(rnorm(L2 * B), L2, B)
  tmp.Vtau2 <- (t(Vmatrix2) / tauhatvec2) %*% Wmatrix2

  R.left.m2 <- numeric(n)
  R.right.m2 <- numeric(n)
  R.left.one.m2 <- numeric(n)
  for (ooo in 1:n) {
    tmpGMmatrix02 <- matrix(rep(tmp.Vtau2[ooo, ], n) - c(t(tmp.Vtau2)), B, n)
    tmpGMmatrix2 <- abs(t(t(tmpGMmatrix02) / sqrt(sigmahatmatrix2[ooo, ])) / dval2)
    tmpGMmatrixone2 <- t(t(tmpGMmatrix02) / sqrt(sigmahatmatrix2[ooo, ])) / dval2
    tmp.GMvecmax2 <- apply(tmpGMmatrix2, 1, max)
    tmp.GMvecmaxone2 <- apply(tmpGMmatrixone2, 1, max)
    cutval2 <- stats::quantile(tmp.GMvecmax2, 0.95)
    cutvalone2 <- stats::quantile(tmp.GMvecmaxone2, 0.95)
    tmp.theta.sd2 <- sqrt(sigmahatmatrix2[ooo, ])
    tmp.theta.sd2 <- tmp.theta.sd2[-ooo]
    R.left.m2[ooo] <- 1 + sum(1 * (((thetahat2[-ooo] - thetahat2[ooo]) / tmp.theta.sd2) > cutval2))
    R.right.m2[ooo] <- n - sum(1 * (((thetahat2[-ooo] - thetahat2[ooo]) / tmp.theta.sd2) < (-cutval2)))
    R.left.one.m2[ooo] <- 1 + sum(1 * (((thetahat2[-ooo] - thetahat2[ooo]) / tmp.theta.sd2) > cutvalone2))
  }

  # Uniform left-sided CI
  Wmatrix2b <- matrix(rnorm(L2 * B), L2, B)
  tmp.Vtau2b <- (t(Vmatrix2) / tauhatvec2) %*% Wmatrix2b
  GMvecmaxone2 <- numeric(B) - Inf
  for (ooo in 1:n) {
    tmpGMmatrix02 <- matrix(rep(tmp.Vtau2b[ooo, ], n) - c(t(tmp.Vtau2b)), B, n)
    tmpGMmatrixone2 <- t(t(tmpGMmatrix02) / sqrt(sigmahatmatrix2[ooo, ])) / dval2
    tmp.GMvecmaxone2 <- apply(tmpGMmatrixone2, 1, max)
    GMvecmaxone2 <- c(GMvecmaxone2, tmp.GMvecmaxone2)
  }
  GMmaxmatrixone2 <- matrix(GMvecmaxone2, B)
  GMmaxone2 <- apply(GMmaxmatrixone2, 1, max)
  cutvalone2 <- stats::quantile(GMmaxone2, 0.95)
  R.left.one2 <- numeric(n)
  for (oooo in 1:n) {
    tmp.theta.sd2 <- sqrt(sigmahatmatrix2[oooo, ])
    tmp.theta.sd2 <- tmp.theta.sd2[-oooo]
    R.left.one2[oooo] <- 1 + sum(1 * (((thetahat2[-oooo] - thetahat2[oooo]) / tmp.theta.sd2) > cutvalone2))
  }

  RR2[3, ] <- R.left.m2
  RR2[4, ] <- R.right.m2
  RR2[5, ] <- R.left.one.m2
  RR2[6, ] <- R.left.one2
  rownames(RR2) <- c("theta.hat", "Ranking", "two-sided CI", "two-sided CI", "left-sided CI", "uniform left-sided CI")
  RR2
}

main <- function() {
  start_time <- Sys.time()
  args <- parse_args()
  csv_path <- args$csv
  out_dir <- args$out
  bigbetter_flag <- as.integer(args$bigbetter) == 1
  B <- as.integer(args$B)
  seed <- as.integer(args$seed)

  safe_dir_create(out_dir)
  set.seed(seed)

  # Read CSV
  df <- tryCatch({
    readr::read_csv(csv_path, show_col_types = FALSE)
  }, error = function(e) {
    message("Falling back to base::read.csv: ", e$message)
    utils::read.csv(csv_path, stringsAsFactors = FALSE, check.names = TRUE)
  })

  # Drop non-numeric columns and known metadata columns if present
  if (requireNamespace("dplyr", quietly = TRUE)) {
    df <- dplyr::select(df, -dplyr::any_of(c("case_num", "model", "description")))
    df <- dplyr::select(df, where(is.numeric))
  } else {
    keep <- vapply(df, is.numeric, logical(1))
    df <- df[, keep, drop = FALSE]
  }

  if (ncol(df) < 2) {
    stop("At least two numeric method columns are required")
  }

  pdata <- process_data(df, bigbetter = bigbetter_flag)
  RR2 <- vanilla_spectrum_method(pdata$aa, pdata$ww, pdata$idx, B = B)

  # Calculate Diagnostic Metrics
  # 1. Heterogeneity Index: CV of comparison counts (rowSums of pdata$aa)
  # pdata$aa is L x n (num interactions x num items)
  # Total comparisons per item = colSums(pdata$aa)
  item_counts <- colSums(pdata$aa)
  heterogeneity_index <- sd(item_counts) / mean(item_counts)
  
  # 2. Spectral Gap: Difference between top-1 and top-2 eigenvalues of Transition Matrix P
  # We need P from vanilla_spectrum_method. Since vanilla_spectrum_method doesn't return P,
  # we re-compute the eigenvalues here efficiently or we modify vanilla_spectrum_method.
  # For minimal intrusion, we re-compute P locally.
  n <- ncol(pdata$aa)
  dval <- 2 * max(colSums(pdata$aa))
  weights <- numeric(nrow(pdata$aa)) + 2 # Vanilla weights
  P <- matrix(0, n, n)
  for (i in 1:n) {
    for (j in 1:n) {
      if (j != i) {
        P[i, j] <- sum(pdata$aa[, i] * pdata$aa[, j] * pdata$ww[, j] / weights) / dval
      }
    }
    P[i, i] <- 1 - sum(P[i, ])
  }
  eigen_vals <- eigen(t(P))$values
  sorted_moduli <- sort(Mod(eigen_vals), decreasing = TRUE)
  # Spectral gap is typically 1 - |lambda_2| (lambda_1 is always 1 for stochastic matrix)
  spectral_gap <- if (length(sorted_moduli) >= 2) (sorted_moduli[1] - sorted_moduli[2]) else 0

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

  runtime_sec <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  payload <- list(
    job_id = basename(dirname(out_dir)),
    params = list(bigbetter = bigbetter_flag, B = B, seed = seed),
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
      heterogeneity_index = heterogeneity_index,
      spectral_gap = spectral_gap,
      sparsity_ratio = nrow(df) / (n * log(n)),
      mean_ci_width_top_5 = mean(results_df$ci_two_right[results_df$rank <= 5] - results_df$ci_two_left[results_df$rank <= 5], na.rm = TRUE)
    )
  )

  jsonlite::write_json(
    payload,
    file.path(out_dir, "ranking_results.json"),
    pretty = TRUE, auto_unbox = TRUE
  )

  utils::write.csv(results_df, file.path(out_dir, "ranking_results.csv"), row.names = FALSE)
}

tryCatch({
  main()
}, error = function(e) {
  message("Error: ", e$message)
  quit(status = 1)
})






