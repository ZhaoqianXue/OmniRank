#!/usr/bin/env Rscript
# Plot violin and heatmaps for phenotype ranking data
# Usage: Rscript scripts/plot_phenotype_rankings.R [OPTIONS]
# Phenotype plots (require --csv):
#   --csv PATH: phenotype CSV path
#   --out PATH: violin PNG output
#   --heatmap-out PATH: heatmap PNG output
# CI forest plot (all modes, no --csv needed):
#   --ci-plot PATH: JSON with ranking results (items, theta_hat, ranks, ci_lower, ci_upper)
#   --ci-out PATH: CI forest PNG output
# Without args, generates violin + heatmaps + CI forest to output/

library(ggstatsplot)
library(gghalves)
library(tidyverse)
library(paletteer)
library(ggplot2)
library(dplyr)
library(ggrepel)
library(patchwork)
library(forcats)
library(grid)
library(jsonlite)

# Parse args
args <- commandArgs(trailingOnly = TRUE)
data_path <- file.path("data", "examples", "example_data_multiway_phenotype.csv")
out_dir <- "output"
violin_out_path <- NULL
heatmap_out_path <- NULL
ci_plot_path <- NULL
ci_out_path <- NULL
i <- 1
while (i <= length(args)) {
  if (args[[i]] == "--csv" && i + 1 <= length(args)) {
    data_path <- args[[i + 1]]
    i <- i + 2
  } else if (args[[i]] == "--out" && i + 1 <= length(args)) {
    violin_out_path <- args[[i + 1]]
    i <- i + 2
  } else if (args[[i]] == "--heatmap-out" && i + 1 <= length(args)) {
    heatmap_out_path <- args[[i + 1]]
    i <- i + 2
  } else if (args[[i]] == "--ci-plot" && i + 1 <= length(args)) {
    ci_plot_path <- args[[i + 1]]
    i <- i + 2
  } else if (args[[i]] == "--ci-out" && i + 1 <= length(args)) {
    ci_out_path <- args[[i + 1]]
    i <- i + 2
  } else {
    i <- i + 1
  }
}

# CI forest plot (all modes) - run and exit if requested
if (!is.null(ci_plot_path) && !is.null(ci_out_path)) {
  dat <- jsonlite::fromJSON(ci_plot_path)
  items <- dat$items
  theta_hat <- dat$theta_hat
  ranks <- dat$ranks
  ci_lower <- dat$ci_lower
  ci_upper <- dat$ci_upper
  if (length(items) == 0) {
    # Empty plot
    df <- data.frame(Method = character(), rank = numeric(), theta_hat = numeric(), ci_lower = numeric(), ci_upper = numeric())
    p_ci <- ggplot(df) + theme_void() + labs(title = "No ranking data available")
  } else {
    # Use original items order from JSON (no reorder by rank)
    df <- data.frame(
      Method = factor(items, levels = rev(items)),
      rank = ranks,
      theta_hat = theta_hat,
      ci_lower = ci_lower,
      ci_upper = ci_upper
    )
    x_min <- max(0.5, min(ci_lower) - 0.5)
    x_max <- min(max(ci_upper) + 0.5, max(ranks) + 2)
    x_breaks <- seq(1, ceiling(x_max), 1)
    x_breaks <- x_breaks[x_breaks >= x_min & x_breaks <= x_max]
    p_ci <- ggplot(df, aes(x = rank, y = Method)) +
      geom_vline(xintercept = x_breaks, color = "grey85", linewidth = 0.5) +
      geom_errorbar(aes(xmin = ci_lower, xmax = ci_upper), width = 0.3, linewidth = 0.8, color = "black", orientation = "y") +
      geom_point(size = 3, color = "red", fill = "red", shape = 21, stroke = 0) +
      geom_text(aes(x = rank, y = as.numeric(Method) + 0.35, label = round(rank)), inherit.aes = FALSE, size = 4, color = "blue", fontface = "bold") +
      scale_x_continuous(limits = c(x_min, x_max), breaks = x_breaks, expand = c(0.02, 0)) +
      labs(x = "Rank") +
      theme_bw(base_size = 14) +
      theme(
        plot.title = element_blank(),
        panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
        panel.grid.minor = element_blank(),
        axis.text = element_text(color = "black", size = 12),
        axis.title = element_text(size = 16, face = "bold", color = "black"),
        panel.background = element_rect(fill = "white"),
        plot.background = element_rect(fill = "white")
      )
  }
  dir.create(dirname(ci_out_path), showWarnings = FALSE, recursive = TRUE)
  ggsave(ci_out_path, p_ci, width = 10, height = max(5, 0.4 * max(1, length(items))), dpi = 150, limitsize = FALSE)
  message("Saved CI forest plot to ", ci_out_path)
  quit(save = "no", status = 0)
}

dir.create(out_dir, showWarnings = FALSE)

# Read data and convert to long format for violin plot
df_wide <- read_csv(data_path, show_col_types = FALSE)
colnames(df_wide)[1] <- "Traits"

table <- df_wide %>%
  pivot_longer(cols = -Traits, names_to = "Method", values_to = "Ranking") %>%
  filter(!is.na(Ranking))

# Normalize ranking to 1-N within each phenotype (lower raw value = better = rank 1)
table <- table %>%
  group_by(Traits) %>%
  mutate(Ranking = rank(Ranking, ties.method = "average")) %>%
  ungroup()

# Method order: use CSV column order (match Report), not sorted by mean rank
method_cols <- setdiff(colnames(df_wide), "Traits")
sorted_names <- method_cols
n_rank_levels <- max(length(method_cols), 2L)  # adaptive: 1 to n_methods
rank_breaks <- seq(1, n_rank_levels, 1)
table$Method <- factor(table$Method, levels = sorted_names)

# Violin plot - only right half violin (hide centered violin from ggbetweenstats)
p_violin <- ggbetweenstats(
  data = table,
  x = Method,
  y = Ranking,
  sample.size.label = "K = ",
  violin.args = list(width = 0, linewidth = 0),
  point.args = list(position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.6)),
  centrality.label.args = list(
    size = 7,
    nudge_x = 0.35,
    box.padding = 0.5,
    point.padding = 0.3,
    min.segment.length = 0.2
  ),
  ggtheme = ggplot2::theme_bw(base_size = 16),
  results.subtitle = FALSE,
  pairwise.display = "none",
  package = "colorBlindness",
  palette = "paletteMartin"
) +
  geom_half_violin(
    data = table,
    aes(x = Method, y = Ranking, fill = Method),
    side = "r",
    position = position_nudge(x = 0.15),
    alpha = 0.25,
    width = 0.75,
    scale = "width",
    adjust = 2
  ) +
  scale_fill_paletteer_d("colorBlindness::paletteMartin") +
  scale_y_reverse(breaks = rank_breaks, limits = c(1, n_rank_levels)) +
  labs(title = NULL, y = "Rank") +
  theme(
    plot.margin = margin(12, 16, 12, 12),
    text = element_text(size = 14, color = "black"),
    axis.title = element_text(size = 20, face = "bold", color = "black"),
    axis.title.x = element_text(margin = margin(t = 1)),
    axis.title.y = element_text(margin = margin(r = 1)),
    axis.text = element_text(color = "black"),
    axis.text.x = element_text(angle = 0, hjust = 0.5, size = 14),
    axis.text.y = element_text(size = 14)
  )

run_heatmaps <- is.null(violin_out_path) && is.null(heatmap_out_path)
if (run_heatmaps) {
  violin_out_path <- file.path(out_dir, "violin_ranking_over_phenotypes.png")
}
if (!is.null(violin_out_path)) {
  if (run_heatmaps) {
    ggsave(
      file.path(out_dir, "violin_ranking_over_phenotypes.pdf"),
      p_violin,
      width = 22,
      height = 7,
      device = "pdf"
    )
  }
  ggsave(violin_out_path, p_violin, width = 22, height = 7, dpi = 150)
  message("Saved violin plot to ", violin_out_path)
}

# Heatmap - when running standalone (no --out) or when --heatmap-out is provided
run_heatmap <- run_heatmaps || !is.null(heatmap_out_path)
if (run_heatmap) {
# Heatmap - original pheatmap structure
data_for_heatmap <- read_csv(data_path, show_col_types = FALSE)
data_for_heatmap <- as.matrix(data_for_heatmap)

rownames(data_for_heatmap) <- data_for_heatmap[, 1]
# Remove phenotype/traits column (original used "Traits")
trait_col <- colnames(data_for_heatmap)[1]
data_for_heatmap <- data_for_heatmap[, colnames(data_for_heatmap) != trait_col, drop = FALSE]

numeric_vector <- as.numeric(data_for_heatmap)
numeric_matrix <- matrix(
  numeric_vector,
  nrow = nrow(data_for_heatmap),
  ncol = ncol(data_for_heatmap)
)
colnames(numeric_matrix) <- colnames(data_for_heatmap)
rownames(numeric_matrix) <- rownames(data_for_heatmap)
# Normalize to ranks 1-N within each row (phenotype)
numeric_matrix <- t(apply(numeric_matrix, 1, function(x) rank(x, na.last = "keep", ties.method = "average")))
colnames(numeric_matrix) <- colnames(data_for_heatmap)
rownames(numeric_matrix) <- rownames(data_for_heatmap)
numeric_matrix <- numeric_matrix[, sorted_names]

# Single Phenotype Rankings heatmap (ggplot2 for legend height control)
# No title; legend: 1 (orange) at top, n_rank_levels (blue) at bottom; adaptive to data
mat_long <- as.data.frame(numeric_matrix) %>%
  tibble::rownames_to_column("Phenotype") %>%
  tidyr::pivot_longer(-Phenotype, names_to = "Method", values_to = "rank")
n_pheno <- nrow(numeric_matrix)
legend_bar_height_cm <- 15
p_heatmap <- ggplot2::ggplot(mat_long, ggplot2::aes(x = Method, y = Phenotype, fill = rank)) +
  ggplot2::geom_tile(color = "grey60", linewidth = 0.2) +
  ggplot2::scale_fill_gradientn(
    colors = colorRampPalette(c("orange", "blue"))(n_rank_levels),
    values = scales::rescale(seq(1, n_rank_levels)),
    limits = c(1, n_rank_levels),
    breaks = rank_breaks,
    na.value = "darkgrey",
    guide = ggplot2::guide_colorbar(
      title = NULL,
      barheight = grid::unit(legend_bar_height_cm, "cm"),
      nbin = n_rank_levels,
      reverse = TRUE  # 1 (orange) at top, n_rank_levels (blue) at bottom
    )
  ) +
  ggplot2::scale_x_discrete(limits = sorted_names) +
  ggplot2::scale_y_discrete(limits = rev(rownames(numeric_matrix))) +
  ggplot2::theme_bw(base_size = 14) +
  ggplot2::theme(
    plot.title = ggplot2::element_blank(),
    axis.text = ggplot2::element_text(color = "black"),
    axis.text.x = ggplot2::element_text(angle = 315, hjust = 0, vjust = 1, size = 14),
    axis.text.y = ggplot2::element_text(size = 14),
    axis.title = ggplot2::element_text(size = 18, face = "bold", color = "black"),
    panel.grid = ggplot2::element_blank(),
    legend.position = "right",
    legend.text = ggplot2::element_text(size = 14, color = "black")
  )

heatmap_png <- if (!is.null(heatmap_out_path)) heatmap_out_path else file.path(out_dir, "heatmap_phenotype_rankings.png")
if (run_heatmaps) {
  ggplot2::ggsave(
    file.path(out_dir, "heatmap_phenotype_rankings.pdf"),
    p_heatmap,
    width = 12,
    height = 14
  )
}
ggplot2::ggsave(
  heatmap_png,
  p_heatmap,
  width = 12,
  height = 14,
  dpi = 150
)
message("Saved heatmap to ", heatmap_png)
}

# CI forest plot in standalone mode (for tuning, uses mean rank from phenotype data)
if (run_heatmaps) {
  rank_summary <- table %>%
    group_by(Method) %>%
    summarise(mean_rank = mean(Ranking, na.rm = TRUE), .groups = "drop") %>%
    mutate(rank = rank(mean_rank, ties.method = "average"))
  n_m <- nrow(rank_summary)
  # Use original data order (sorted_names) like violin and heatmap
  df_ci <- rank_summary %>%
    mutate(Method = factor(Method, levels = rev(sorted_names))) %>%
    filter(!is.na(Method))
  ranks_ci <- df_ci$rank
  ci_lower_ci <- pmax(1, ranks_ci - 1.2)
  ci_upper_ci <- pmin(n_m, ranks_ci + 1.2)
  df_ci$ci_lower <- ci_lower_ci
  df_ci$ci_upper <- ci_upper_ci
  x_min_ci <- max(0.5, min(ci_lower_ci) - 0.5)
  x_max_ci <- min(max(ci_upper_ci) + 0.5, max(ranks_ci) + 2)
  x_breaks_ci <- seq(1, ceiling(x_max_ci), 1)
  x_breaks_ci <- x_breaks_ci[x_breaks_ci >= x_min_ci & x_breaks_ci <= x_max_ci]
  p_ci_standalone <- ggplot(df_ci, aes(x = rank, y = Method)) +
    geom_vline(xintercept = x_breaks_ci, color = "grey85", linewidth = 0.5) +
    geom_errorbar(aes(xmin = ci_lower, xmax = ci_upper), width = 0.3, linewidth = 0.8, color = "black", orientation = "y") +
    geom_point(size = 3, color = "red", fill = "red", shape = 21, stroke = 0) +
    geom_text(aes(x = rank, y = as.numeric(Method) + 0.35, label = round(rank)), inherit.aes = FALSE, size = 4, color = "blue", fontface = "bold") +
    scale_x_continuous(limits = c(x_min_ci, x_max_ci), breaks = x_breaks_ci, expand = c(0.02, 0)) +
    labs(x = "Rank") +
    theme_bw(base_size = 14) +
    theme(
      plot.title = element_blank(),
      panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      axis.text = element_text(color = "black", size = 12),
      axis.title = element_text(size = 16, face = "bold", color = "black"),
      panel.background = element_rect(fill = "white"),
      plot.background = element_rect(fill = "white")
    )
  ci_standalone_path <- file.path(out_dir, "ci_forest_ranking.png")
  ggsave(ci_standalone_path, p_ci_standalone, width = 10, height = max(5, 0.4 * n_m), dpi = 150, limitsize = FALSE)
  message("Saved CI forest plot to ", ci_standalone_path)
}

message("Done. Check output/ for violin_ranking_over_phenotypes.*, heatmap_phenotype_rankings.*, and ci_forest_ranking.png")
