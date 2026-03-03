# benchmarks/bench_stan.R
# Benchmark CmdStan on the JointALM model using CmdStanR.
#
# Prerequisites:
#   install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
#   cmdstanr::install_cmdstan()
#   install.packages("jsonlite")
#
# Usage: Rscript benchmarks/bench_stan.R

library(cmdstanr)
library(jsonlite)

# ── Configuration ────────────────────────────────────────────────────────────

NUM_CHAINS  <- 4
NUM_WARMUP  <- 1000
NUM_SAMPLES <- 2000
SEED        <- 42

bench_dir <- dirname(sys.frame(1)$ofile)
if (is.null(bench_dir)) bench_dir <- "benchmarks"

data_path   <- file.path(bench_dir, "data", "joint_alm_data.json")
stan_path   <- file.path(bench_dir, "joint_alm.stan")
result_path <- file.path(bench_dir, "results", "stan_results.json")

# ── Load data ────────────────────────────────────────────────────────────────

cat("Loading data from:", data_path, "\n")
data <- fromJSON(data_path)

# CmdStan expects matrices as list-of-rows (already correct from JSON arrays)
# Integer arrays need to stay integer
stan_data <- list(
  n1               = as.integer(data$n1),
  n2               = as.integer(data$n2),
  p                = as.integer(data$p),
  n_countries      = as.integer(data$n_countries),
  MRC_MAX          = as.integer(data$MRC_MAX),
  tier1_times      = as.numeric(data$tier1_times),
  tier1_X          = as.matrix(data$tier1_X),
  tier1_country_ids = as.integer(data$tier1_country_ids),
  n1_obs           = as.integer(data$n1_obs),
  n1_cens          = as.integer(data$n1_cens),
  tier1_obs_idx    = as.integer(data$tier1_obs_idx),
  tier1_cens_idx   = as.integer(data$tier1_cens_idx),
  tier2_times      = as.numeric(data$tier2_times),
  tier2_X          = as.matrix(data$tier2_X),
  tier2_country_ids = as.integer(data$tier2_country_ids),
  n2_obs           = as.integer(data$n2_obs),
  n2_cens          = as.integer(data$n2_cens),
  tier2_obs_idx    = as.integer(data$tier2_obs_idx),
  tier2_cens_idx   = as.integer(data$tier2_cens_idx),
  total_mrc_obs    = as.integer(data$total_mrc_obs),
  mrc_scores_flat  = as.integer(data$mrc_scores_flat),
  mrc_times_flat   = as.numeric(data$mrc_times_flat),
  mrc_patient_ids  = as.integer(data$mrc_patient_ids)
)

# ── Compile model ────────────────────────────────────────────────────────────

cat("Compiling Stan model:", stan_path, "\n")
model <- cmdstan_model(stan_path)

# ── Sample ───────────────────────────────────────────────────────────────────

cat(sprintf("Sampling: %d warmup, %d samples, %d chains\n",
            NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS))

t_start <- proc.time()

fit <- model$sample(
  data            = stan_data,
  chains          = NUM_CHAINS,
  parallel_chains = NUM_CHAINS,
  iter_warmup     = NUM_WARMUP,
  iter_sampling   = NUM_SAMPLES,
  seed            = SEED,
  refresh         = 500
)

t_elapsed <- (proc.time() - t_start)["elapsed"]

# ── Diagnostics ──────────────────────────────────────────────────────────────

cat("\n")
fit$summary()

diag <- fit$diagnostic_summary()
n_divergent <- sum(diag$num_divergent)

# Get per-parameter ESS and Rhat from summary
summ <- fit$summary()
ess_bulk_vals <- summ$ess_bulk
rhat_vals     <- summ$rhat

# Filter to only parameters (exclude lp__, etc with NA ESS)
valid <- !is.na(ess_bulk_vals) & !is.na(rhat_vals)
min_ess  <- min(ess_bulk_vals[valid])
max_rhat <- max(rhat_vals[valid])

# Sampling time only (excluding warmup)
sampling_time <- sum(fit$time()$chains$sampling)
total_time    <- as.numeric(t_elapsed)

cat(sprintf("\n── Results ──────────────────────────────────────\n"))
cat(sprintf("  Total wall time:  %.1f s\n", total_time))
cat(sprintf("  Sampling time:    %.1f s\n", sampling_time))
cat(sprintf("  Min ESS (bulk):   %.0f\n", min_ess))
cat(sprintf("  Max Rhat:         %.4f\n", max_rhat))
cat(sprintf("  ESS/s:            %.1f\n", min_ess / sampling_time))
cat(sprintf("  Divergences:      %d\n", n_divergent))

# ── Save results ─────────────────────────────────────────────────────────────

dir.create(file.path(bench_dir, "results"), showWarnings = FALSE)

results <- list(
  backend        = "CmdStan",
  num_chains     = NUM_CHAINS,
  num_warmup     = NUM_WARMUP,
  num_samples    = NUM_SAMPLES,
  total_time_s   = total_time,
  sampling_time_s = sampling_time,
  min_ess_bulk   = min_ess,
  max_rhat       = max_rhat,
  ess_per_sec    = min_ess / sampling_time,
  divergences    = n_divergent
)

writeLines(toJSON(results, auto_unbox = TRUE, pretty = TRUE), result_path)
cat(sprintf("\nResults saved to: %s\n", result_path))
