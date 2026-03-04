"""Plate diagram for the Joint ALM model using daft==0.0.4."""
import matplotlib
matplotlib.use("Agg")
import daft

pgm = daft.PGM([16, 17], origin=[-1, -5.5], node_unit=1.4)

# ═══════════════════════════════════════════════════════════════
# Global parameters (top row)
#   Left:   log_alpha (shape) + theta_W = {log_lambda, gamma_k}
#   Center: progression (beta, beta_k, mu_k)
#   Right:  Sigmoid group (theta_S = {gamma_H, EC50, P0, log_phi})
# ═══════════════════════════════════════════════════════════════
pgm.add_node(daft.Node("log_alpha", r"$\log\alpha$", 0, -1, scale=1.2))
pgm.add_node(daft.Node("theta_w", r"$\boldsymbol{\theta}_W$", 0, 3, scale=1.2))
# Regression coefficients in shared predictor plate
pgm.add_node(daft.Node("beta", r"$\boldsymbol{\beta}_s$", 5, 10, scale=1.2))
pgm.add_node(daft.Node("beta_k", r"$\boldsymbol{\beta}_k$", 7, 10, scale=1.2))
pgm.add_plate(daft.Plate([4.2, 9.3, 3.6, 1.2], label=r"$p{=}1..P$", shift=-0.1))
pgm.add_node(daft.Node("mu_k", r"$\mu_k$", 11, -2.8, scale=1.2))
pgm.add_node(daft.Node("theta_s", r"$\boldsymbol{\theta}_S$", 13.5, 2, scale=1.2))

# Second row — hyperparameters / variances
pgm.add_node(daft.Node("sigma_cs", r"$\sigma_c^{(s)}$", 2.5, 8.5, scale=1.2))
pgm.add_node(daft.Node("sigma_ck", r"$\sigma_c^{(k)}$", 8.5, 8.5, scale=1.2))
pgm.add_node(daft.Node("omega_k", r"$\omega_k$", 10.5, 8.5, scale=1.2))

# Country random effects — survival scale (left, near beta)
pgm.add_node(daft.Node("mu_cs", r"$\mu_c^{(s)}$", 2.5, 7.2, scale=1.2))
pgm.add_plate(daft.Plate([1.7, 6.6, 1.6, 1.2], label=r"$c{=}1..C$", shift=-0.1))
pgm.add_edge("sigma_cs", "mu_cs")

# Country random effects — progression rate (right, near mu_k/beta_k)
pgm.add_node(daft.Node("mu_ck", r"$\mu_c^{(k)}$", 8.5, 7.2, scale=1.2))
pgm.add_plate(daft.Plate([7.7, 6.6, 1.6, 1.2], label=r"$c{=}1..C$", shift=-0.1))
pgm.add_edge("sigma_ck", "mu_ck")

# ═══════════════════════════════════════════════════════════════
# TIER 2 plate  (n2 patients: survival + MRC)
# ═══════════════════════════════════════════════════════════════
pgm.add_node(daft.Node("z_i", r"$z_i$", 10.5, 5, scale=1.2))
pgm.add_node(daft.Node("log_k2", r"$\log k_i$", 8, 4, scale=1.2))
pgm.add_node(daft.Node("log_s2", r"$\log\sigma_i$", 4, 3, scale=1.2))
pgm.add_node(daft.Node("t2", r"$t_i^{(2)}$", 4, 1.5, observed=True, scale=1.2))

# MRC sub-plate (J obs per patient)
pgm.add_node(daft.Node("mu_mrc", r"$\mu_{ij}$", 11, 3, scale=1.2))
pgm.add_node(daft.Node("y_mrc", r"$y_{ij}$", 11, 1.5, observed=True, scale=1.2))
pgm.add_plate(daft.Plate([10.1, 0.8, 1.8, 2.8], label=r"$j{=}1..J$", shift=-0.1))

# Tier 2 outer plate
pgm.add_plate(daft.Plate([2.5, 0.3, 10, 5.5], label=r"$i{=}1..n_2$", shift=-0.1))

# Tier 2 edges — progression rate
pgm.add_edge("mu_k", "log_k2")
pgm.add_edge("beta_k", "log_k2")
pgm.add_edge("mu_ck", "log_k2")
pgm.add_edge("omega_k", "z_i")
pgm.add_edge("z_i", "log_k2")

# Tier 2 edges — survival scale
pgm.add_edge("theta_w", "log_s2")
pgm.add_edge("beta", "log_s2")
pgm.add_edge("log_k2", "log_s2")
pgm.add_edge("mu_cs", "log_s2")

# Tier 2 edges — survival time
pgm.add_edge("log_alpha", "t2")
pgm.add_edge("log_s2", "t2")

# Tier 2 edges — MRC
pgm.add_edge("log_k2", "mu_mrc")
pgm.add_edge("theta_s", "mu_mrc")
pgm.add_edge("mu_mrc", "y_mrc")
pgm.add_edge("theta_s", "y_mrc")

# ═══════════════════════════════════════════════════════════════
# TIER 1 plate  (n1 patients: survival only, no z_i)
# ═══════════════════════════════════════════════════════════════
pgm.add_node(daft.Node("log_k1", r"$\log k_i$", 8, -1.5, scale=1.2))
pgm.add_node(daft.Node("log_s1", r"$\log\sigma_i$", 4, -2.8, scale=1.2))
pgm.add_node(daft.Node("t1", r"$t_i^{(1)}$", 4, -4, observed=True, scale=1.2))

pgm.add_plate(daft.Plate([2.5, -4.7, 7, 4.2], label=r"$i{=}1..n_1$", shift=-0.1))

# Tier 1 edges — progression rate
pgm.add_edge("mu_k", "log_k1")
pgm.add_edge("beta_k", "log_k1")
pgm.add_edge("mu_ck", "log_k1")

# Tier 1 edges — survival scale
pgm.add_edge("theta_w", "log_s1")
pgm.add_edge("beta", "log_s1")
pgm.add_edge("log_k1", "log_s1")
pgm.add_edge("mu_cs", "log_s1")

# Tier 1 edges — survival time
pgm.add_edge("log_alpha", "t1")
pgm.add_edge("log_s1", "t1")

# ═══════════════════════════════════════════════════════════════
# Render
# ═══════════════════════════════════════════════════════════════
pgm.render()
fig = pgm.figure
fig.savefig("plate_diagram.pdf", bbox_inches="tight")
fig.savefig("plate_diagram.png", bbox_inches="tight", dpi=200)
print("Saved plate_diagram.pdf and plate_diagram.png")
