```@raw html
---
layout: home

hero:
  name: PhaseSkate
  text: High Performance Bayesian Inference In Julia
  tagline: Fast sampling built for complex models on your laptop.
  actions:
    - theme: brand
      text: Getting Started
      link: /getting_started
    - theme: alt
      text: API Reference
      link: /api

---
```

````@raw html
<div class="vp-doc case-study showcase-toggle">

<input type="radio" name="showcase" id="showcase-casestudy" checked>
<input type="radio" name="showcase" id="showcase-ide">

<div class="showcase-bar">
  <label for="showcase-casestudy">Case Study</label>
  <label for="showcase-ide">PPL IDE</label>
</div>

<div class="showcase-panels">

<div class="showcase-panel showcase-panel-casestudy">

<h1 class="case-study-title">Case Study: Multi-site Hierarchical Survival Model</h1>
<p class="case-study-subtitle">Sample a complex, hierarchical survival model with thousands of observations in approximately a minute with Enzyme powered dense mass matrix NUTS sampling.</p>

<table class="model-summary">
  <thead><tr><th>Observations</th><th>Hospitals</th><th>Covariates</th><th>Parameters</th><th>Chains</th><th>Draws / chain</th></tr></thead>
  <tbody><tr><td>5,000</td><td>100</td><td>8</td><td>214</td><td>4</td><td>2,000</td></tr></tbody>
</table>

<div class="case-study-grid">

<div class="plot-card">
  <img src="/assets/survival_example.svg"
       alt="Hierarchical survival model: posterior predictive curves, scaling benchmark, and posterior agreement" />
</div>

<div class="model-toggle">
  <input type="radio" name="model-view" id="toggle-desc" checked>
  <input type="radio" name="model-view" id="toggle-code">

  <div class="toggle-bar">
    <label for="toggle-desc">Description</label>
    <label for="toggle-code">Code</label>
  </div>

  <div class="panels-wrapper">
  <div class="panel-desc">
    <ul>
      <li><b>Weibull accelerated failure time (AFT) likelihood</b> for right-censored time-to-event data across <i>N</i> patients</li>
      <li><b>8 patient-level covariates</b> with regularised regression coefficients (&beta; ~ Normal(0, 1))</li>
      <li><b>Correlated site-level random effects</b> &mdash; hospital intercepts and treatment slopes share a bivariate correlation &rho; &isin; [&minus;1, 1] via non-centred parameterisation</li>
      <li><b>Right-censoring</b> handled natively: observed events contribute the Weibull log-pdf, censored observations contribute the log-CCDF</li>
      <li><b>Scales to hundreds of sites</b>: total parameter dimension grows as 13 + 2<i>H</i></li>
      <li><b>4 chains &times; 2 000 draws</b> with Enzyme LLVM autodiff &mdash; full posterior in seconds on a laptop</li>
    </ul>
  </div>

  <div class="panel-code">

```julia
@skate SurvivalFrailty begin
    @constants begin
        N::Int; P::Int; H::Int
        X::Matrix{Float64}; trt::Vector{Float64}; times::Vector{Float64}
        hosp::Vector{Int}; obs_idx::Vector{Int}; cens_idx::Vector{Int}
    end
    @params begin
        log_alpha::Float64; beta_0::Float64
        trt_effect::Float64
        beta = param(Vector{Float64}, P)
        log_sigma_int::Float64; log_sigma_slope::Float64
        rho = param(Float64; lower=-1.0, upper=1.0)
        z_int = param(Vector{Float64}, H)
        z_slope = param(Vector{Float64}, H)
    end
    @logjoint begin
        alpha = exp(log_alpha)
        sigma_int = exp(log_sigma_int)
        sigma_slope = exp(log_sigma_slope)
        sqrt_1mrho2 = sqrt(1.0 - rho * rho)

        target += normal_lpdf(log_alpha, 0.0, 0.5)
        target += normal_lpdf(beta_0, 2.0, 2.0)
        target += normal_lpdf(trt_effect, 0.0, 1.0)
        target += multi_normal_diag_lpdf(beta, 0.0, 1.0)
        target += normal_lpdf(log_sigma_int, -1.0, 1.0)
        target += normal_lpdf(log_sigma_slope, -1.0, 1.0)
        target += log(1.0 - rho * rho)
        target += multi_normal_diag_lpdf(z_int, 0.0, 1.0)
        target += multi_normal_diag_lpdf(z_slope, 0.0, 1.0)

        @for log_scale = beta_0 .+ (X * beta)
            .+ sigma_int .* z_int[hosp]
            .+ trt .* (trt_effect .+ sigma_slope
                .* (rho .* z_int[hosp]
                .+ sqrt_1mrho2 .* z_slope[hosp]))

        target += weibull_logsigma_lpdf_sum(times, alpha, log_scale, obs_idx)
        target += weibull_logsigma_lccdf_sum(times, alpha, log_scale, cens_idx)
    end
end
```

  </div>
  </div>
</div>

</div> <!-- end case-study-grid -->

</div> <!-- end showcase-panel-casestudy -->

<div class="showcase-panel showcase-panel-ide">

<h1 class="case-study-title">PhaseSkate IDE</h1>
<p class="case-study-subtitle">Watch a demonstration of the real-time streaming of posterior samples from the PhaseSkate Terminal User Interface (TUI).</p>

<div class="ide-demo-card">
  <div id="ide-player"></div>
</div>

<div
  id="ide-player-loader"
  data-cast="/PhaseSkate/dev/assets/demo.cast"
  style="display:none;"
></div>

</div> <!-- end showcase-panel-ide -->

</div> <!-- end showcase-panels -->

</div> <!-- end showcase-toggle -->

<script setup>
import { onMounted } from 'vue'

onMounted(() => {
  const el = document.getElementById('ide-player')
  if (!el || el.hasChildNodes()) return

  const script = document.createElement('script')
  script.src = 'https://unpkg.com/asciinema-player@3.9.0/dist/bundle/asciinema-player.min.js'
  script.onload = () => {
    AsciinemaPlayer.create(
      '/PhaseSkate/dev/assets/demo.cast',
      el,
      { autoPlay: true, loop: true, speed: 2, theme: 'monokai', fit: 'width', cols: 220, rows: 55 }
    )
  }
  document.head.appendChild(script)
})
</script>
````
