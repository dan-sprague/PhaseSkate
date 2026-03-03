using Documenter
using PhaseSkate

makedocs(
    sitename = "PhaseSkate",
    modules = [PhaseSkate],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://YOUR_USERNAME.github.io/PhaseSkate.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Model DSL" => "dsl.md",
        "Samplers" => "samplers.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/YOUR_USERNAME/PhaseSkate.jl.git",
    devbranch = "main",
)
