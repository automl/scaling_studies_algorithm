from __future__ import annotations



MARKERS = {
    "good-good": "o",      # Circle
    "good-bad": "v",       # Triangle Down
    "good-medium": ".",    # dot
    "good": "h",           # Hexagon
    "bad-good": "^",       # Triangle Up
    "bad-bad": "x",        # X
    "bad-medium": "d",     # Diamond
    "bad": "P",            # Plus (Filled)
    "medium-good": "p",    # Pentagon
    "medium-bad": "*",     # Star
    "medium-medium": "+",  # Plus
    "medium": "8",         # Octagon
}

# Optimizer Colors for Scaling Studies (NeurIPS-friendly palette)
OPTIMIZER_COLORS = {
    # Bayesian Optimization variants
    "bo_space_expansion": "#0173B2",
    "bo_space_expansion_with_priors": "#0173B2",

    "bo_left_budget": "#DE8F05",
    "bo_left_budget_with_priors": "#DE8F05",
    

    "random_search": "#CC78BC",
    "random_search_with_priors": "#CC78BC",
    
    # Cost-Aware methods
    "carbs": "#CA0020",
    "carbs_with_priors": "#CA0020",
    
    "scaling_law_guided": "#8214A1",
    "chinchilla_guided_scaling": "#8214A1",
    
    # Chinchilla approach 1 variants
    "chinchilla_approach1": "#005C31",
    "chinchilla_approach1_with_priors": "#005C31",
    
    # Chinchilla approach 2 variants
    "chinchilla_approach2": "#0173B2",
    "chinchilla_approach2_with_priors": "#0173B2",
    
    
    # gemstones variants
    "gemstones": "#029E73",
    "gemstones_with_priors": "#029E73",

    # Default fallback
    "default": "#404040",
}

BENCHMARK_TITLES = {
    "openclip": "OpenCLIP",
    "llm": "LLM",
}