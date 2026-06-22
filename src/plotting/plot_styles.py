from __future__ import annotations


OPTIMIZER_COLORS = {
    "bo_space_expansion_mes": "#009E73",              # Okabe-Ito bluish green
    "bo_space_expansion_mes_with_priors": "#005C42",  # dark bluish green

    "bo_left_budget_zero_mean_ei": "#9E02FF",              
    "bo_left_budget_zero_mean_ei_with_priors": "#4B0082",

    "bo_space_expansion_modified_jes": "#110C0E",              # Okabe-Ito sky blue
    "bo_left_budget_modified_jes": "#5DE86D",              # Ok

    "bo_space_expansion_ei": "#56B4E9",              # Okabe-Ito sky blue
    "bo_space_expansion_ei_with_priors": "#2277BB",  # dark sky blue

    "bo_space_expansion_jes": "#000000",              # black
    "bo_space_expansion_jes_with_priors": "#555555",  # dark gray

    "bo_left_budget_mes": "#002CF2",                    # Okabe-Ito orange
    "bo_left_budget_mes_with_priors": "#A67000",        # dark orange

    "bo_left_budget": "#D55E00",                     # Okabe-Ito vermilion
    "bo_left_budget_with_priors": "#8B3D00",         # dark vermilion
    "bo_left_budget_ei": "#7F756E",                     # Okabe-Ito vermilion
    "bo_left_budget_ei_with_priors": "#8B3D00",         # dark vermilion

    "bo_left_budget_jes": "#F0E442",                    # Okabe-Ito yellow
    "bo_left_budget_jes_with_priors": "#B0A800",        # dark yellow

    "random_search": "#CC78BC",
    "random_search_with_priors": "#CC78BC",
    
    # Cost-Aware methods
    "carbs": "#CA0020",
    "carbs_with_priors": "#CA0020",
    
    # Chinchilla approach 1 variants
    "chinchilla_approach1": "#F4A261",
    "chinchilla_approach1_with_priors": "#F4A261",
    
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
    "tabpfn": "TabPFN",
}
