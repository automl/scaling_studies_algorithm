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

COLORS_MAIN = {
    # Optimizers without Priors
    "SMAC_ParEGO": "saddlebrown",
    "RandomSearch": "peru",
    "NepsRW": "lightseagreen",
    "Nevergrad_EvolutionStrategy": "lime",

    "NepsHyperbandRW": "darkslategray",
    "NepsHyperbandRW_w_continuations": "darkslategray",

    "MOMFBO": "magenta",
    "MOMFBO_w_continuations": "magenta",

    "NepsMOASHA": "darkviolet",
    "NepsMOASHA_w_continuations": "darkviolet",

    "NepsMOASHA_RS": "violet",
    "NepsMOASHA_RS_w_continuations": "violet",

    "NepsMOASHABO": "darkblue",
    "NepsMOASHABO_w_continuations": "darkblue",

    # Optimizers with Priors

    # NepsPriorMOASHA
    "NepsPriorMOASHA": "darkgreen",
    "NepsPriorMOASHA_w_continuations": "darkgreen",

    # NepsPriorRSMOASHA
    "NepsPriorRSMOASHA": "darkblue",
    "NepsPriorRSMOASHA_w_continuations": "darkblue",

    # NepsMOPriorband
    "NepsMOPriorband": "cornflowerblue",
    "NepsMOPriorband_w_continuations": "cornflowerblue",

    # NepsPiBORW
    "NepsPiBORW": "olive",

    # NepsPriMO
    "NepsPriMO": "crimson",
    "NepsPriMO_w_continuations": "crimson",

    "NepsPriMO_HV": "darkred",
    "NepsPriMO_HV_w_continuations": "darkred",

    # RandomSearchWithPriors
    "RandomSearchWithPriors": "darkorange",

    # NepsNoInitPriMO
    "NepsNoInitPriMO": "lightseagreen",
    "NepsNoInitPriMO_w_continuations": "lightseagreen",

    # NepsPriMO_Init10
    "NepsPriMO_Init10": "darkslateblue",
    "NepsPriMO_Init10_w_continuations": "darkslateblue",

    # NepsInitPiBORW
    "NepsInitPiBORW": "olive",
    "NepsInitPiBORW_w_continuations": "olive",

    # NepsEtaPriorPriMO
    "NepsEtaPriorPriMO": "darkgreen",
    "NepsEtaPriorPriMO_w_continuations": "darkgreen",

    # NepsMFPriMO
    "NepsMFPriMO": "darkviolet",
    "NepsMFPriMO_w_continuations": "darkviolet",

    # NepsEtaPriorMFPriMO
    "NepsEtaPriorMFPriMO": "violet",
    "NepsEtaPriorMFPriMO_w_continuations": "violet",

    # Optuna
    "Optuna": "royalblue",

    # BO with EHVI
    "NepsMOBO": "darkgreen",
}


# Color for Single Objective Optimizers
COLORS_SO = {
    "NepsHyperband": "darkviolet",
    "NepsHyperband_w_continuations": "darkviolet",

    "NepsBO": "darkcyan",

    "NepsPriorband": "cornflowerblue",
    "NepsPriorband_w_continuations": "cornflowerblue",

    "NepsPriorbandBO": "darkslategray",
    "NepsPriorbandBO_w_continuations": "darkslategray",

    "NepsMFBO": "crimson",
    "NepsMFBO_w_continuations": "crimson",

    "NepsPiBO": "olive",
}


COLORS_HPS = {
    ("NepsMOASHABO", "initial_design_size=5"): "darkblue",
    ("NepsMOASHABO_w_continuations", "initial_design_size=5"): "darkblue",

    ("NepsMOASHABO", "initial_design_size=7"): "limegreen",
    ("NepsMOASHABO_w_continuations", "initial_design_size=7"): "limegreen",

    ("NepsMOASHABO", "initial_design_size=10"): "goldenrod",
    ("NepsMOASHABO_w_continuations", "initial_design_size=10"): "goldenrod",

    ("NepsPriMO", "initial_design_size=5"): "crimson",
    ("NepsPriMO_w_continuations", "initial_design_size=5"): "crimson",

    ("NepsPriMO", "initial_design_size=7"): "darkorange",
    ("NepsPriMO_w_continuations", "initial_design_size=7"): "darkorange",

    ("NepsPriMO", "initial_design_size=10"): "darkslateblue",
    ("NepsPriMO_w_continuations", "initial_design_size=10"): "darkslateblue",

    ("NepsPriMO", "sampler=etaprior"): "green",
    ("NepsPriMO_w_continuations", "sampler=etaprior"): "green",
}


HPOSUITE_COLORS = {
    "SMAC_BO": "saddlebrown",
    "Scikit_Optimize": "darkslategray",
    "Optuna": "royalblue",
}

# Optimizer Colors for Scaling Studies
OPTIMIZER_COLORS = {
    # Bayesian Optimization variants
    "bo_space_expansion": "blue",
    "bo_space_expansion_with_priors": "darkblue",

    "bo_left_budget": "orchid",
    "bo_left_budget_with_priors": "darkorchid",
    

    "random_search": "gold",
    "random_search_with_priors": "silver",
    
    # Cost-Aware methods
    "carbs": "crimson",
    "carbs_with_priors": "brown",
    
    "scaling_law_guided": "darkviolet",
    "chinchilla_guided_scaling": "mediumpurple",
    
    # Chinchilla approach 1 variants
    "chinchilla_approach1": "olive",
    "chinchilla_approach1_with_priors": "cyan",
    
    # Chinchilla approach 2 variants
    "chinchilla_approach2": "darkorange",
    "chinchilla_approach2_with_priors": "yellow",
    
    
    # gemstones variants
    "gemstones": "yellow",
    "gemstones_with_priors": "gray",

    # Default fallback
    "default": "gray",
}

RC_PARAMS = {
    "text.usetex": False,  # True,
    # "pgf.texsystem": "pdflatex",
    # "pgf.rcfonts": False,
    # "font.family": "serif",
    # "font.serif": [],
    # "font.sans-serif": [],
    # "font.monospace": [],
    # "font.size": "25",
    # "legend.fontsize": "large",
    "text.color": "black",
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    # "legend.title_fontsize": "small",
    "lines.linewidth": 1,
    "patch.linewidth": 1,
    "axes.linewidth": 2,
    "lines.markersize": 4,
    "axes.titlesize": 20,
    # "bottomlabel.weight": "normal",
    # "toplabel.weight": "normal",
    # "leftlabel.weight": "normal",
    # "tick.labelweight": "normal",
    # "title.weight": "normal",
    # "pgf.preamble": r"""
    #    \usepackage[T1]{fontenc}
    #    \usepackage[utf8x]{inputenc}
    #    \usepackage{microtype}
    # """,
}

XTICKS = {
    (1, 5): [1, 2, 3, 4, 5],
    (1, 10): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    (1, 15): [1, 3, 6, 9, 12, 15],
    (1, 20): [1, 4, 8, 12, 16, 20],
    (1, 25): [1, 5, 10, 15, 20, 25],
    (1, 30): [1, 6, 12, 18, 24, 30],
    (1, 100): [1, 20, 40, 60, 80, 100],
}


other_fig_params = {
    "suptitle_bbox": (0.5, 1.02),
    "xytick_labelsize": 15,
    "fig_size": (20, 15),
    "ovrank_xsize": 7,
    "ovrank_ysize": 7,
    "n_rows_cols": {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (1, 4),
        5: (2, 3),
        6: (2, 3),
        8: (2, 4),
        9: (3, 3),
        12: (3, 4),
        13: (3, 5),
        14: (2, 7),
    },
    "tight_layout_pads": {
        "pad": 0,
        "h_pad": 2.5,
    },
    "bbox_to_anchor": (0.5, -0.05),
    "legend_fontsize": 16,
    "title_fontsize": 20,
    "xylabel_fontsize": 22,
    "xlabel_start_i": {
        1 : 0,
        2 : 0,
        3 : 0,
        4 : 2,
        5 : 3,
        6 : 3,
        8 : 4,
        9 : 6,
        12: 8,
        13: 10,
        14: 7,
    },
    "ylabel_i_inc": {
        1 : 0,
        2 : 0,
        3 : 0,
        4 : 2,
        5 : 3,
        6 : 3,
        8 : 4,
        9 : 3,
        12: 4,
        13: 5,
        14: 2,
    },
    "multi_fig_leg_cols": {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 4,
        9: 5,
        10: 5,
        11: 5,
        12: 5,
        13: 5,
        14: 5,
        15: 5,
        16: 4,
    },
    "single_fig_leg_cols": {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 3,
        7: 4,
        8: 4,
        9: 4,
        10: 2,
        11: 3,
        12: 2,
        13: 5,
        14: 2,
    },
    "stitched_cols": {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 4,
        9: 5,
        10: 5,
        11: 5,
        12: 5,
    },
    "stitched_xylabel_fontsize": 22,
    "stitched_leg_fontsize": 18,
}