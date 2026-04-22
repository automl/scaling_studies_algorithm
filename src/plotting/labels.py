from __future__ import annotations

INTRO_LABELS = {
    "SMAC_ParEGO": "ParEGO",
    "NepsRW": "BO+RW",
    "RandomSearchWithPriors": "RS+Prior",
    "NepsPriMO": "PriMO",
    "NepsMOASHA": "MOASHA",
    "Nevergrad_EvolutionStrategy": "NSGA-II",
}



LABELS_1 = {

    "NepsPriMO": "PriMO",

    "NepsPriMO_HV": "PriMO-HV",

    "SMAC_ParEGO": "ParEGO",
    "RandomSearch": "Random Search",
    "NepsRW": "BO+RW",
    "Nevergrad_EvolutionStrategy": "NSGA-II",

    "NepsHyperbandRW": "HB+RW",

    "NepsMOASHA": "MOASHA",

    "NepsMOASHA_RS": "MOASHA+RS",

    "NepsMOASHABO": "MOASHA-BO+RW",

    "MOMFBO": "MOMF-BO",

    "NepsPriorMOASHA": "MOASHA+Prior",

    "NepsMOPriorband": "MO-Priorband",

    "NepsPiBORW": "πBO+RW",

    "RandomSearchWithPriors": "RS+Prior",

    "NepsPriorRSMOASHA": "MOASHA+Prior(50%)",

    "Optuna": "MO-TPE",

    "NepsMOBO": "LogNEHVI",
}

ABLATION_LABELS = {

    "NepsNoInitPriMO": "PriMO w/o initial design",

    "NepsMOASHA_RS": "PriMO w/o Priors and BO",

    "NepsPriMO": "PriMO",

    "NepsMOASHABO": "PriMO w/o Priors",

    "NepsInitPiBORW": "PriMO w/o ϵ-BO",

    "NepsEtaPriorPriMO": "PriMO + 1/η Priors in initial design",

    "NepsMFPriMO": "MO-ASHA + PriMO Sampler",

    "NepsEtaPriorMFPriMO": "MO-ASHA + 1/η Priors + PriMO Sampler",
}

ABLATION_INIT_LABELS = {

    "NepsPriMO": "init=5",
    "NepsNoInitPriMO": "init=0",
    "NepsPriMO_Init10": "init=10",
}

HP_LABELS = {
    "initial_design_size": "init",
}

SIG_LABELS = {

    "NepsPriMO": "PriMO",

    "SMAC_ParEGO": "ParEGO",
    "RandomSearch": "RS",
    "NepsRW": "BO+RW",
    "Nevergrad_EvolutionStrategy": "NSGA-II",

    "NepsHyperbandRW": "HB+RW",

    "NepsMOASHA": "MOASHA",

    "NepsMOASHA_RS": "MOASHA+RS",

    "NepsMOASHABO": "MOASHA-BO+RW",

    "MOMFBO": "MOMF-BO",

    "NepsPriorMOASHA": "MOASHA+Prior",

    "NepsMOPriorband": "MO-PB",

    "NepsPiBORW": "πBO+RW",

    "RandomSearchWithPriors": "RS+Prior",

    "Optuna": "MO-TPE",

    "NepsMOBO": "LogNEHVI",
}

SO_LABELS = {

    "NepsHyperband": "HyperBand",
    "NepsBO": "BO",
    "NepsPriorband": "Priorband",
    "NepsPriorbandBO": "Priorband+BO",
    "NepsMFBO": "PriMO",
    "NepsPiBO": "πBO",
}


GENERAL_HPOSUITE = {
    "NepsBO": "Neps BO",
    "NepsHyperband": "Neps HyperBand",
    "NepsPriorband": "Neps Priorband",
    "SMAC_BO": "SMAC BO",
    "RandomSearch": "Random Search",
    "Scikit_Optimize": "Skopt BO",
    "Optuna": "Optuna TPE",
}




#################
# NOT IN USE
#################

LABELS_99 = {

    # Our Optimizer

    "NepsPriMO-all": "PriMO(all)",
    "NepsPriMO-good-good": "PriMO(good-good)",
    "NepsPriMO-bad-bad": "PriMO(bad-bad)",
    "NepsPriMO-bad-good": "PriMO(bad-good)",

    # Optimizers without Priors

    "SMAC_ParEGO": "ParEGO",
    "RandomSearch": "Random Search",
    "NepsRW": "BO+RW",
    "Nevergrad_EvolutionStrategy": "NSGA-II",

    "NepsHyperbandRW": "HB+RW",

    "NepsMOASHA": "MOASHA",

    "NepsMOASHA_RS": "MOASHA+RS",

    "NepsMOASHABO": "MOASHA-BO+RW",

    "MOMFBO": "MOMF-BO",

    # Optimizers with Priors

    "NepsPriorMOASHA-all": "MOASHA+Prior(all)",
    "NepsPriorMOASHA-good-good": "MOASHA+Prior(good-good)",
    "NepsPriorMOASHA-bad-bad": "MOASHA+Prior(bad-bad)",
    "NepsPriorMOASHA-bad-good": "MOASHA+Prior(bad-good)",

    "NepsMOPriorband-all": "MO-Priorband(all)",
    "NepsMOPriorband-good-good": "MO-Priorband(good-good)",
    "NepsMOPriorband-bad-bad": "MO-Priorband(bad-bad)",
    "NepsMOPriorband-bad-good": "MO-Priorband(bad-good)",

    "NepsPiBORW-all": "πBO+RW(all)",
    "NepsPiBORW-good-good": "πBO+RW(good-good)",
    "NepsPiBORW-bad-bad": "πBO+RW(bad-bad)",
    "NepsPiBORW-bad-good": "πBO+RW(bad-good)",

    "RandomSearchWithPriors-all": "RS+Prior(all)",
    "RandomSearchWithPriors-good-good": "RS+Prior(good-good)",
    "RandomSearchWithPriors-good-bad": "RS+Prior(good-bad)",
    "RandomSearchWithPriors-good-medium": "RS+Prior(good-medium)",
    "RandomSearchWithPriors-good-None": "RS+Prior(good-None)",
    "RandomSearchWithPriors-None-good": "RS+Prior(None-good)",
    "RandomSearchWithPriors-bad-good": "RS+Prior(bad-good)",
    "RandomSearchWithPriors-bad-bad": "RS+Prior(bad-bad)",
    "RandomSearchWithPriors-bad-medium": "RS+Prior(bad-medium)",
    "RandomSearchWithPriors-bad-None": "RS+Prior(bad-None)",
    "RandomSearchWithPriors-None-bad": "RS+Prior(None-bad)",
    "RandomSearchWithPriors-medium-good": "RS+Prior(medium-good)",
    "RandomSearchWithPriors-medium-bad": "RS+Prior(medium-bad)",
    "RandomSearchWithPriors-medium-medium": "RS+Prior(medium-medium)",
    "RandomSearchWithPriors-medium-None": "RS+Prior(medium-None)",
    "RandomSearchWithPriors-None-medium": "RS+Prior(None-medium)",
}