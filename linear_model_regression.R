# Helper packages
library(dplyr)    # for data manipulation
library(ggplot2)  # for awesome graphics
library(modelsummary)
library(readxl)
# Feature engineering packages
library(sjPlot)
library(MuMIn)
library(readxl) 
library(performance)
library(vip)      # variable importance
library(patchwork)
library(randomForest)
library(lme4)
library(datawizard) # for standardization
library(lmerTest)
library(purrr)

# load data ---------

df_raw0 = read_xlsx('tables/df_estbar_local_merged.xlsx')
df_raw0$Study = as.factor(df_raw0$Study)
df_raw0$model = as.factor(df_raw0$model)
df_raw0$Nlimitation_strategy = as.factor(df_raw0$Nlimitation_strategy)

strategy ='N-Retention'
modelname = 'PWOV'
df_raw0 <- df_raw0 %>% rename(CN = `C:N`)

col_to_keep =c('Study','Nlimitation_strategy','model','Climate', 'Csource', 'MATC', 'CN', 'carbohydrate_MMM',
               'protein_MMM', 'lignin_MMM', 'lipid_MMM', 'carbonyl_MMM')



targets = c('vh_max', 'vp_max', 'vlig', 'vlip', 'vCr')


# Prepare dataset
df0 <- df_raw0 %>%
  select(all_of(c(col_to_keep, targets))) %>%
  na.omit() %>%
  mutate(across(-c(Study, Nlimitation_strategy, model, all_of(targets)),
                ~ standardize(.x))) # column-wise standardization selectively for all column except Study, Nlimitation_strategy, model, all_of(targets)

df0 <- df0 %>%
  mutate(across(
    all_of(targets),
    ~ log10(.x + 1e-6),  # add small offset to avoid log(0)
    .names = "{.col}_log"  # create new column with _log suffix
  ))

targets = c('vh_max_log', 'vp_max_log', 'vlig_log', 'vlip_log', 'vCr_log')

# fit lmer or all are targets with model scenario and study as random effects ---------
# --- Function: fit + reduce for one target on full dataset ---
fit_and_reduce_lmer <- function(target, df) {
  
  # --- Full model with both random effects ---
  formula_full <- as.formula(
    paste0(target, " ~ (lignin_MMM + carbohydrate_MMM + protein_MMM + lipid_MMM + carbonyl_MMM + CN + MATC)^2 + (1|Study) + (1|model)")
  )
  # formula_full <- as.formula(
  #   paste0(target, " ~  (carbohydrate_MMM+lignin_MMM)^2 + (protein_MMM+lignin_MMM)^2 +carbonyl_MMM  + lipid_MMM + MATC + (1|Study) + (1|model)")
  # )
  fit <- lmer(formula_full, data = df, REML = FALSE)
  
  # --- Backward elimination of fixed effects ---
  repeat {
    anova_tab <- anova(fit, type = 3)
    pvals <- anova_tab$`Pr(>F)`
    names(pvals) <- rownames(anova_tab)
    pvals <- pvals[!is.na(pvals)]
    
    if (length(pvals) == 0 || all(pvals < 0.05)) break
    
    worst_term <- names(which.max(pvals))
    if (pvals[worst_term] < 0.05) break
    
    reduced_formula <- update.formula(formula(fit), paste(". ~ . -", worst_term))
    fit <- lmer(reduced_formula, data = df, REML = FALSE)
  }
  
  # --- Check if random effects are significant ---
  random_effects <- c("Study", "model")
  for (re in random_effects) {
    if (re %in% names(ranef(fit))) {
      # Fit reduced model without this random effect
      reduced_formula <- update.formula(formula(fit), paste(". ~ . - (1|", re, ")", sep = ""))
      fit_reduced <- lmer(reduced_formula, data = df, REML = FALSE)
      
      # Likelihood ratio test
      test <- anova(fit, fit_reduced)
      pval_re <- test$`Pr(>Chisq)`[2]
      
      if (!is.na(pval_re) && pval_re > 0.05) {
        fit <- fit_reduced  # drop random effect if not significant
      }
    }
  }
  
  return(fit)
}
# --- Fit one model per target on full dataset ---
models_list <- map(targets, ~ fit_and_reduce_lmer(.x, df0)) %>% 
  set_names(targets)

# --- Summarize in modelsummary ---
modelsummary(models_list,
             fmt = 3,
             estimate = "{estimate} ({std.error}){stars}",
             statistic = NULL,
             gof_omit = "ICC|RMSE")

# --- Export to Excel ---

modelsummary(models_list,
             fmt = 3,
             estimate = "{estimate} ({std.error}){stars}",
             statistic = NULL,
             gof_omit = "ICC|RMSE",
             output = "regression_table.xlsx")


  
  
  
  