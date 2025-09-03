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

cols <- c(col_to_keep, targets)
df0 <- df_raw0 %>% dplyr::select(all_of(cols))
df0 <- na.omit(df0)

df <- df0 %>%
  mutate(across(-c(vh_max, Study,Nlimitation_strategy,model), ~ datawizard::standardize(.x)))


fit1= lmer(vh_max ~ (MATC + CN + carbohydrate_MMM +
                       lignin_MMM + lipid_MMM + carbonyl_MMM)^2 + (1|Study),
           data = df)

fit2= lmer(vh_max ~ (MATC + CN + carbohydrate_MMM +
                       lignin_MMM + lipid_MMM + carbonyl_MMM)^2 + (1|Study) + (1 | model),
           data = df)

summary(fit1)
modelsummary(list(fit1, fit2), fmt = 4, estimate = "{estimate} ({std.error}){stars}",
             statistic = NULL, gof_omit = 'ICC|RMSE'
             # output = "regression_table_R1.docx"
)



# Prepare dataset
df0 <- df_raw0 %>%
  select(all_of(c(col_to_keep, targets))) %>%
  na.omit() %>%
  mutate(across(-c(Study, Nlimitation_strategy, model, all_of(targets)),
                ~ standardize(.x)))

# fit lmer or all are targets with model scenario and study as random effects ---------
# --- Function: fit + reduce for one target on full dataset ---
fit_and_reduce_lmer <- function(target, df) {
  
  # --- Full model with both random effects ---
  formula_full <- as.formula(
    paste0(target, " ~ (MATC + CN + carbohydrate_MMM + lignin_MMM + lipid_MMM + carbonyl_MMM)^2 + (1|Study) + (1|model)")
  )
  
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
library(modelsummary)
modelsummary(models_list,
             fmt = 3,
             estimate = "{estimate} ({std.error}){stars}",
             statistic = NULL,
             gof_omit = "ICC|RMSE")
# modelsummary(models_list,
#              fmt = 3,
#              estimate = "{estimate} ({std.error}){stars}",
#              statistic = NULL,
#              gof_omit = "ICC|RMSE",
#              output = "regression_table.xlsx")
#   
  
  
  
  
  # fit separate lmer for each target and model scenario ---------
  
  library(purrr)
  
  # --- Function: fit + reduce for one target within one model group ---
  fit_and_reduce_lmer <- function(target, model_group) {
    
    df_sub <- df0 %>% filter(model == model_group)
    
    formula_str <- paste0(
      target, " ~ (MATC + CN + carbohydrate_MMM + lignin_MMM + lipid_MMM + carbonyl_MMM)^2 + ",
      "(1|Study)"
    )
    formula_obj <- as.formula(formula_str)
    
    # fit full model
    fit_full <- lmer(formula_obj, data = df_sub)
    current_fit <- fit_full
    
    # backward elimination of non-significant fixed effects
    repeat {
      anova_tab <- anova(current_fit, type = 3)
      pvals <- anova_tab$`Pr(>F)`
      names(pvals) <- rownames(anova_tab)
      pvals <- pvals[!is.na(pvals)]
      
      if (length(pvals) == 0 || all(pvals < 0.05)) break
      
      worst_term <- names(which.max(pvals))
      if (pvals[worst_term] < 0.05) break
      
      reduced_formula <- update.formula(formula(current_fit), paste(". ~ . -", worst_term))
      current_fit <- lmer(reduced_formula, data = df_sub)
    }
    
    return(current_fit)
  }
  
  # --- Loop over targets and model groups ---
  models_nested <- map(targets, function(tgt) {
    map(unique(df0$model), function(mcat) {
      fit_and_reduce_lmer(tgt, mcat)
    }) %>% set_names(unique(df0$model))
  }) %>% set_names(targets)
  
  # You now have models_nested[target][model_group]
  
  # --- Flatten for modelsummary ---
  models_list <- flatten(models_nested)
  
  # give readable names like "target_modelgroup"
  names(models_list) <- unlist(map(names(models_nested), function(tgt) {
    paste0(tgt, "_", names(models_nested[[tgt]]))
  }))
  
  # --- Summarize all models ---
modelsummary(models_list,
             fmt = 3,
             estimate = "{estimate} ({std.error}){stars}",
             statistic = NULL,
             gof_omit = "ICC|RMSE",
             output = "regression_table_R2.docx")
  
  
  ## visualization type 1: lmer for each target and model scenario ----------
  
  library(broom.mixed)
  
  # --- Extract coefficients ---
  coef_df <- map_dfr(names(models_list), function(nm) {
    broom.mixed::tidy(models_list[[nm]], effects = "fixed", conf.int = TRUE) %>%
      filter(term != "(Intercept)") %>%
      mutate(model_name = nm)
  })
  
  
  # --- Split into target + model_group (last "_" rule) ---
  coef_df <- coef_df %>%
    mutate(
      target = sub("_(?!.*_).*", "", model_name, perl = TRUE),
      model_group = sub(".*_", "", model_name)
    )
  
  # --- Reorder targets in your specified sequence ---
  targets_order <- c("vh_max", "vp_max", "vlig", "vlip", "vCr")
  coef_df <- coef_df %>%
    mutate(target = factor(target, levels = targets_order))
  
  # --- Dot-and-whisker plot ---
  p <- ggplot(coef_df,
              aes(x = term, y = estimate,
                  ymin = conf.low, ymax = conf.high,
                  color = estimate > 0)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey60") +
    geom_pointrange(position = position_dodge(width = 0.6), show.legend = FALSE) +
    facet_grid(rows = vars(model_group), cols = vars(target),
               scales = "free_x", space = "free_x") +
    labs(x = "Predictor", y = "Coefficient estimate",
         title = "Predictor effects across targets and model groups") +
    theme_bw(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.margin = ggplot2::margin(t = 10, r = 10, b = 10, l = 30) # enlarge bottom margin
    )
  
p
# # --- Save high-res PNG ---
# ggsave("coef_plot.png", p,
#        width = 19, height = 10, dpi = 600)
# 
# # --- Save high-res PDF (vector, best for publications) ---
# ggsave("coef_plot.pdf", p,
#        width = 19, height = 10, device = cairo_pdf)


  
  
  
  
  