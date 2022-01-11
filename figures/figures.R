# figures and tables for the article

# library ----
library(ggplot2)
library(dplyr)
library(tidyr)
library(tibble)
library(xtable)
library(stringr)
library(readr)
library(patchwork)
library(forcats)
library(caret)
library(janitor)

# helpers ----

# function that is used to get the verdict and fp and np from the nb_basic results
# uses the raw imported df as input
convert_nb_mtc <- function(df) {
  df_conv <- df %>%
    mutate(verdict = case_when(nbPrediction == 0 ~ 1,
                               nbPrediction == 1 ~ 2,
                               nbPrediction == 2 ~ 3,
                               nbPrediction == 3 ~ 4,
                               nbPrediction == 4 ~ 5,
                               nbPrediction == 5 ~ 6,
                               nbPrediction == 6 ~ 7,
                               nbPrediction == 7 ~ 8,
                               nbPrediction == 8 ~ 9,
                               nbPrediction == 9 ~ 10,
                               nbPrediction == 10 ~ 12,
                               nbPrediction == 11 ~ 13,
                               nbPrediction == 12 ~ 14,
                               nbPrediction == 13 ~ 15,
                               nbPrediction == 14 ~ 16,
                               nbPrediction == 15 ~ 17,
                               nbPrediction == 16 ~ 18,
                               nbPrediction == 17 ~ 19,
                               nbPrediction == 18 ~ 20,
                               nbPrediction == 19 ~ 21,
                               nbPrediction == 20 ~ 100),
           correct = if_else(majortopic == verdict, 1, 0),
           classified = if_else(verdict >= 1, "classified", "not classified"),
           fp = if_else(correct == 0 & classified == "classified", 1, 0),
           tp = if_else(correct == 1 & classified == "classified", 1, 0))
  return(df_conv)
}



# calculate the item-wise precision and recall scores for the ml results (both nb and svm)
# in case of base_np, it should use the output of the convert_nb_mtc function
# otherwise takes the results df as input
get_pr <- function(df) {
  
  pr_table <- df %>%
    group_by(verdict) %>%
    summarise(fp = sum(fp), tp = sum(tp), n_verdict =  n(), sim = unique(sim)) %>%
    ungroup() %>% 
    drop_na() %>% 
    bind_cols(ungroup(summarise(group_by(df, majortopic), n_mtc = n()))) %>% 
    mutate(precision = tp / (fp + tp),
           recall = tp  / n_mtc) %>% 
    select(sim, verdict, precision, recall)
  
  return(pr_table)
  
}

# data import ---- 

nyt_clean <- read_delim("../data/Data_NYT_clean_SPARK_START_sim.csv", delim = ";") %>% 
  mutate(mtc_long = case_when(majortopic == 1 ~ "Macroeconomics",
                              majortopic ==  2 ~ "Civil Rights",
                              majortopic ==  3 ~ "Health",
                              majortopic ==  4 ~ "Agriculture",
                              majortopic == 5 ~ "Labor",
                              majortopic ==  6 ~ "Education",
                              majortopic ==  7 ~ "Environment",
                              majortopic == 8 ~ "Energy",
                              majortopic ==  9 ~ "Immigration",
                              majortopic ==  10 ~ "Transportation",
                              majortopic ==  12 ~ "Law and Crime",
                              majortopic ==  13 ~ "Social Welfare",
                              majortopic == 14 ~ "Housing",
                              majortopic ==  15 ~ "Domestic Commerce",
                              majortopic ==  16 ~ "Defense",
                              majortopic ==  17 ~ "Technology",
                              majortopic == 18 ~ "Foreign Trade",
                              majortopic ==  19 ~ "International Affairs",
                              majortopic ==  20 ~ "Government Operations",
                              majortopic == 21 ~ "Public Lands",
                              majortopic ==  23 ~ "Culture",
                              majortopic >  23 ~ "Other"),
         majortopic = if_else(majortopic > 23, 100, majortopic),
         category = as.factor(paste(majortopic, mtc_long, sep = ". ")))



# category labels for the plots below
cat_labels <- nyt_clean %>% 
  select(verdict = majortopic, category) %>% 
  distinct()





# topics per sim table -----

sim_rounds <- c(1:3)

a1_table <- as.data.frame(lapply(sim_rounds, function(x) {
  
    nyt_clean %>% 
    filter(sim == x) %>% 
    group_by(sim, majortopic) %>% 
    summarise(sim_name = n()) %>% 
    ungroup() %>% 
    select(majortopic, sim_name) %>% 
    setNames(c("majortopic", paste("sim", x, sep = "_")))
  
  
})) %>% 
  select(category, starts_with("sim_")) 
  

# save a1 table as tex table

print.xtable(xtable(a1_table), file = "./a1_tab.tex")




# alternative representation
nyt_clean %>% 
  group_by(sim, majortopic) %>% 
  summarise(n = n()) %>% 
  ggplot(aes(as.factor(majortopic), n, fill = as.factor(sim))) +
  geom_col(position = "dodge") +
  coord_flip() +
  scale_color_grey() +
  theme_bw()




# TOPIC DISTRIBUTION FIG ----
nyt_clean %>% 
  group_by(category) %>% 
  summarise(n_articles = n()) %>% 
  ungroup() %>% 
  ggplot(aes(reorder(category, n_articles), n_articles)) +
  scale_y_continuous(breaks = seq(from = 0, to = 6000, by = 1000)) +
  geom_col(color = "white", fill = "#333333") +
  labs(x = NULL,
       y = "Number of articles") +
  coord_flip() +
  theme_minimal() +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank())

ggsave(filename = "./b1_fig.png", dpi = "retina", width = 7, height = 4, units = "in")


# NB VS SMV 3x3  ----




# svm
svm_v1 <- read_csv("../data/output/validation_compare12to3.csv") %>% 
  mutate(fp = if_else(correct == 0 & classified == "classified", 1, 0),
         tp = if_else(correct == 1 & classified == "classified", 1, 0))

svm_v2 <- read_csv("../data/output/validation_compare13to2.csv") %>%
  mutate(fp = if_else(correct == 0 & classified == "classified", 1, 0),
         tp = if_else(correct == 1 & classified == "classified", 1, 0))

svm_v3 <- read_csv("../data/output/validation_compare23to1.csv") %>% 
  mutate(fp = if_else(correct == 0 & classified == "classified", 1, 0),
         tp = if_else(correct == 1 & classified == "classified", 1, 0))

# svm total
svm_total_pr <- svm_v1 %>%
  bind_rows(svm_v2) %>% 
  bind_rows(svm_v3) %>% 
  mutate(fp = if_else(correct == 0 & classified == "classified", 1, 0),
         tp = if_else(correct == 1 & classified == "classified", 1, 0))

# precision
sum(svm_total_pr$tp) / (sum(svm_total_pr$tp) + sum(svm_total_pr$fp)) * 100 

# recall
sum(svm_total_pr$tp) / nrow(svm_total_pr) * 100

# precision and recall by item
svm_pr <- bind_cols(get_pr(svm_v1), get_pr(svm_v2)[, c(3:4)], get_pr(svm_v3)[, c(3:4)])  %>% 
  select(verdict, starts_with("prec"), starts_with("rec")) %>% 
  clean_names() %>% 
  rowwise() %>% 
  mutate(precision_svm = sum(precision_3, precision_5, precision_7) / 3,
         recall_svm = sum(recall_4, recall_6, recall_8) / 3) %>% 
  ungroup() %>% 
  select(verdict, ends_with("_svm"))


# nb
nb_v1 <- read_csv("../data/output/validation_compare12to3_nb.csv") %>% 
  mutate(fp = if_else(correct == 0 & classified == "classified", 1, 0),
         tp = if_else(correct == 1 & classified == "classified", 1, 0))

nb_v2 <- read_csv("../data/output/validation_compare13to2_nb.csv") %>% 
  mutate(fp = if_else(correct == 0 & classified == "classified", 1, 0),
         tp = if_else(correct == 1 & classified == "classified", 1, 0))

nb_v3 <- read_csv("../data/output/validation_compare23to1_nb.csv") %>% 
  mutate(fp = if_else(correct == 0 & classified == "classified", 1, 0),
         tp = if_else(correct == 1 & classified == "classified", 1, 0))


nb_pr <- bind_cols(get_pr(nb_v1), get_pr(nb_v2)[, c(3:4)], get_pr(nb_v3)[, c(3:4)]) %>% 
  select(verdict, starts_with("prec"), starts_with("rec")) %>% 
  clean_names() %>% 
  rowwise() %>% 
  mutate(precision_nb = sum(precision_3, precision_5, precision_7) / 3,
         recall_nb = sum(recall_4, recall_6, recall_8) / 3) %>% 
  ungroup() %>% 
  select(verdict, ends_with("_nb"))


pr_df <- left_join(svm_pr, nb_pr) %>% 
  left_join(cat_labels) %>% 
  select(category, verdict, starts_with("prec")) %>% 
  pivot_longer(cols = "precision_svm":"precision_nb", names_to = "prec_type", values_to = "score_prec")


rec_df <- left_join(svm_pr, nb_pr) %>% 
  left_join(cat_labels) %>% 
  select(category, verdict, starts_with("rec")) %>% 
  pivot_longer(cols = "recall_svm":"recall_nb", names_to = "rec_type", values_to = "score_rec") 


pr_combo_df <- bind_cols(pr_df, rec_df[, c(3:4)])

  
  
highlight_pr <- pr_combo_df %>% 
  filter(score_prec >= 0.8)

highlight_rc <- pr_combo_df %>% 
  filter(score_rec >= 0.5)


pr_versus <- ggplot(pr_combo_df, aes(reorder(category, score_prec), score_prec)) +
  geom_line(aes(group = category), alpha = 0.3) +
  geom_point(aes(shape = prec_type), size = 2, alpha = 0.3, fill = 'darkgrey') +
  geom_line(data = highlight_pr, aes(group = category)) +
  geom_point(data = highlight_pr, aes(shape = prec_type), size = 2, fill = 'darkgrey') +
  geom_hline(yintercept = 0.8, alpha = 0.3, linetype = "dashed") +
  scale_color_manual(values = c("#999999", "#333333"), labels = c("NB", "SVM")) +
  scale_shape_manual(values = c(21, 23), labels = c("NB", "SVM")) +
  scale_y_continuous(breaks = seq(from = 0, to = 1, by = 0.1)) +
  expand_limits(y = c(0, 1)) +
  guides(color = guide_legend(), shape = guide_legend()) +
  coord_flip() +
  labs(x = NULL,
       y = "Precision") +
  theme_minimal() +
  theme(legend.position = c(0.9, 0.2),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        legend.title = element_blank(),
        axis.text = element_text(size = 8))





rec_versus <- ggplot(pr_combo_df, aes(reorder(category, score_rec), score_rec)) +
  geom_line(aes(group = category), alpha = 0.3) +
  geom_point(aes(shape = rec_type), size = 2, alpha = 0.3, fill = "darkgrey") +
  geom_line(data = highlight_rc, aes(group = category)) +
  geom_point(data = highlight_rc, aes(shape = rec_type), size = 2, fill = "darkgrey") +
  geom_hline(yintercept = 0.5, alpha = 0.3, linetype = "dashed") +
  scale_color_manual(values = c("#999999", "#333333"), labels = c("NB", "SVM")) +
  scale_shape_manual(values = c(21, 23), labels = c("NB", "SVM")) +
  scale_y_continuous(breaks = seq(from = 0, to = 1, by = 0.1)) +
  guides(color = guide_legend(), shape = guide_legend()) +
  expand_limits(y = c(0, 1)) +
  coord_flip() +
  labs(x = NULL,
       y = "Recall") +
  theme_minimal() +
  theme(legend.position = c(0.9, 0.2),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        legend.title = element_blank(),
        axis.text = element_text(size = 8))




pr_versus / rec_versus +
  plot_layout(guides = 'collect') +
  plot_annotation(
    title = NULL,
    subtitle = NULL
  ) &
  theme(plot.tag = element_text(size = 12),
        legend.position = 'bottom')

ggsave("./sml_comparison.png", dpi = "retina", width = 6.5, height = 6, units = "in")



# PR CURVE FIG ----
# precision recall curve

nyt_sumtable <- read_csv("../data/output/summary_of_num_samples_average.csv")

prec_sum <- nyt_sumtable[1, 2:11] %>% 
  pivot_longer(`1x1`:`10x10`, names_to = "rounds", values_to = "precision")

rec_sum <- nyt_sumtable[2, 2:11] %>% 
  pivot_longer(`1x1`:`10x10`, names_to = "rounds", values_to = "recall")


pr_flipped <- left_join(prec_sum, rec_sum) %>% 
  mutate(rowid = row_number(),
         plot_label = if_else(rowid == 3, 1, 0)) %>% 
  ggplot(aes(x = precision, y = recall)) +
  geom_line() +
  geom_point(size = 2) +
  geom_text(aes(label = if_else(plot_label == 1, rounds, NULL), vjust = -1.5)) +
  labs(title = NULL,
       subtitle = NULL) +
  theme_minimal()

pr_flipped


ggsave(pr_flipped, "./pr_plot_flipped.jpg")


# BASIC comparison----
# svm
svm_pr <- read_csv("../data/output/svm_pr.csv")


round(mean(svm_pr$precision_svm), 2)
round(mean(svm_pr$recall_svm), 2)

# lasso
cm_lasso <- read_rds("../data/lasso_cm.rds")


eval_lasso <- confusionMatrix(cm_lasso, mode = "everything")


lasso_pr <- as.data.frame(eval_lasso$byClass) %>% 
  select(precision_lasso = Precision, recall_lasso = Recall) %>% 
  rownames_to_column(var = "majortopic") %>% 
  mutate(verdict = as.numeric(str_remove(majortopic, "Class: "))) %>% 
  select(-majortopic)

round(mean(lasso_pr$precision_lasso), 2)
round(mean(lasso_pr$recall_lasso), 2)


# random forest
rf <- read_rds("../data/rf.rds")

eval_rf <- confusionMatrix(rf$test$confusion[1:21, 1:21])

rf_pr <- as.data.frame(eval_rf$byClass) %>% 
  select(precision_rf = Precision, recall_rf = Recall) %>% 
  rownames_to_column(var = "majortopic") %>% 
  mutate(verdict = as.numeric(str_remove(majortopic, "Class: "))) %>% 
  select(-majortopic)

round(mean(rf_pr$precision_rf), 2)
round(mean(rf_pr$recall_rf), 2)


# naive bayes
nb_basic1 <- read_csv("../data/output/NB_basic_compare12to3.csv") %>% 
  convert_nb_mtc()

nb_basic2 <- read_csv("../data/output/NB_basic_compare13to2.csv") %>% 
  convert_nb_mtc()

nb_basic3 <- read_csv("../data/output/NB_basic_compare32to1.csv") %>% 
  convert_nb_mtc()

nb_basic_df <- bind_cols(get_pr(nb_basic1), get_pr(nb_basic2)[, c(3:4)], get_pr(nb_basic3)[, c(3:4)]) %>% 
  select(verdict, starts_with("prec"), starts_with("rec")) %>% 
  clean_names() %>% 
  rowwise() %>% 
  mutate(precision_nb = sum(precision_3, precision_5, precision_7) / 3,
         recall_nb = sum(recall_4, recall_6, recall_8) / 3) %>% 
  ungroup() %>% 
  select(verdict, ends_with("_nb")) %>% 
  left_join(cat_labels)


round(mean(nb_basic_df$precision_nb), 2)
round(mean(nb_basic_df$recall_nb), 2)


basic_pr <- nb_basic_df %>% 
  left_join(rf_pr) %>% 
  left_join(lasso_pr) %>% 
  left_join(svm_pr)



basic_pr %>%
  select(category, contains("_")) %>% 
  bind_rows(colMeans(basic_pr[c(2:9)])) %>% 
  mutate(across(c(2:9), round, 2)) %>% 
  write_csv("../data/output/basic_pr.csv")

basic_pr_df <- basic_pr %>% 
  select(category, verdict, starts_with("prec")) %>% 
  pivot_longer(cols = "precision_nb":"precision_svm", names_to = "prec_type", values_to = "score_prec")


basic_rec_df <- basic_pr %>% 
  select(category, verdict, starts_with("rec")) %>% 
  pivot_longer(cols = "recall_nb":"recall_svm", names_to = "rec_type", values_to = "score_rec") 


highlight_b_pr <- basic_pr_df %>% 
  filter(score_prec >= 0.8)

highlight_b_rc <- basic_rec_df %>% 
  filter(score_rec >= 0.5)

########################
ggplot(basic_pr_df, aes(reorder(category, score_prec), score_prec)) +
  #geom_line(aes(group = category), alpha = 0.3) +
  geom_point(aes(shape = prec_type), size = 2, alpha = 0.3, fill = 'darkgrey') +
  #geom_line(data = highlight_b_pr, aes(group = category)) +
  geom_point(data = highlight_b_pr, aes(shape = prec_type), size = 2, fill = 'darkgrey') +
  geom_hline(yintercept = 0.8, alpha = 0.3, linetype = "dashed") +
  #scale_color_manual(values = c("#999999", "#333333"), labels = c("NB", "SVM")) +
  #scale_shape_manual(values = c(21, 23), labels = c("NB", "SVM")) +
  scale_y_continuous(breaks = seq(from = 0, to = 1, by = 0.1)) +
  expand_limits(y = c(0, 1)) +
  guides(color = guide_legend(), shape = guide_legend()) +
  coord_flip() +
  labs(x = NULL,
       y = "Precision") +
  theme_minimal() +
  theme(legend.position = 'bottom',
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        legend.title = element_blank(),
        axis.text = element_text(size = 8))





ggplot(basic_rec_df, aes(reorder(category, score_rec), score_rec)) +
  #geom_line(aes(group = category), alpha = 0.3) +
  geom_point(aes(shape = rec_type), size = 2, alpha = 0.3, fill = 'darkgrey') +
  #geom_line(data = highlight_b_pr, aes(group = category)) +
  geom_point(data = highlight_b_rc, aes(shape = rec_type), size = 2, fill = 'darkgrey') +
  geom_hline(yintercept = 0.5, alpha = 0.3, linetype = "dashed") +
  #scale_color_manual(values = c("#999999", "#333333"), labels = c("NB", "SVM")) +
  #scale_shape_manual(values = c(21, 23), labels = c("NB", "SVM")) +
  scale_y_continuous(breaks = seq(from = 0, to = 1, by = 0.1)) +
  expand_limits(y = c(0, 1)) +
  guides(color = guide_legend(), shape = guide_legend()) +
  coord_flip() +
  labs(x = NULL,
       y = "Precision") +
  theme_minimal() +
  theme(legend.position = c(0.9, 0.2),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        legend.title = element_blank(),
        axis.text = element_text(size = 8))





########################

nb_basic_pr <- ggplot(nb_basic_df, aes(y = reorder(category, precision_nb), x = precision_nb)) +
  geom_segment(aes(x = 0, y = reorder(category, precision_nb), xend = precision_nb, yend = category), color = '#999999') +
  geom_point(size = 2, color = "#333333") +
  scale_x_continuous(breaks = seq(from = 0, to = 1, by = 0.1)) +
  expand_limits(x = c(0, 1)) +
  labs(y = NULL,
       x = "Precision") +
  theme_minimal() +
  theme(legend.position = c(0.9, 0.2),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        legend.title = element_blank(),
        axis.text = element_text(size = 8))




nb_basic_rec <- ggplot(nb_basic_df, aes(y = reorder(category, recall_nb), x = recall_nb)) +
  geom_segment(aes(x = 0, y = reorder(category, recall_nb), xend = recall_nb, yend = category), color = '#999999') +
  geom_point(size = 2, color = "#333333") +
  scale_x_continuous(breaks = seq(from = 0, to = 1, by = 0.1)) +
  expand_limits(x = c(0, 1)) +
  labs(y = NULL,
       x = "Recall") +
  theme_minimal() +
  theme(legend.position = c(0.9, 0.2),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        legend.title = element_blank(),
        axis.text = element_text(size = 8))



nb_basic_pr / nb_basic_rec +
  plot_annotation(
    title = NULL,
    tag_levels = 'A'
  ) &
  theme(plot.tag = element_text(size = 12),
        legend.position = 'bottom')

ggsave("./nb_baseline.png", dpi = "retina", width = 6.5, height = 6, units = "in")


# ad-hoc figures for the article discussion
rec_df_vs <- left_join(svm_pr, nb_pr) %>% 
  left_join(cat_labels) %>% 
  select(category, verdict, starts_with("rec")) %>% 
  mutate(svm_win = if_else(recall_svm > recall_nb, 1, 0))

sum(rec_df_vs$svm_win)


pr_df_vs <- left_join(svm_pr, nb_pr) %>% 
  left_join(cat_labels) %>% 
  select(category, verdict, starts_with("prec")) %>% 
  mutate(svm_win = if_else(precision_svm > precision_nb, 1, 0))

sum(pr_df_vs$svm_win)



sum(pr_df_vs$precision_svm >= .85)

sum(pr_df_vs$precision_svm >= .8)


# HV SUMMARY ----
hv0 <- read_csv("../data/output/hv_v0_summary.csv") %>% 
  select(1, precision = precision_svm, recall = recall_svm) %>% 
  mutate(hv_type = "ml_only")

hv0_totals <- read_csv("../data/output/hv_v0_totals.csv")


hv0 <- bind_rows(hv0, hv0_totals)


hv1 <- read_csv("../data/output/hv_v1_summary.csv") %>% 
  select(1, precision = precision_v1, recall = recall_v1) %>% 
  mutate(hv_type = "basic")

hv2 <- read_csv("../data/output/hv_v2_summary.csv") %>% 
  select(1, precision = precision_v2, recall = recall_v2) %>% 
  mutate(hv_type = "basic_enhanced")

hv3 <- read_csv("../data/output/hv_v3_summary.csv") %>% 
  select(1, precision = precision_v3, recall = recall_v3) %>% 
  mutate(hv_type = "balanced")

hv4 <- read_csv("../data/output/hv_v4_summary.csv") %>% 
  select(1, precision = precision_v4, recall = recall_v4) %>% 
  mutate(hv_type = "balanced_active")



hv_df <- bind_rows(hv0, hv1, hv2, hv3, hv4) %>% 
  mutate(hv_type = as_factor(hv_type),
         hv_type = fct_relevel(hv_type, "ml_only", "basic", "basic_enhanced", "balanced", "balanced_active")) %>% 
  left_join(cat_labels) %>% 
  drop_na()



hv_pr <-  ggplot(hv_df, aes(reorder(category, precision), precision)) +
  geom_point(aes(shape = hv_type, fill = hv_type), size = 2) +
  scale_shape_manual(values = c(21, 22, 23, 24, 25), labels = c("ML only" ,"Basic", "Enhanced Basic", "Balanced", "Balanced + \nActive Learn")) +
  scale_fill_manual(values = c("#999999", "#808080", "#666666", "#333333", "#000000"), labels = c("ML only" ,"Basic", "Enhanced Basic", "Balanced", "Balanced + \nActive Learn")) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(from = 0, to = 1, by = 0.1)) +
  labs(x = NULL,
       y = "Precision") +
  coord_flip() +
  guides(fill = guide_legend(), shape = guide_legend()) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    axis.text = element_text(size = 8)
    )



# highlight above 60pct
highlight_hv_rc <- hv_df %>% 
  filter(recall >= 0.6)

hv_rec <- ggplot(hv_df, aes(reorder(category, recall), recall)) +
  geom_point(aes(shape = hv_type, fill = hv_type), size = 2, alpha = 0.35) +
  geom_point(data = highlight_hv_rc, aes(shape = hv_type, fill = hv_type), colour = "white", size = 2) +
  scale_shape_manual(values = c(21, 22, 23, 24, 25), labels = c("ML only" ,"Basic", "Enhanced Basic", "Balanced", "Balanced + \nActive Learn")) +
  scale_fill_manual(values = c("#999999", "#808080", "#666666", "#333333", "#000000"), labels = c("ML only" ,"Basic", "Enhanced Basic", "Balanced", "Balanced + \nActive Learn")) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(from = 0, to = 1, by = 0.1)) +
  labs(x = NULL,
       y = "Recall") +
  coord_flip() +
  guides(fill = FALSE, shape = FALSE) +
  theme_minimal() +
  theme(
     axis.text = element_text(size = 8)
  )



hv_pr / hv_rec +
  plot_layout(guides = 'collect') +
  plot_annotation(
    title = NULL,
    subtitle = NULL
  ) &
  theme(plot.tag = element_text(size = 12),
        legend.position = 'bottom')

ggsave("./hv_comparison.png", dpi = "retina", width = 6.5, height = 6, units = "in")



# hv table ----

hv_table <- hv_df %>% 
  pivot_wider(names_from = "hv_type", values_from = c("precision", "recall"))


write_csv(hv_table, "./hv_summary_table.csv")


# hv convergence ----
hv_conv <- read_csv("./hv_convergence.csv") %>% 
  mutate(sml_round = as_factor(sml_round),
         sml_round = fct_relevel(sml_round, "5", "4", "3", "2", "1"),
         hv_type = as_factor(hv_type),
         hv_type = fct_relevel(hv_type, "ml_only", "basic", "basic_enhanced", "balanced", "balanced_active"))

ggplot(hv_conv, aes(hv_type, num_classified, fill = sml_round)) +
  geom_col(position = "dodge", color = "white", width = 0.5) +
  scale_y_continuous(breaks = seq(from = 0, to = 7000, by = 1000)) +
  scale_fill_manual(values = c("#000000", "#333333", "#666666", "#808080", "#999999")) +
  labs(x = NULL,
       y = "Number of documents with unitary votes per round") +
  coord_flip() +
  theme_minimal() +
  theme(panel.grid.major.y = element_blank())


ggsave("./hv_rounds.png", dpi = "retina")
