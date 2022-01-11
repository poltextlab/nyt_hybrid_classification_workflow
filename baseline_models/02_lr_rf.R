# Baseline classification

library(dplyr)
library(readr)
library(quanteda)
library(quanteda.textmodels)
library(glmnet)
library(caret)
library(randomForest)
library(e1071)
library(janitor)
library(tibble)
library(stringr)



set.seed(1830)

nyt_raw <- read_delim("../data/Data_NYT_clean_SPARK_START_sim.csv", delim = ";") %>%
  mutate(majortopic = ifelse(majortopic > 21, 100, majortopic))


nyt_corpus <- corpus(nyt_raw)


smp <- sample(c("train", "test"), size = ndoc(nyt_corpus),
              prob = c(0.80, 0.20), replace = TRUE)
train <- which(smp == "train")
test <- which(smp == "test")

nyt_dfm <- nyt_corpus %>%
  tokens(remove_punct = TRUE, remove_number = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("en")) %>%
  tokens_wordstem() %>%
  dfm()


# Generalized linear regression
# Multinomial regression with the LASSO discriminant
lasso <- cv.glmnet(x = nyt_dfm[train, ],
                   y = docvars(nyt_corpus, "majortopic")[train],
                   alpha = 1,
                   nfolds = 3,
                   family = "multinomial"
                   )



predicted_class_lasso <- as.integer(predict(lasso, nyt_dfm[test, ], type = "class"))


cm_lasso <- table(predicted_class_lasso, docvars(nyt_corpus, "majortopic")[test])


cm_lasso


confusionMatrix(cm_lasso, mode = "everything")



# random forest
X <- as.matrix(dfm_trim(nyt_dfm, min_docfreq = 50,
                        max_docfreq = 0.80 * nrow(nyt_dfm), verbose = TRUE))


rf <- randomForest(x = as.matrix(nyt_dfm[train, ]),
                   y = factor(docvars(nyt_corpus, "majortopic")[train]),
                   xtest = as.matrix(nyt_dfm[test, ]),
                   ytest = factor(docvars(nyt_corpus, "majortopic")[test]),
                   importance = TRUE,
                   mtry = 20,
                   ntree = 100
                   )



p1 <- as.numeric(predict(rf, X[test, ]))

actual <- docvars(nyt_corpus, "majortopic")[test]

confusionMatrix(factor(p1), factor(actual))



# svm
svm_nyt <- textmodel_svm(nyt_dfm[train, ], y = docvars(nyt_corpus, "majortopic")[train])

predicted_svm <- predict(svm_nyt, newdata = nyt_dfm[test, ])

cm_svm <- table(predicted_svm, docvars(nyt_corpus, "majortopic")[test])


confusionMatrix(cm_svm, mode = "everything")



# svm setup (identical to the nb python script)
sim <- sort(unique(nyt_raw$sim))

svm_models <- vector(mode = "list", length = length(sim))

for (i in 1:length(sim)) {

  # split training and test set
  training_dfm <- nyt_dfm[nyt_dfm$sim != sim[i], ]
  test_dfm <- nyt_dfm[nyt_dfm$sim == sim[i], ]

#   # start fitting the svm
  print(paste(Sys.time(), ">>>", "SVM with sim =", i, "as test set started"))

  svm_fit <- svm(x = training_dfm, y = factor(docvars(training_dfm, "majortopic")),
                 kernel = "linear", cost = 0.1)

#   # store the fitted model
  svm_models[[i]] <- svm_fit

  print(paste(Sys.time(), ">>>", "SVM with sim =", i, "as test set complete"))

}


predictions <- vector("list", length = length(sim))


predictions <- lapply(sim, function(sim_nr) {
  prediction <- predict(svm_models[[sim_nr]], nyt_dfm[nyt_dfm$sim == sim[sim_nr], ])
  cm <- confusionMatrix(table(predictions[[sim_nr]], docvars(nyt_dfm[nyt_dfm$sim == sim[sim_nr], ], "majortopic")), mode = "everything")
  output <- as.data.frame(cm$byClass[, c("Precision", "Recall")])

  return(output)
})


svm_pr <- bind_cols(predictions) %>%
  clean_names() %>%
  rownames_to_column(var = "verdict") %>%
  mutate(verdict = as.numeric(str_remove(verdict, "Class: "))) %>%
  rowwise() %>%
  mutate(precision_svm = sum(precision_1, precision_3, precision_5) / 3,
         recall_svm = sum(recall_2, recall_4, recall_6) / 3) %>%
  select(verdict, ends_with("_svm"))
