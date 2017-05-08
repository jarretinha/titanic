# Project configuration
source(file.path('/pub/dev/titanic/local_conf.R'))

library(ggplot2)
library(scales)
library(data.table)
library(xgboost)
library(Matrix)
library(caret)
library(mlr)

options(na.action = 'na.pass')
class_threshold <- 0.5

# Data and project files
train <- fread(file.path(data_dir, 'train.csv'))
test <- fread(file.path(data_dir, 'test.csv'))
dataset <- rbindlist(list(train = train, test = test), use.names = T, fill = T, idcol = 'Type')

# Remove useless columns
dataset[ , c('Name', 'Cabin', 'Ticket') := NULL]

# Remove those 2 observations without an origin
dataset <- dataset[Embarked %in% c('Q', 'S', 'C')]

# Transform into factors
categorical_cols <- c('Pclass', 'SibSp', 'Parch', 'Embarked', 'Sex')
dataset[ , (categorical_cols) := lapply(.SD, factor), .SDcols = categorical_cols]

# Add median ages to be used as replacement for NAs
group_cols <- c('Pclass', 'Sex', 'Embarked')
median_age <- dataset[ , list(median_age = median(Age, na.rm = T))
                       , by = group_cols]
dataset <- merge(dataset, median_age, by = group_cols, all.x = T)
dataset[ , Age := ifelse(is.na(Age), median_age, Age)]

# Split train and test
t <- split(dataset, dataset$Type)
train <- copy(t[['train']])
train[ , Type := NULL]
test <- copy(t[['test']])
test[ , Type := NULL]

# Set the model
model_formula <- formula(Survived ~ -1 + Pclass + Sex + Age + Embarked + Fare + SibSp + Parch)

# Prepare datasets
model_data <- sparse.model.matrix(model_formula, data = train)
model_data <- xgb.DMatrix(data = model_data, label = train$Survived)

# Labels
model_labels <- train$Survived

# Model default parameters
params <- list(max_depth = 6
               , eta = 0.3
               , min_child_weight = 1
               , subsample = 1
               , colsample_bytree = 1
               , objective = 'binary:logistic'
               , eval_metric = 'error'
               , gamma = 0)

# Cross validation and gamma optimization
error_diff <- 1
error_threshold <- 0.01
gamma_increment <- 0.5
while (abs(error_diff) > error_threshold) {

  params$gamma <- params$gamma + gamma_increment

  model_cv <- xgb.cv(params = params
                     , data = model_data
                     , label = model_labels
                     , nrounds = 100
                     , showsd = T
                     , nfold = 10
                     , stratified = T
                     , print_every_n = 10
                     , early_stopping_rounds = 20
                     , maximize = F
                     , verbose = 0)

  best <- model_cv$best_iteration
  last_row <- model_cv$evaluation_log[best]
  train_error <- last_row$train_error_mean
  test_error <- last_row$test_error_mean

  error_diff <- train_error - test_error

}

# Train model and tune parameters
model_params <- copy(params)

train_task <- makeClassifTask(data = train, target = 'Survived', positive = '1')
learner <- makeLearner("classif.xgboost", predict.type = "response")
learner$par.vals <- list(objective = "binary:logistic"
                         , eval_metric = "error"
                         , nrounds = 100L
                         , eta = 0.1
                         , early_stopping_rounds = 20
                         , booster = 'gbtree'
                         , silent = 1)

params <- makeParamSet(makeIntegerParam("max_depth" , lower = 3L , upper = 10L)
                       , makeNumericParam("min_child_weight" , lower = 1L,upper = 10L)
                       , makeNumericParam("subsample" , lower = 0.5 , upper = 1)
                       , makeNumericParam("colsample_bytree" , lower = 0.5 , upper = 1)
                       , makeNumericParam("gamma" , lower = 5 , upper = 20))

sampling <- makeResampleDesc("CV", stratify = T, iters = 5)
ctrl <- makeTuneControlRandom(maxit = 10)
fit <- tuneParams(learner = learner
                  , task = train_task
                  , resampling = sampling
                  , measures = acc
                  , par.set = params
                  , control = ctrl
                  , show.info = F)

tuned_params <- setHyperPars(learner, par.vals = fit$x)
tuned_fit <- train(learner = tuned_params, task = train_task)

# Check performance
predictions <- predict(tuned_fit, train_task)
metrics <- confusionMatrix(predictions$data$response, predictions$data$truth, positive = '1')

# Run model
test[ , Survived := 1]
test_task <- makeClassifTask(data = test, target = 'Survived', positive = '1')

predictions <- predict(tuned_fit, test_task)

results <- data.table(PassengerID = test$PassengerId, Survived = predictions$data$response)

fwrite(results, file.path(results_dir, 'results.csv'))







