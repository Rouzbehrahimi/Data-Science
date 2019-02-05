folder_path <- "/Users/Rouzbeh/desktop/IE-BD/R/Group_pro"

load(file=file.path(folder_path,"solar_dataset.Rdata"))

library(readr)
dt <- read_rds("/Users/Niloo/desktop/IE-BD/R/Group_pro/solar_dataset.RData")
dt <-setDT(dt)
dim(dt)
class(dt)
View(dt)
library(data.table)
colnames(dt)[99]
## Solar_dataset has 6909 rows and 456 columns,  PC1 is column 100
######## 1.EDA ########

##### Adding the extra variables to the dataset


######## 1.1Statistic of each column #########

mean_dt <-dt[,sapply(.SD,mean),,.SDcols=-c(1:99)]
min_dt <- dt[,sapply(.SD,min),,.SD=-c(1:99)]
median_dt <- dt[,sapply(dt,median),]
max_dt <- dt[,sapply(.SD,max),,.SD=-c(1:99)]
IQr_dt <- dt[,sapply(.SD,IQR),,.SDcols=-c(1:99)]
sd_dt <- dt[,sapply(dt,sd),]
which(is.na(dt$ACME))  ### this pattern is the same for rows 2:99 



## in columns 2-99 from 5114 to 6909 are null

####### Visualization ########
plot(mean_dt[1:5113], type = "p",fg="blue",col="purple",xlim=c(0,400))
plot(min_dt[1:5113],type="p",fg="blue",col="purple",xlim=c(0,400))
plot(max_dt[1:5113],type="p",fg="blue",col="purple",xlim=c(0,400))


########1.2 Correlations ########

########1.2.1 correlations between PC variables ########
remove_redundant <- function(correlations,redundant_threshold){
  redundancy<-apply(correlations,2,function(x){which(x>redundant_threshold)});
  redundancy<-redundancy[which(sapply(redundancy,length)>1)]
  
  redundant_variables<-c();
  for (i in redundancy){
    imp<-sort(correlations[1,i],decreasing = TRUE);
    redundant_variables<-c(redundant_variables,names(imp)[2:length(i)])
  }
  redundant_variables<-unique(redundant_variables);
  return(redundant_variables);
} 
##  We will use the remove_redundant for columns 100:456
cors <- abs(cor(dt[,100:456]))

remove_redundant(cors,0.8) # no redundant feature among columns 100:456

########1.2.2 correlatins Among the columns 2-99 ########


## Now we will measure the correlation among columns2:99 
cors_1<-(abs(cor(as.data.frame(dt)[1:5113,2:99])))


length(remove_redundant(cors_1,0.1)) #### as the result shows all the columns 2:99 are behaving in very similar way
# therefore we can just optimize the model for one of them

View(cors_1)
colnames(cors_1)[1]
boxplot(cors_1[,30:42],col="yellow")
cor_comp <- matrix(,nrow=3,ncol=98)
for(i in 1:nrow(cors_1)){
  cor_comp[1,i] <- colnames(cors_1)[i]
  cor_comp[2,i] <- min(cors_1[,i])
  cor_comp[3,i] <- mean(cors_1[,i])
} 
which(cor_comp[2,]==max(cor_comp[2,]))
which(cor_comp[3,]==max(cor_comp[3,])) ## GUTH column 38 is the column that we will keep from the first 98 cols 
# and will run the model to predict it as target as it has the highest association with the rest of group

######### 1.2.3 irrelevant features ########

# IRRELEVANT VARIABLES
remove_irrelevant<-function(correlations,irrelevant_threshold, target){
  index <- which(target == colnames(correlations));
  
  # Irrelevant variables
  relevance<-correlations[index,-index];
  irrelevant_variables<-names(relevance)[is.na(relevance) | relevance<irrelevant_threshold];
  return(irrelevant_variables);
}

dt_without_NA <- dt[1:5113,2:456]   # this data table contains only the historical values and not those that are to be predicted
cors_2 <-abs(cor(dt_without_NA))
class(cors_2)
irrelevant_features <-remove_irrelevant(cors_2,0.02,"GUTH")  ## with this threshold we have 76 relevant features
length(irrelevant_features)
class(irrelevant_features)
dt_relevant_features <- dt[,.SD,,.SDcols=-(as.vector(irrelevant_features))] ## for now we will continue using
# this table for regression analysis and other machine learning models.
length(dt_relevant_features) # we have 76 relevant features
dim(dt_relevant_features)
View(dt_relevant_features)

## the final table that we will use for the following analysis, consists of GUTH col and 75 features 
dt_final <- cbind(dt_relevant_features[,GUTH],dt_relevant_features[,100:175])
colnames(dt_final)[1]="GUTH"
View(dt_final)
dim(dt_final)

########1.3 Outlier detection ########

# Now that we know which are the relevant variabls, we would identify their outliers
# boxplot is a good tool here to identify outliers
boxplot(dt_final[,2:77],outline=FALSE,col = "yellow")

##
library(outliers)
dt_features <- dt_final[1:5113,2:77]  # it is a data table containing only the features and not columns that should be predicted

median_dt_features <- dt_features[,sapply(dt_features,median),]
IQR_dt_features <- dt_features[,sapply(dt_features,IQR),]

## below for loop modifies the outlier based on this criteria : outlier-median>2.5*IQR
nrow(dt_features)
df_features <- as.data.frame(dt_features)
for (i in 1:length((df_features))){
  for (j in 1:nrow(df_features)){
    if (df_features[j,i]>median_dt_features[i]+2.5*IQR_dt_features[i] |df_features[j,i]<median_dt_features[i]-2.5*IQR_dt_features[i]){
      df_features[j,i] <- median_dt_features[i]
    }
  }
}

dt_features <- as.data.table(df_features)

## below function checks to see if the outliers has been modified correctly 
outlier_detector <- function(x){
  features_with_outliers<-c()
  for (i in colnames(df_features)){
    
    if (outlier_1[i]-median_dt_features[i]>2.5*IQR_dt_features[i]){
      features_with_outliers<-c(features_with_outliers,i)
    }
  }
  return(features_with_outliers)
  
}
outlier_1 <- sapply(df_features,outlier)
outlier_detector(df_features) # the result is null therefore we do not have outlier anymore in our dataset

######## 1.4 Data scaling ########
# in order to prevent variables bias we will normalize all the features to make sure they will not affect the result 
#more than reallity.

tipify <- function(x){
  mu <- mean(x, na.rm = TRUE);
  s <- sd(x, na.rm = TRUE);
  
  # Special condition for constant variables
  s[s == 0] <- 1;
  
  # Tipify
  x <- (x - mu) / s;
}

## Scalling should be done only for features and this case relevant features:
dt_rf_normalized <-dt_final[,sapply(.SD,tipify),,.SDcols=-"GUTH"] #relevant normalized features

normalization_plot <- boxplot(dt_rf_normalized[,1:76],outline = FALSE,col="purple")

dt_rf_normalized<- cbind(dt_final[,GUTH],dt_rf_normalized)

colnames(dt_rf_normalized)[1] <- "GUTH"

dt_final_nrm <-as.data.table(dt_rf_normalized)

## We had don the coding up to this stage, and in this part we recieved two new dataset from professor so decided to add the extra variables to calcualtions.
## Normalization of the extra_var :
### importing the new dataset
dummy_read<- read_rds("/Users/Rouzbeh/desktop/IE-BD/R/Group_pro/Additional_variables.RData")

View(dummy_read)
dim(dummy_read)
dummy_read <- setDT(dummy_read)
####measuring the correlations and removing redundant:

cor(dummy_read[,1:100])
cors_dummy <- abs(cor(dummy_read[,1:100]))

extra_redun <- as.vector(remove_redundant(cors_dummy,0.75))

dummy_read[,.SD,,.SDcols=-extra_redun]

extra_var <- dummy_read[,.SD,,.SDcols=-extra_redun]
extra_var<- extra_var[,1:9]  ####


extra_var <- extra_var[,sapply(extra_var,tipify)]

dt_rf_normalized<- cbind(dt_rf_normalized,extra_var)

dt_rf_normalized <- as.data.table(dt_rf_normalized)

######## 1.5 Dimensionality reduction ########
library(caret);

select_important<-function(dat, n_vars, y){
  varimp <- filterVarImp(x = dat, y=y, nonpara=TRUE);
  varimp <- data.table(variable=rownames(varimp),imp=varimp[, 1]);
  varimp <- varimp[order(-imp)];
  selected <- varimp$variable[1:n_vars];
  return(selected);
}

dt_rf_normalized <-as.data.table(dt_rf_normalized)
dt_rf_normalized <- as.data.frame(dt_rf_normalized)
dt_f_varImp <- dt_rf_normalized[1:5113,]
select_important(dt_f_varImp[,2:86],10,dt_f_varImp[,1]) # "PC1"   "PC2"   "PC7"   "PC4"   "PC6"   "PC5"   "V6137" "V6410" "V634"  "V345"
##### interesting insight here is that the new variables that we added to the dataset are important than the majority of 
## the initial ones, so in the top ten important variables we have 4 from the new dataset.

dummy_var_imp<-filterVarImp(dt_f_varImp[,2:86],dt_f_varImp[,1],nonpara = TRUE)


dt_var_Imp<-as.data.table(dummy_var_imp)
dt_var_Imp <- cbind(rownames(dummy_var_imp),dt_var_Imp)

dt_var_Imp<- data.table(rownames(dummy_var_imp),dummy_var_imp)

#### we will set threshold to eliminate variables with importance less than 0.01


final_var_names <-as.data.table((dt_var_Imp[which(dt_var_Imp[,2]>0.015)])[,1]) #### these are the variables that we are going to run model with
final_var_names <-as.vector(dt_var_Imp[which(dt_var_Imp[,2]>0.015)])[,1]
final_var_list <- as.list(final_var_names)


dt_rf_test[,.SD,,.SDcols=c("PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC10","PC11","PC12","PC16","PC17","PC22","PC27","PC28","PC29","PC35","PC38","PC39","PC43","PC44","PC45","PC49","PC50","PC52","PC53","PC54","PC55","PC60","PC64","PC72","PC75","PC87","PC88","PC96","PC102","PC103","PC106","PC108","PC114","PC134","PC137","PC143","PC146","PC150","PC160","PC176","PC191","PC192","PC196","PC205","PC220","PC234","PC283","PC297","PC308","PC316","PC319","PC323","PC332","PC340","PC355","V6409","V6410","V6120","V634","V633","V6425","V345","V2504","V6137")]

##### the most important features to be used in th Ml models :

dt_features_final <- dt_rf_test[,.SD,,.SDcols=c("PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC10","PC11","PC12","PC16","PC17","PC22","PC27","PC28","PC29","PC35","PC38","PC39","PC43","PC44","PC45","PC49","PC50","PC52","PC53","PC54","PC55","PC60","PC64","PC72","PC75","PC87","PC88","PC96","PC102","PC103","PC106","PC108","PC114","PC134","PC137","PC143","PC146","PC150","PC160","PC176","PC191","PC192","PC196","PC205","PC220","PC234","PC283","PC297","PC308","PC316","PC319","PC323","PC332","PC340","PC355","V6409","V6410","V6120","V634","V633","V6425","V345","V2504","V6137")]

###preparing the final dataset to use for the Ml models:
final_dt <- cbind(dt_rf_normalized[,1],dt_features_final)

colnames(final_dt)[1] <- "GUTH"

final_dt <- as.data.table(final_dt)
   ####  the final dataset ####
final_dt




####### 4. Machine learning models ########

######## 4.1 Split into Test/Train ########

final_dt_historic <-final_dt[1:5113] #### the model should be trained and test on the historic data first.
dim(final_dt_historic)
train_index <- sample(1:nrow(final_dt_historic), 0.7*nrow(final_dt_historic)); ## it means from the whole set you are selecting 70% 

View(final_dt)
# training data
train <- final_dt_historic[train_index]; 
dim(train)
# test data
test  <- final_dt_historic[-train_index ]
dim(test)

####### 4.2Linear regression model #########
model_lall <- lm(GUTH ~  ., data = train)
summary(model_lall)

# Get model predictions for train
predictions_train <- predict(model_lall, newdata = train);

# Get model predictions for test
predictions_test <- predict(model_lall, newdata = test);

# Get errors
errors_train <- predictions_train - train$GUTH;
errors_test <- predictions_test - test$GUTH;


# Compute Metrics
mae_train_lm <- round(mean(abs(errors_train)), 2);
mae_test_lm <- round(mean(abs(errors_test)), 2)

mse_test <- round(mean(errors_test^2), 2);
mae_test <- round(mean(abs(errors_test)), 2);

comp <- cbind(mae_train,mae_test)

######## 4.3.SVM model ########

#####Building the finla dataset to be used for ML process#####

Final_ml_dt <- cbind(dt[,2:99],final_dt[,2:73]) # it includes all the 98 station and the features, chosen as importnat for prediction.

## up to this point we did not consider the date in our modelling, after running several models without date,
#we decided to add date and run the model once again. So the proccess of adding date as a new features could be 
# found below. it should be noticed that we used the timediff to calculate the differece from the beginning.
#We also normalized the time differnece in order to prevent over-influencing other parameters' effect.

date<-dt[,1]
class(date)
date <- as.Date(date,format="%Y%m%d")
time_line <- floor(difftime(date,"1994-01-01",unit="days"))
#### Date differnece normalization ####
time_line_normalized <- tipify(time_line)


Final_ml_dt <- cbind(dt[,2:99],final_dt[,2:73]) # this datatable consists of 98 station and 72 features.

Final_ml_dt <- cbind(Final_ml_dt,time_line_normalized) # this data table has also normalized time feature included.
Final_ml_historic <- Final_ml_dt[1:5113,]              # rows with present station values called historic data
Final_ml_test <- Final_ml_dt[5114:6909,]               # rows with NA station values called test data

set.seed(100); 

# row indices for training data (70%)
train_index <- sample(1:nrow(Final_ml_historic), 0.7*nrow(Final_ml_historic));  

# row indices for validation data (15%)
val_index <- sample(setdiff(1:nrow(Final_ml_historic), train_index), 0.15*nrow(Final_ml_historic);  

# row indices for test data (15%)
test_index <- setdiff(1:nrow(Final_ml_historic), c(train_index, val_index));

# split data
train <- Final_ml_historic[train_index,]; 
val <- Final_ml_historic[val_index,]; 
test  <- Final_ml_historic[test_index,];

dim(Final_ml_historic);
dim(train);
dim(val);
dim(test);




##### 4.3.1 hyper parameter optimization####
c_values <- 10^seq(from = -1, to = 2, by = 1);
eps_values <- 10^seq(from = -3, to = 0, by = 1)
gamma_values <- 10^seq(from = -5, to = 1, by = 1);

### Compute grid search
grid_results <- data.table();



for (c in c_values){
  for (eps in eps_values){
    for (gamma in gamma_values){
      
      print(sprintf("Start of c = %s - eps = %s - gamma = %s", c, eps, gamma));
      
      # train SVM model with a particular set of hyperparamets
      G <- as.formula(paste"GUTH","~", paste(colnames(Final_ml_historic[,99:171]), collapse="+")))
      model <- svm(G, data = train, kernel="radial",
                   cost = c, epsilon = eps, gamma = gamma);
      
      # Get model predictions
      predictions_train <- predict(model, newdata = train;
      predictions_val <- predict(model, newdata = val;
      
      # Get errors
      errors_train <- predictions_train - train[,38];
      errors_val <- predictions_val - val[,38];
      
      # Compute Metrics
      mse_train <- round(mean(errors_train^2), 2);
      mae_train <- round(mean(abs(errors_train)), 2);
      
      mse_val <- round(mean(errors_val^2), 2);
      mae_val <- round(mean(abs(errors_val)), 2);
      
      # Build comparison table
      grid_results <- rbind(grid_results,
                            data.table(c = c, eps = eps, gamma = gamma, 
                                       mse_train = mse_train, mae_train = mae_train,
                                       mse_val = mse_val, mae_val = mae_val));
    }
  }
}

# Order results by increasing mse and mae
grid_results <- grid_results[order(mse_val, mae_val)];


# Check results
View(grid_results);


##### 4.3.2 Final model with optimized hyperparameters #####
# having gone through many trial and error process we got the below results as the most optimized parameters:
# c=1, epsilon= 0.1 Gamma= 0.001

# now we will train the model with these parameters for GUTH station and get the predictions for all other stations:
# key point the model should be trained for all the station one by one but with the same hyper parameters
# the prediction result then should be stored in a new datatable as the final output.
G <- as.formula(paste"GUTH","~", paste(colnames(Final_ml_historic[,99:171]), collapse="+")))
model_final <- svm(G, data = train, kernel="radial",
             cost = 1, epsilon = 0.1, gamma = 0.001)

# Get model predictions
predictions_train <- predict(model_final, newdata = train);
predictions_val <- predict(model, newdata = val);
predictions_test <- predict(model, newdata = test);

# Get errors
errors_train <- predictions_train - train[,38];
errors_val <- predictions_val - val[,38];
errors_test <- predictions_test - test[,38];

# Compute Metrics
mae_train <- round(mean(abs(errors_train)), 2);
mae_val <- round(mean(abs(errors_val)), 2);
mae_test <- round(mean(abs(errors_test)), 2);

result_check<-data.table(model = c("optimized_svm"), 
            mae_train = mae_train,
            mae_test = mae_test)

## Summary
sprintf("MAE_train = %s - MAE_val = %s - MAE_test = %s", mae_train, mae_val, mae_test);





## while working with SVM model for predicting future values we noticed that if NA values reamin in the dataset it gives us 
# an error therefore we repalced NA with 0

## replacing missing value in last rows by zero, otherwise we will get error during the prediction 
fill_missing_values <- function(x){      
  if (class(x) == "numeric"){
    x[is.na(x)] <- 0
  } 
  return(x);
}

Final_ml_dt <- as.data.table(Final_ml_dt)

# Getting the final result
dummy_ml <- sapply(as.data.table(Final_ml_test),fill_missing_values) # this the test set with 0 values insyead of NAs just for the prediction part


#### 4.3.4 Predicting the value of all the stations####

result <- dt[5114:6909,1] # this data table would be used in the prediction to storethe results in
class(result)

### this loop train the model with optimized hyperparameters for all the station and get their predictions and stores it in result data table.

for (i in colnames(Final_ml_historic)[1:98]){
  f <- as.formula(paste(i,"~", paste(colnames(Final_ml_historic[,99:171]), collapse="+")))
  final_model<-svm(f , data = Final_ml_historic, kernel="radial",
      cost = 1, epsilon = 0.1, gamma = 0.001)
  predictions<- predict(final_model, newdata = dummy_ml)
  result<- data.table(result,predictions)
}
View(result)
dim(result)
######## 4.4.preparing the submission file#######

class(result)
colnames(result)<-colnames(dt)[1:99]

write.csv(result,"Gprop11",row.names = FALSE,dec=".",sep = ",")

# The final result uploaded to Kaggle and file score was 2402646.20 #



