# read CSV 

training = read.csv("C:\\Users\\leahn\\Documents\\R\\CFE\\train_raw.csv") 


# convert AppReceiveDate to date type mm/dd/yyyy

training$AppReceiveDate <- as.Date(training$AppReceiveDate,"%m/%d/%Y")


# extract year of application, make it numeric 
training$AppYear = as.numeric(format(training$AppReceiveDate,'%Y'))
table(training$AppYear)


# extract month of application, treat it as a factor 
training$AppMonth = as.factor(format(training$AppReceiveDate,"%m"))

# extract day of month of application

training$AppDay = as.factor(format(training$AppReceiveDate,"%d"))


# clean vehicle year - there are clear errors
table(training$VehicleYear)

training$VehicleYear[training$VehicleYear == 2104] = 2014
training$VehicleYear[training$VehicleYear == 15]= 2015
training$VehicleYear[training$VehicleYear == 20]= 0

# create vehicle age variable, NA values for people with no car picked out, 0 is for new cars 

training$VehicleAge = ifelse(training$VehicleYear > 0, training$AppYear - training$VehicleYear, 0)
training$VehicleAge[training$VehicleAge == -1]= 0

summary(training$VehicleAge)

# convert factors that are supposed to be numeric
training$CoMonthlyLiability = as.numeric(as.character(training$CoMonthlyLiability))
training$CoMonthlyRent = as.numeric(as.character(training$CoMonthlyRent))
training$LTV = as.numeric(as.character(training$LTV))


# change CoMonthlyLiability NA to zero 

training$CoMonthlyLiability[is.na(training$CoMonthlyLiability)] = 0


# change CoMOnthlyLiability NA to zero 

training$CoMonthlyRent[is.na(training$CoMonthlyRent)] = 0


# group car model 

summary(training$VehicleMake)

training$VehicleMake = as.factor(toupper(training$VehicleMake))

makes = c("UNDECIDED", "NISSAN", "CHEVROLET", "DODGE", "FORD", "HONDA", "HYUNDAI", "KIA", "MAZDA", "TOYOTA")

training$VehicleMake = ifelse(training$VehicleMake %in% makes, as.character(training$VehicleMake) , "OTHER")

training$VehicleMake = as.factor(training$VehicleMake)

table(training$VehicleMake)




# re-group OccupancyStatus

training$OccupancyStatus[training$OccupancyStatus == "GOVQUARTERS"] = "OTHER"
training$OccupancyStatus[is.na(training$OccupancyStatus)] = "OTHER"

table(training$OccupancyStatus)


# LTV has 10,000 missing LTV values. 9,000 of these have not decided on car. 
# change to zero and create LTV indicator to accompany it ? 

training$LTVindicator = ifelse(is.na(training$LTV),0,1) 
training$LTV[is.na(training$LTV)] = 0

# New LTV. Should include UNDECIDED with this one because undecided people have zero LTV
training$LTVNew = ifelse(training$TotalVehicleValue == 0, 0,training$AmountRequested / training$TotalVehicleValue)



# clean RequestType


types = c("A CAR SALE PREAPPROVAL", "LEASE BUYOUT", "PRIVATE PARTY", "REFINANCE", 
          "REFINANCE-PROMO", "TITLE LOAN", "VEHICLE - CROSS SELL")

training$RequestType = ifelse(training$RequestType %in% types, 
                              "OTHER", as.character(training$RequestType))


training$RequestType[is.na(training$RequestType)] = "OTHER"

training$RequestType = as.factor(training$RequestType)


###################################################  Subset the training set to remove outliers

outliers = subset(training, (training$EmployedMonths > 600 | training$PrevEmployedMonths > 600 | training$CoEmployedMonths > 600 |
                training$CoPrevEmployedMonths > 600 | training$TotalMonthlyIncome > 41666.67 | training$PrimeMonthlyLiability > 41666.67 |
                 training$TotalMonthlyDebtBeforeLoan > 10000 | training$OccupancyDuration > 960 | training$PrimeMonthlyRent > 6000 | 
                training$CoMonthlyRent > 6000 | training$VehicleYear < 2000 | (1 <= training$TotalVehicleValue & training$TotalVehicleValue < 1000) | 
                 training$Loanterm < 6 | training$Loanterm > 84 | training$VehicleMileage > 200000) | ((training$DownPayment > training$TotalVehicleValue) & training$TotalVehicleValue != 0) )
                             

############################################################ Random Forest 

library(randomForest)

# drop variables that I don't want as predictors
rforestdata = subset(training, select = -c(LoanNumber,AppReceiveDate))


# figure out missing values 
colSums(is.na(rforestdata))

# get a train & test set
smp_size = floor(.75*nrow(rforestdata))

set.seed(25)
ind = sample(seq_len(nrow(rforestdata)), size = smp_size)
forest_train = rforestdata[ind,]
forest_test = rforestdata[-ind,]

# this is the line that I get a memory error for 
# imputed_train=  rfImpute(LoanStatus ~ ., data = forest_train)

model1 = randomForest(LoanStatus ~ ., data = forest_train, na.action = na.omit)
model1
pred = predict(model1,forest_test)


# calculate test error 
testerror = length(log[log == TRUE])/length(log)
log = pred == forest_test$LoanStatus

# sensitivity 

correct = forest_test$LoanStatus[log] 

sens =  length(correct[correct == "Approved"])/ length(which(forest_test$LoanStatus == "Approved"))
print(sens)

spec = length(correct[correct == "Declined"])/ length(which(forest_test$LoanStatus == "Declined"))
print(spec)


######################################################################## boosting
install.packages("gbm")
library(gbm)



set.seed(100)

# remove Loan number and AppReceiveDate
boostdata = subset(training, select = -c(LoanNumber,AppReceiveDate))

# randomly select 50,000 observations for training set 
ind = sample(seq_len(nrow(boostdata)), size = 50000)
boost_train = boostdata[ind,]

# remaining observations are test set 
boost_test = boostdata[-ind,]


# change response to 0/1 (1 is Approved)
boost_train$response = ifelse(boost_train$LoanStatus == "Approved", 
                              1, 0)
boost_train = subset(boost_train, select = -c(LoanStatus))

# model
mod_gb <- gbm(response ~ ., distribution = "bernoulli", data = boost_train, n.trees = 1000, shrinkage = .1, cv.folds = 10, interaction.depth = 3)


# predict on test set 
pred = predict(mod_gb, boost_test, 1000)

# predictions are in form of log likelihood so anything prediction less than zero is Declined
predlabels = ifelse(pred < 0, "Declined", "Approved")

# percent of test obs that were correctly predicted  
accuracy = sum(predlabels == boost_test$LoanStatus)/length(predlabels)
accuracy


### shrinkage = .1 , ntrees = 400 accuracy = .796

### shrinkage = .1, ntree = 400 accuracy = .84

### shrinkage = .1, ntree = 500 accruacy = .84

### shrinkage = .1 ntree = 1000 accuracy = .854

## shrinkage = .1 ntree = 1000 accuracy = .858

## shrinkage = .1 ntre = 1000 depth = 2  accuracy = .869

## shrinkage = .1 ntree = 1000 depth = 3  accuracy = .872

