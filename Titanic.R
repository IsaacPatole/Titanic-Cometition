train = read.csv("titanic-train.csv")
attach(train)

install.packages("ggplot2")
library(ggplot2)

# check for empty data
is.na(train)

# get the number of missing records
missing = sapply(train,function(x)sum(is.na(x)))
missing 

# as per the above code 177 people's ages are missing

# replacing the missing ages with mean

train$Age = ifelse(is.na(train$Age),
                     ave(train$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     train$Age)
summary(train)

# box plot for checking outliers and as per the plot Age above 70 are outlier
boxplot(train$Age,main = "outliers",xlab = "Age", col = "blue")


# checking corelation
cor(train[c(2,6)])
pairs(~factor(train$Survived)+train$Age)
pairs(~factor(train$Survived)+train$Sex)
plot(train$Sex,train$Age,xlab = "Sex", ylab = "Age")


# plot the graph to identitfy the percentage of relation between sex and survival. As per the Plot 75% of female survived
# where as only 20 percentage of male survived.
install.packages("ggplot2")
library(ggplot2)

ggplot()+ 
  geom_bar(data=train, aes(x=factor(train$Sex),fill = factor(train$Survived)),
           position = "fill")+
  scale_x_discrete("Sex")+
  scale_y_continuous("Percent")+
  guides(fill = guide_legend(title = "Survived"))+
  scale_fill_manual(values = c("blue","yellow"))

# Bucketing the Age


AgeBucket = cut(Age, breaks = c(0, 20, 40, 60, 80))
AgeBucket

# Plot Age vs Survived
ggplot()+ 
  geom_bar(data=train, aes(x=factor(AgeBucket),fill = factor(train$Survived)),
           position = "fill")+
  scale_x_discrete("Age")+
  scale_y_continuous("Percent")+
  guides(fill = guide_legend(title = "Survived"))+
  scale_fill_manual(values = c("blue","yellow"))

# asper the plot the highest percent of survival is for the age less than 20

# decision tree

library(rpart)
library(rpart.plot)

tree = rpart(train$Survived~AgeBucket+Sex+Fare+SibSp+Parch, data = train,method = "class", cp = 0.001)
rpart.plot(tree)

