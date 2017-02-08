library(glmnet)
library(doParallel)
library(caret)

mae <- function(errors){MAE <- mean(abs(errors));return(MAE)}

args = c('Age_Response age','sorted_age') #for running in Rstudio
if (length(args)< 2){stop('please supply 2 arguments: exp.name and response var')}
if (length(args)==3){seed = as.numeric(args[3])}else{seed=2}

cl <- setup('parallel',3)
title <- paste('LASSO Pair Selectivity',as.character(args[1]),sep=' ')

## Setup ##
features <- list()
coefs <- list()
RSS <- list()
L <- list()

#parameters
type=as.character(args[2])  #type of response variable

## MAIN ##
age <-  as.data.frame(1:20); colnames(age) <- 'age'
set.seed(seed)
n_start <- 1 #start at 2^n features
n_end <- 6 #finish with 2^n features

#storage of data
L <- list() #store lambdas
P <- list() #store MADs
C <- list() #store coefficients
RSS <- list() #store RSS

for (j in n_start:n_end){
  set.seed(seed)
  #create fake data frame
  
  beta2 <- as.data.frame(cbind(age,replicate(2^(j),rnorm(nrow(age),1,1))));colnames(beta2)[1] <-'age'
  q = length(beta2$age)%/%2
  f1 <- c(beta2$age[1:q],rep(0,length(beta2$age)-q)) 
  f2 <- c(rep(0,q),beta2$age[(q+1):length(beta2$age)])
  beta2 <- cbind(beta2,f1,f2)
  
  #storage variables
  L[[j]] <- vector()
  P[[j]] <- vector()
  C[[j]] <- list()
  RSS[[j]] <- vector()
  
  #### DCV LASSO ####
  set.seed(2) #make folds same over 10 iterations
  for (i in 1:10){
    
    print(paste(j,i))
    index <- createFolds(beta2$age,k=10)
    t.train <- beta2[-index[[i]],];row.names(t.train) <- NULL
    t.test <- beta2[index[[i]],];row.names(t.test) <- NULL
    
    L[[j]][i] <- cv.glmnet(x=as.matrix(t.train[,-1]),y=as.matrix(t.train[,1]),parallel = T,alpha=.5)$lambda.min #,lambda=seq(0,10,0.1)
    model <- glmnet(x=as.matrix(t.train[,-1]),y=as.matrix(t.train[,1]),lambda=L[[j]][i],alpha=1)
    C[[j]][[i]] <- coef(model)[,1][coef(model)[,1] != 0]
    pred <- predict(model,as.matrix(t.test[,-1]))
    RSS[[j]][i] <- sum((pred - t.test$age)^2)
    P[[j]][i] <- mae(t.test$age - pred)
    gc()
  }
}

##############
## PLOTTING ##
##############

#calculate plots features
beta_sum = unlist(lapply(unlist(C,recursive = F),function(x){sum(abs(x[-1]))}))
penalty = unlist(L) * beta_sum
RSS = unlist(RSS)
pair_coefs <- unlist(lapply(unlist(C,recursive = F),function(x){
  if('f1' %in% names(x)){f1 = x['f1']}else{f1=0;names(f1)='f1'}
  if('f2' %in% names(x)){f2 = x['f2']}else{f2=0;names(f2)='f2'}
  return(c(f1,f2))}));pair_coefs <- split(pair_coefs,c('f1','f2'))
inout <- lapply(unlist(C,recursive = F),function(x){c('f1','f2') %in% names(x)})
colors <- unlist(lapply(inout,function(x){if (x[1]*x[2]){'green'}else{'red'}}))
featlength <- unlist(lapply(unlist(C,recursive = F),function(x){length(x)-1}))

plot(rep(n_start:n_end,each=10),pair_coefs$f1,col='red',xaxt = "n",xlab='n/o randomly generated features (log2)',main=title,ylim=c(0,1),ylab='pair coefficients');axis(1, at=n_start:n_end);points(rep(n_start:n_end,each=10),pair_coefs$f2,col='blue');axis(1, at=n_start:n_end, labels=(n_start:n_end));legend('bottomleft',fill=c('red','blue'),legend = c('f1','f2'),inset=.02)
plot(rep(n_start:n_end,each=10),RSS+penalty,col=colors,xaxt = "n",xlab='n/o randomly generated features (log2)',main=title);axis(1, at=n_start:n_end, labels=(n_start:n_end));legend('topleft',fill=c('green','red'),legend = c('Pair Selected','Pair not Selected'),inset=.02)
plot(rep(n_start:n_end,each=10),penalty,col=colors,xaxt = "n",xlab='n/o randomly generated features (log2)',main=title);axis(1, at=n_start:n_end, labels=(n_start:n_end));legend('topleft',fill=c('green','red'),legend = c('Pair Selected','Pair not Selected'),inset=.02)
plot(rep(n_start:n_end,each=10),RSS,col=colors,xaxt = "n",xlab='n/o randomly generated features (log2)',main=title);axis(1, at=n_start:n_end, labels=(n_start:n_end));legend('topleft',fill=c('green','red'),legend = c('Pair Selected','Pair not Selected'),inset=.02)
plot(rep(n_start:n_end,each=10),unlist(L),col=colors,xaxt = "n",xlab='n/o randomly generated features (log2)',main=title,ylab=expression(paste(lambda)));axis(1, at=n_start:n_end, labels=(n_start:n_end));legend('topleft',fill=c('green','red'),legend = c('Pair Selected','Pair not Selected'),inset=.02)
plot(rep(n_start:n_end,each=10),featlength,ylab='n/o features per fold',col=colors,xaxt = "n",xlab='n/o randomly generated features (log2)',main=title);axis(1, at=n_start:n_end, labels=(n_start:n_end));legend('topleft',fill=c('green','red'),legend = c('Pair Selected','Pair not Selected'),inset=.02)
plot(penalty,RSS,col=colors,main=title)
plot(penalty,RSS,col=colors,main=title,xlim=c(0,2),ylim=c(0,50))
