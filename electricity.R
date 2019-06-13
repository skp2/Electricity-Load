library(data.table)
library(xgboost)
set.seed(78567)
ts <- fread("curl https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip | funzip", sep=';', dec = ',')
ts[,V1:=substr(V1,1,13)]
ts <- ts[,lapply(.SD,sum),by=V1]
setnames(ts,'V1','DAY_HR')

#create day/year/dow/month hr/month dow features
long <- melt(ts, id.vars = c('DAY_HR'))
long[,`:=`(day=substr(DAY_HR,1,10),hr=substr(DAY_HR,12,13))]
long[,train:=day>'2014-09-07']
long[,dow:=format(strptime(day,'%Y-%m-%d'),'%w' )]
long[,monhr:=paste0(substr(day,6,7),hr)]
long[,mondow:=paste0(substr(day,6,7),dow)]

#create lagged by 24hr variables
setkey(long,variable,DAY_HR)
long[,paste0("prev.value",1:24):=shift(value,n=1:24,fill=0,type="lag"),by=variable]
long[,csum:=cumsum(prev.value24),by=variable]
long <- long[csum>0,]

long[,csum:=NULL]

#compute a random cauchy projection for encoding large cardinality features
sgn.cauchy <- function(dt, var){
  f <- formula(paste0(" ~ factor(",var, ") -1"))
  v.mat <- model.matrix(f, data=dt)
  dm <- dim(v.mat)
  k <- as.integer(ceiling(sqrt(dm[[2]])))+1
  v.c <- matrix(rcauchy( dm[[2]]*k),nrow = dm[[2]], byrow = T)
  v.proj <- v.mat %*% v.c
  for(i in 1:ncol(v.proj)){
    for(j in 1:nrow(v.proj)){
      if (v.proj[j,i] < 0)
        v.proj[j,i] <- 0
      else
        v.proj[j,i] <- 1
    }
  }
  v.proj
}

#apply the projection for month-hr
v.proj <- sgn.cauchy(long,'variable')
for (f in c('monhr')){
  v.proj <- cbind(v.proj, sgn.cauchy(long, f))
}

  v.proj <- cbind(v.proj, as.matrix(long[,paste0("prev.value",1:24)]))

#train==0 is the train set
train <- v.proj[long$train==0,]
test <- v.proj[long$train==1,]
target <- log( 1 + long$value[long$train==0])
truth <- long$value[long$train==1]

long <- NULL

#train using boosted trees
bst <- xgboost(data = train, label = target, subsample=0.5,colsample_bytree =0.5,
               max_depth = 8, eta = .1, nthread = 4, nrounds = 100, objective = "reg:linear")
preds <- exp(predict(bst, test, type='response'))-1
preds <- ifelse(preds <0,0,preds)
nd <- sum(abs(preds - truth))/sum(abs(truth))
nrmse <- sqrt(sum((preds - truth)^2)*length(preds))/sum(abs(truth))
