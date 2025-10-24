# Purpose   : Comparing the variance explained by a "real" data set with the variance explained by random data sets
# Input     : PCA object obtained on the "real" data set
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP - CM
# Version   : 2.5
# Licence   : GPL2.1
#
suppressWarnings(suppressPackageStartupMessages(library("Rcpp")))
require(pcaMethods)

##################################

set.seed(as.numeric(Sys.time()))

nr <- dim(PCA$dataset)[1];nc <- dim(PCA$dataset)[2]

ans<-gnfib('*Number of randomizations','100')
n <- as.numeric(ans[[1]])

var_rnd <- data.frame(matrix(rep(0,n*as.numeric(PCA$res@nPcs)),nrow = n))

pb<-winProgressBar("computation progress", "% done",
                   min = 0, max = n, initial = 0, width = 300)
for(i in 1:n){
  
  M_ <- data.frame(matrix(rnorm(n = nr*nc),nrow = nr))
 
  sgt<-as.integer(PCA$res@nPcs)
  if(!PCA$scale)sgt<-sum(apply(M_,2,var))
  ccs<-'none';if(PCA$scale)ccs<-'uv'
  md<-prep(M_,scale=ccs,center=PCA$center,simple=FALSE,rev=FALSE)
  res<-pca(md$data,method="nipals",nPcs=as.numeric(PCA$res@nPcs),scale=ccs,center=PCA$center)
  
  var_rnd[i,] <- res@R2*100
  
  setWinProgressBar(pb, i,label = sprintf("%d%% done", round(i/n*100)))
  
}
close(pb)

var_rnd_mean <- apply(var_rnd,2,FUN = 'mean')
var_rnd_sd <- apply(var_rnd,2,FUN = 'sd')

var_rnd_ic_sup <- var_rnd_mean+qt(p = 0.975,df = n-1)*var_rnd_sd
var_rnd_ic_inf <- var_rnd_mean-qt(p = 0.975,df = n-1)*var_rnd_sd

dev.new(title="scree plot")
op<-par(pty='s')
V<-PCA[[1]]@R2*100
xlab<-'Component Number'
if(PCA$type=='varimax')xlab<-'Factor Number'
plot(V,xlab=xlab,ylab="% Explained Variance",
     main='Scree plot',ylim=c(0,max(V)*1.2),type='n')
for(i in 1:length(V)){
  if(V[i]!=0){points(i,V[i],col='red') }}
lines(1:i,V[1:i],col='red')

lines(1:i,var_rnd_mean[1:i],col='green2',lty=2)
lines(1:i,var_rnd_ic_sup[1:i],col='green2',lty=3)
lines(1:i,var_rnd_ic_inf[1:i],col='green2',lty=3)
grid();par(op);rm(V,i,op)
