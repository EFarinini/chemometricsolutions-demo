# Purpose   : Plot the Q contributions
# Input     : PCA object achieved with one of related methods
# Output    : one plot for each of the two indexes
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 3.7
# Licence   : GPL2.1
#
pcaconplot<-function(i,PCA,n,m,ncp,lbl,nm){
X<-as.matrix(PCA[[1]]@completeObs)
name_r <- rownames(as.data.frame(X))[i]
P<-PCA[[1]]@loadings[,1:ncp]
S<-PCA[[1]]@scores[,1:ncp]
Ls<-PCA[[1]]@sDev[1:ncp]#not variance because the sum must be the Malanobis distance
sgl<-sum(Ls)
sgr<-PCA$sgt-sgl
MQ<-S%*%t(P)
Q<-sign(X-MQ)*(X-MQ)^2
Qi<-Q[i,]

minQ<-min(Qi)
maxQ<-max(Qi)
if(minQ>0){minQ<-0}
if(maxQ<0){maxQ<-0}
Qlim<-c(minQ,maxQ)

if(nm){
	Qi<-Qi/apply(abs(Q),2,quantile,probs=0.95)
	Qlim<-c(min(Qi,-1.1),max(Qi,1.1))
}
dev.new(title="Q contribution plot")
options(scipen=1)
barplot2(Qi,main=paste('Q of object',name_r),ylim=Qlim,
cex.lab=1.2,names.arg=lbl,cex.names=0.6,plot.grid=TRUE,las=2,cex.axis=0.6)
box(which="plot",lty="solid")
if(nm)abline(h=1,col='red')
if(nm)abline(h=-1,col='red')
return(Qi)
}
# Menu
library(gplots)
if(exists("PCA",envir=.GlobalEnv)){
  if(sum(PCA$res@missing)>0){
    mess<-paste('Not possible to compute Q diagnostics with', sum(PCA$res@missing),'missing data')
    tk_messageBox(type=c("ok"),message=mess,caption="Input Error")
  }else{
    if(PCA$type=='pca'){
      qst <- c('*Row number','*Number of Components','Normalized')
      obj <- as.character(1:PCA[[1]]@nObs)
      ans<-inpboxrr('Select by',c('Row number','Row name'))
      if(!is.null(ans)){
        if(ans[[2]]){
          qst <- c('*Row name','*Number of Components','Normalized')
          obj <- attr( PCA[[1]]@completeObs,"dimnames")[[1]]
        }
        ans<-inpboxc2k(qst,obj,as.character(1:PCA[[1]]@nPcs),c('0','1','TRUE'))
        rm(qst,obj)
        if(!is.null(ans)){
          vc<-as.numeric(ans[[1]])
          ncp<-as.numeric(ans[[2]])
          nm<-as.logical(ans[[3]])
          nc<-PCA[[1]]@nVar
          nr<-PCA[[1]]@nObs
          lbl<-names(as.data.frame(PCA[[2]]))
          if(ncp<=nc){
            contr<-pcaconplot(vc,PCA,nr,nc,ncp,lbl,nm)
            assign('contr',contr,envir=.GlobalEnv)
            print('Contributions saved in "contr"',quote=FALSE)
          }else{
            tk_messageBox(type = c("ok"),message='Number of component greater than number of variables!',caption="Input Error")
          }
          rm(ans,vc,ncp,nm,nc,nr,lbl)
        }
      }
    }else{
      tk_messageBox(type=c("ok"),message='Function not allowed with Varimax!',caption="Input Error")}
  }
  }else{tk_messageBox(type=c("ok"),message='Run Matrix Evaluation First!',
caption="Input Error")}
rm(pcaconplot)
