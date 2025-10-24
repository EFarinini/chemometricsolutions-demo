# Purpose   : T2 contribution plot on test set
# Input     : PCA object achieved with one of related methods
# Output    : T2 contribution plot
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 1.3
# Licence   : GPL2.1
#
pcaconplot_testset<-function(X,i,PCA,ncp,lbl,nm){
# new dataset evaluation
nr<-nrow(X)
nc<-ncol(X)
name_r <- rownames(X)
unity<-matrix(rep(1,nr),nr,1)
if(PCA$center)X<-X-(unity%*%PCA$centered)
if(PCA$scale)X<-X/(unity%*%PCA$scaled)
X<-as.matrix(X)
P<-PCA[[1]]@loadings[,1:ncp]
S<-X%*%P
Ls<-PCA[[1]]@sDev[1:ncp]
sgl<-sum(Ls)
sgr<-PCA$sgt-sgl
MQ<-S%*%t(P)
MT<-P%*%(diag(1/Ls))%*%t(P)
T<-X%*%MT
Ti<-T[i,]
minT<-min(Ti)
maxT<-max(Ti)
if(minT>0){minT<-0}
if(maxT<0){maxT<-0}
Tlim<-c(minT,maxT)
if(nm){
	X<-as.matrix(PCA[[1]]@completeObs)
	S<-PCA[[1]]@scores[,1:ncp]
	MQ<-S%*%t(P)
	T<-X%*%MT
	Ti<-Ti/apply(abs(T),2,quantile,probs=0.95)
	Tlim<-c(min(Ti,-1.1),max(Ti,1.1))
}
dev.new(title="T^2 contribution plot external")
options(scipen=1)
barplot2(Ti,main=paste('T^2 of object',name_r),ylim=Tlim,cex.lab=1.2,names.arg=lbl,cex.names=0.6,plot.grid=TRUE,las=2,cex.axis=0.6)
box(which="plot",lty="solid")
if(nm)abline(h=1,col='red')
if(nm)abline(h=-1,col='red')
return(Ti)}

library(gplots)
if(exists("PCA",envir=.GlobalEnv)){
  if(PCA$type=='pca'){
    ans<-inpboxer2(c('* External Data Set','Select by','Row number','Row name'),previous.name)
    if(!is.null(ans)){
      M_<-eval(parse(text=ans[[1]]))
      qst <- c('* Row number (e.g.,10)','* Columns to be selected (e.g., "1:3,7" or "7:end")','External Vector with Variable Names (e.g., A[1,])','* Number of Components','Normalized')
      obj <-1:nrow(M_)
      if(ans[[3]]){
        qst <- c('* Row name ','* Columns to be selected (e.g., "1:3,7" or "7:end")','External Vector with Variable Names (e.g., A[1,])','* Number of Components','Normalized')
        obj <- row.names(M_)
      }
      ans<-inpboxceeck(qst,obj,as.character(1:PCA[[1]]@nPcs),c(0,'all','None',1,'TRUE'))
      rm(qst,obj)
      if(!is.null(ans)){
        if((ans[[2]]!='all')& strsplit(unlist(strsplit(ans[[2]], ',')),':')[[1]][2]=='end')
          ans[[2]]<-paste(strsplit(unlist(strsplit(ans[[2]], ',')),':')[[1]][1],':',ncol(M_))
        if(ans[[2]]!='all')M_<-M_[,givedim(ans[[2]])]
        M_ <- M_[ans[[1]],]
        if(sum(is.na(M_))!=0){
          print('>>NA found: remove them before evaluation<<')
        }else{
          lbl<-names(as.data.frame(PCA[[2]]))
          if(ans[[3]]!='None')lbd<-eval(parse(text=ans[[3]]))
          ncp<-as.numeric(ans[[4]])
          nm<-as.numeric(ans[[5]])
          nc<-PCA[[1]]@nVar
          if(ncp<=nc){
            if(nrow(M_)==1){
              contr<-pcaconplot_testset(M_,1,PCA,ncp,lbl,nm)
              assign('contr',contr,envir=.GlobalEnv)
              print('Contributions saved in "contr"',quote=FALSE)
            }else{
              tk_messageBox(type = c("ok"),message='You must choose just one row!',caption="Input Error")
            }
          }else{
            tk_messageBox(type = c("ok"),message='Number of component greater than number of variables!',caption="Input Error")}
          
          rm(ans,M_,nm,nc,ncp,lbl)
        }
      }
      }
  }else{
    tk_messageBox(type=c("ok"),message='Function not allowed with Varimax!',caption="Input Error")}
}else{
  tk_messageBox(type=c("ok"),message='Run Matrix Evaluation First!',caption="Input Error")}
rm(pcaconplot_testset)
  
  
  
  
  
  
  
  
  
  
  

  
  





      
      
      
      
      







    
    
    
  
  
  
  
  
  

