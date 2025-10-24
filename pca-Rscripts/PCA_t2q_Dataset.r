# Purpose   : Plot of the graph T^2 vs Q and superimpose a new Dataset
# Input     : PCA object achieved with one of related methods
# Output    : plot graph with 5% confidence
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 3.5
# Licence   : GPL2.1
#
pcanewdia<-function(PCA,n,m,ncp,M,lbd){
X<-as.matrix(PCA[[1]]@completeObs)
P<-as.matrix(PCA[[1]]@loadings[,1:ncp])
L<-as.vector((PCA[[1]]@sDev[1:ncp])^2)
MQ<-diag(rep(1,m))-(P%*%t(P))
MT<-P%*%(diag(length(L))*(1/L))%*%t(P)
Q<-diag(X%*%MQ%*%t(X))
T<-diag(X%*%MT%*%t(X))
Qlim<-10^(mean(log10(Q))+qt(0.95,n-1)*sd(log10(Q)))
Tlim<-(n-1)*ncp/(n-ncp)*qf(0.95,ncp,n-ncp)
Qlim2<-10^(mean(log10(Q))+qt(0.99,n-1)*sd(log10(Q)))
Tlim2<-(n-1)*ncp/(n-ncp)*qf(0.99,ncp,n-ncp)
Qlim3<-10^(mean(log10(Q))+qt(0.999,n-1)*sd(log10(Q)))
Tlim3<-(n-1)*ncp/(n-ncp)*qf(0.999,ncp,n-ncp)
if(is.na(Tlim))Tlim<-0
if(is.na(Qlim))Qlim<-0
# new dataset evaluation
nr<-nrow(M)
nc<-ncol(M)
unity<-matrix(rep(1,nr),nr,1)
if(PCA$center)M<-M-(unity%*%PCA$centered)
if(PCA$scale)M<-M/(unity%*%PCA$scaled)
M<-as.matrix(M)
QN<-diag(M%*%MQ%*%t(M))
TN<-diag(M%*%MT%*%t(M))
# plot T^2 vs. Q 
# mQ<-max(Q,QN,Qlim)
# mT<-max(T,TN,Tlim)
mQ<-max(QN,Qlim)
mT<-max(TN,Tlim)

dev.new(title="Q")
# op<-par(mfrow=c(1,2))
plot(QN,ylim=c(0,1.1*mQ),ylab="Q",xlab="Sample number",cex.lab=1.2,col='red')
abline(h=0,col='black')
abline(h=Qlim,col='red')
abline(h=Qlim2,lty=2,col='red')
abline(h=Qlim3,lty=3,col='red')
# if(is.null(lbd))points(QN,col='red')
# if(!is.null(lbd))text(QN,as.character(lbd),col='red',cex=0.6,pos=3)
if(!is.null(lbd)){
  xtx<-(1:nr)[QN>Qlim];ytx<-QN[QN>Qlim]
  text(xtx,ytx,as.character(lbd[QN>Qlim]),col='red',cex=0.6,pos=3)}
#title(main=paste("Lines: crit. val. at p=0.05, 0.01, 0.001 Number of components: ",ncp),cex.main=0.6)
title(main=paste("Lines: crit. val. at p=0.05, 0.01, 0.001 -",ncp,"components"),cex.main=0.6)

dev.new(title="T^2")
plot(TN,ylim=c(0,mT*1.1),ylab="T^2",xlab="Sample number",cex.lab=1.2,col='red')
abline(h=0,col='black')
abline(h=Tlim,col='red')
abline(h=Tlim2,lty=2,col='red')
abline(h=Tlim3,lty=3,col='red')
# if(is.null(lbd))points(TN,col='red')
# if(!is.null(lbd))text(TN,as.character(lbd),col='red',cex=0.6)
if(!is.null(lbd)){
  xtx<-(1:nr)[TN>Tlim];ytx<-TN[TN>Tlim]
  text(xtx,ytx,as.character(lbd[TN>Tlim]),col='red',cex=0.6,pos=3)}
title(main=paste("Lines: crit. val. at p=0.05, 0.01, 0.001 -",ncp,"components"),cex.main=0.6)
}

# Menu
if(exists("PCA",envir=.GlobalEnv)){
  if(sum(PCA$res@missing)>0){
    mess<-paste('Not possible to compute Q diagnostics with', sum(PCA$res@missing),'missing data')
    tk_messageBox(type=c("ok"),message=mess,caption="Input Error")
  }else{
    if(PCA$type=='pca'){
      ans<-inpboxeeeeekk(vlabel = c('*Number of Components','*External Data Set','*Rows to be selected (e.g., "1:10,15" or "15:end")',
                                    '*Columns to be selected (e.g., "1:3,7" or "7:end")','Label Vector (e.g., A[,1])'),
                         vcheck = c('Row Names','Row Numbers'),vinp = c('2',previous.name,'all','all','None',FALSE,FALSE))
      if(!is.null(ans)){
        if(as.numeric(ans[[1]])<=PCA[[1]]@nVar){
          M_<-eval(parse(text=ans[[2]]))
          if((ans[[3]]!='all' & grepl(':', ans[[3]], fixed = TRUE))& strsplit(unlist(strsplit(ans[[3]], ',')),':')[[1]][2]=='end')
            ans[[3]]<-paste(strsplit(unlist(strsplit(ans[[3]], ',')),':')[[1]][1],':',nrow(M_))
          if((ans[[4]]!='all' & grepl(':', ans[[4]], fixed = TRUE))& strsplit(unlist(strsplit(ans[[4]], ',')),':')[[1]][2]=='end')
            ans[[4]]<-paste(strsplit(unlist(strsplit(ans[[4]], ',')),':')[[1]][1],':',ncol(M_))
          if((ans[[3]]!='all')&(ans[[4]]!='all'))M_<-M_[givedim(ans[[3]]),givedim(ans[[4]])]
          if((ans[[3]]!='all')&(ans[[4]]=='all'))M_<-M_[givedim(ans[[3]]),]
          if((ans[[3]]=='all')&(ans[[4]]!='all'))M_<-M_[,givedim(ans[[4]])]
          if(sum(is.na(M_))!=0){
            print('>>NA found: remove them before evaluation<<')
          }else{
            lbd<-NULL
            if(as.logical(ans[[6]])) lbd<-rownames(M_)
            if(as.logical(ans[[7]])) lbd<-1:nrow(M_)
            if(ans[[5]]!='None')lbd<-eval(parse(text=ans[[5]]))
            pcanewdia(PCA,PCA[[1]]@nObs,PCA[[1]]@nVar,as.numeric(ans[[1]]),M_,lbd)
            rm(M_,lbd)}
        }else{tk_messageBox(type=c("ok"),message=
                              'Number of components greater than number of variables!',caption="Input Error")}
      }
    }else{
      tk_messageBox(type=c("ok"),message='Function not allowed with Varimax!',caption="Input Error")}
  }
  }else{
tk_messageBox(type=c("ok"),message='Run Matrix Evaluation First!',caption="Input Error")}
rm(pcanewdia)
