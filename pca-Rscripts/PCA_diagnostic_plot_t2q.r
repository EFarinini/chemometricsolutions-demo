# Purpose   : Plot of the 2 graphs of T^2 and Q
# Input     : PCA object achieved with one of related methods
# Output    : one plot for each of the two indexes
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 3.4
# Licence   : GPL2.1
#
if(exists("PCA",envir=.GlobalEnv)){
  if(sum(PCA$res@missing)>0){
    mess<-paste('Not possible to compute Q diagnostics with', sum(PCA$res@missing),'missing data')
    tk_messageBox(type=c("ok"),message=mess,caption="Input Error")
}else{
  if(PCA$type=='pca'){
    ans<-inpboxcrr('Number of Components',as.character(1:PCA[[1]]@nPcs),'',c('Row Numbers','Row Names'))
    if(!is.null(ans)){
      ncp<-ans[[1]]
      n<-PCA[[1]]@nObs
      m<-PCA[[1]]@nVar
      X<-as.matrix(PCA[[1]]@completeObs)
      P<-as.matrix(PCA[[1]]@loadings[,1:ncp])
      L<-as.vector((PCA[[1]]@sDev[1:ncp])^2)
      MQ<-diag(rep(1,m))-(P%*%t(P))
      MT<-P%*% (diag(length(L))*(1/L))%*%t(P)
      Q<-diag(X%*%MQ%*%t(X))
      T<-diag(X%*%MT%*%t(X))
      Qlim<-10^(mean(log10(Q))+qt(0.95,n-1)*sd(log10(Q)))
      Qlim2<-10^(mean(log10(Q))+qt(0.99,n-1)*sd(log10(Q)))
      Qlim3<-10^(mean(log10(Q))+qt(0.999,n-1)*sd(log10(Q)))
      Tlim<-(n-1)*ncp/(n-ncp)*qf(0.95,ncp,n-ncp)
      Tlim2<-(n-1)*ncp/(n-ncp)*qf(0.99,ncp,n-ncp)
      Tlim3<-(n-1)*ncp/(n-ncp)*qf(0.999,ncp,n-ncp)
      if(is.na(Tlim))Tlim<-0
      mT<-max(T,Tlim)
      mQ<-max(Q,Qlim)
      # dev.new(title="T^2 and Q plots")
      dev.new(title="Q")
      # op<-par(mfrow=c(1,2))
      plot(Q,ylim=c(0,1.1*mQ),ylab="Q",xlab="Sample number",cex.lab=1.2)
      abline(h=0,col='black')
      abline(h=Qlim,col='red')
      abline(h=Qlim2,lty=2,col='red')
      abline(h=Qlim3,lty=3,col='red')
      xtx<-(1:n)[Q>Qlim];tx<-as.character(xtx)
      if(!ans[[2]])tx<-row.names(PCA$dataset)[Q>Qlim]
      ytx<-Q[Q>Qlim]
      if(length(xtx)!=0)text(xtx,ytx,label=tx,cex=0.5,pos=3)
      #title(main=paste("Lines: crit. val. at p=0.05, 0.01, 0.001 Number of components: ",ncp),cex.main=0.6)
      title(main=paste("Lines: crit. val. at p=0.05, 0.01, 0.001 -",ncp,"components"),cex.main=0.6)
      dev.new(title="T^2")
      plot(T,ylim=c(0,mT*1.1),ylab="T^2",xlab="Sample number",cex.lab=1.2)
      abline(h=0,col='black')
      abline(h=Tlim,col='red')
      abline(h=Tlim2,lty=2,col='red')
      abline(h=Tlim3,lty=3,col='red')
      xtx<-(1:n)[T>Tlim];tx<-as.character(xtx)
      if(!ans[[2]])tx<-row.names(PCA$dataset)[T>Tlim]
      ytx<-T[T>Tlim]
      if(length(xtx)!=0)text(xtx,ytx,label=tx,cex=0.5,pos=3)
      title(main=paste("Lines: crit. val. at p=0.05, 0.01, 0.001 -",ncp,"components"),cex.main=0.6)
      # par(op)
      rm(ncp,n,m,X,P,L,MQ,MT,Q,T,Tlim,mT,Qlim,xtx)
    }
  }else{
    tk_messageBox(type=c("ok"),message='Function not allowed with Varimax!',caption="Input Error")}
  }
  }else{
tk_messageBox(type=c("ok"),message='Run Matrix Evaluation First!',caption="Input Error")}
