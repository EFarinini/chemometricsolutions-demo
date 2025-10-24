# Purpose   : Plot of the graph T^2 vs Q 
# Input     : PCA object achieved with one of related methods
# Output    : plot graph with 5% confidence
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 3.3
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
        ncp<-as.numeric(ans[[1]])
        n<-PCA[[1]]@nObs
        m<-PCA[[1]]@nVar
        X<-as.matrix(PCA[[1]]@completeObs)
        P<-as.matrix(PCA[[1]]@loadings[,1:ncp])
        L<-as.vector((PCA[[1]]@sDev[1:ncp])^2)
        MQ<-diag(rep(1,m))-(P%*%t(P))
        MT<-P%*%(diag(length(L))*(1/L))%*%t(P)
        Q<-diag(X%*%MQ%*%t(X))
        T<-diag(X%*%MT%*%t(X))
        Qlim<-10^(mean(log10(Q))+qt(0.95,n-1)*sd(log10(Q)))
        Qlim2<-10^(mean(log10(Q))+qt(0.99,n-1)*sd(log10(Q)))
        Qlim3<-10^(mean(log10(Q))+qt(0.999,n-1)*sd(log10(Q)))
        Tlim<-(n-1)*ncp/(n-ncp)*qf(0.95,ncp,n-ncp)
        Tlim2<-(n-1)*ncp/(n-ncp)*qf(0.99,ncp,n-ncp)
        Tlim3<-(n-1)*ncp/(n-ncp)*qf(0.999,ncp,n-ncp)
print("Independent T^2 and Q")
print(Tlim)
print(Tlim2)
print(Tlim3)
print(Qlim)
print(Qlim2)
print(Qlim3)
        if(is.na(Tlim))Tlim<-0
        mT<-max(T,Tlim)
        if(is.na(Qlim))Qlim<-0
        mQ<-max(Q,Qlim)
        dev.new(title="influence plot")
        plot(T,Q,cex=0.5,ylim=c(0,mQ*1.1),xlim=c(0,mT*1.1),ylab="Q Index",xlab="T^2 Hotelling Index",cex.lab=1.2)
        title(main=paste("Number of components:",ncp),sub='Lines show critical values (solid: p=0.05; dashed: p=0.01; dotted: p=0.001) - Outliers coded according to their line number',cex.sub=0.6)
        grid()
        abline(v=Tlim,col='red')
        abline(h=Qlim,col='red')
        abline(v=Tlim2,lty=2,col='red')
        abline(h=Qlim2,lty=2,col='red')
        abline(v=Tlim3,lty=3,col='red')
        abline(h=Qlim3,lty=3,col='red')
        if((Tlim!=0)|(Qlim!=0)){
          txt<-1:n
          if(!ans[[2]])txt<-row.names(PCA$dataset)
          QT<-data.frame(Q=Q,T=T,tx=txt) 
          QTs<-subset(QT,((T>Tlim)|(Q>Qlim)))
          if(nrow(QTs)!=0)text(QTs$T,QTs$Q,label=QTs$tx,cex=0.5,pos=3)
          rm(QT,QTs)}
        Qlim<-10^(mean(log10(Q))+qt(0.974679,n-1)*sd(log10(Q)))
        Qlim2<-10^(mean(log10(Q))+qt(0.994987,n-1)*sd(log10(Q)))
        Qlim3<-10^(mean(log10(Q))+qt(0.9995,n-1)*sd(log10(Q)))
        Tlim<-(n-1)*ncp/(n-ncp)*qf(0.974679,ncp,n-ncp)
        Tlim2<-(n-1)*ncp/(n-ncp)*qf(0.994987,ncp,n-ncp)
        Tlim3<-(n-1)*ncp/(n-ncp)*qf(0.9995,ncp,n-ncp)
print("Joint T^2 and Q")
print(Tlim)
print(Tlim2)
print(Tlim3)
print(Qlim)
print(Qlim2)
print(Qlim3)
        if(is.na(Tlim))Tlim<-0
        mT<-max(T,Tlim)
        if(is.na(Qlim))Qlim<-0
        mQ<-max(Q,Qlim)
        dev.new(title="influence plot joint diagnostics")
        plot(T,Q,cex=0.5,ylim=c(0,mQ*1.1),xlim=c(0,mT*1.1),ylab="Q Index",xlab="T^2 Hotelling Index",cex.lab=1.2)
        title(main=paste("Joint diagnostics - Number of components:",ncp),sub='Boxes define acceptancy regions (solid: p=0.05; dashed: p=0.01; dotted: p=0.001) - Outliers coded according to their line number',cex.sub=0.6)
        grid()
        abline(v=Tlim,col='red')
        abline(h=Qlim,col='red')
        abline(v=Tlim2,lty=2,col='red')
        abline(h=Qlim2,lty=2,col='red')
        abline(v=Tlim3,lty=3,col='red')
        abline(h=Qlim3,lty=3,col='red')
        if((Tlim!=0)|(Qlim!=0)){
          txt<-1:n
          if(!ans[[2]])txt<-row.names(PCA$dataset)
          QT<-data.frame(Q=Q,T=T,tx=txt) 
          QTs<-subset(QT,((T>Tlim)|(Q>Qlim)))
          if(nrow(QTs)!=0)text(QTs$T,QTs$Q,label=QTs$tx,cex=0.5,pos=3)
          rm(QT,QTs)}
        
        t2q<-cbind.data.frame(T,Q)
        colnames(t2q)<-c('T^2','Q')
        print('Values saved and exported in "t2q"',quote=FALSE)
        t2q_tbl<-cbind(c(1:length(T)),t2q)
        colnames(t2q_tbl)[1]<-' '
        write.table(t2q_tbl,'t2q.txt',sep="\t",row.names=FALSE,col.names=TRUE)
        rm(t2q_tbl)
        #rm(ncp,n,m,X,P,L,MQ,MT,Q,T,Tlim,Tlim2,Tlim3,Qlim,Qlim2,Qlim3,mQ,mT,txt)}
rm(ncp,n,m,X,P,L,MQ,MT,Q,T,mQ,mT,txt)}    
}else{
      tk_messageBox(type=c("ok"),message='Function not allowed with Varimax!',caption="Input Error")}
  }
  }else{tk_messageBox(type=c("ok"),message='Run Matrix Evaluation First!',caption="Input Error")}



