# Purpose   : draw a scaatter plot or a line plot of the loadings 
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 1.2
# Licence   : GPL2.
#
require(stringr)
if(exists("PCA",envir=.GlobalEnv)){
  ans<-inpboxrr('Plot Type  ',c('Scatter','Line'))
  if(!is.null(ans)){
    require(gplots)
    if(as.logical(ans[[1]])){
      ans1<-inpboxeeekk(c('*Component on x-axis','*Component on y-axis','Color vector (e.g., A[1,])'),
                        c('Column Names','Arrows'),c('1','2','None','FALSE','FALSE'))
      # dev.new(title="PCA loading plot scatter")
      # op<-par(pty='s') 
      if(!is.null(ans1)){
        n1<-as.integer(ans1[[1]])
        n2<-as.integer(ans1[[2]])
        T<-PCA[[1]]@loadings
        V<-PCA[[1]]@R2
        siz=.9-log10(nrow(T))/10 # defines the size of the characters in the plots, based on the number of variables
        tex<-as.character(1:nrow(T))
        if(as.logical(ans[[4]]))tex<-row.names(T)
        Tlim<-c(min(T[,c(n1,n2)]),max(T[,c(n1,n2)]))
        Tlim<-c(sign(Tlim[1])*max(abs(Tlim)),sign(Tlim[2])*max(abs(Tlim)))
        if(PCA$type=='pca'){
          x_lab=paste('Component ',as.character(n1),' (',as.character(round(V[n1]*100,1)),'% of variance)',sep='')
          y_lab=paste('Component ',as.character(n2),' (',as.character(round(V[n2]*100,1)),'% of variance)',sep='')
        }else{
          x_lab=paste('Factor ',as.character(n1),' (',as.character(round(V[n1]*100,1)),'% of variance)',sep='')
          y_lab=paste('Factor ',as.character(n2),' (',as.character(round(V[n2]*100,1)),'% of variance)',sep='')
        }
        if(ans[[3]]=='None'){
          dev.new(title="PCA loading plot scatter")
          op<-par(pty='s') 
          plot(T[,n1],T[,n2],xlab=x_lab,ylab=y_lab,
               main=paste('Loading Plot (',as.character(round((V[n1]+V[n2])*100,1)),'% of total variance)',sep=''),type='n',xlim=Tlim,ylim=Tlim)
          text(T[,n1],T[,n2],tex,cex=siz) 
          text(0,0,'+',cex=1.2,col='red')
          #grid()
          if(as.logical(ans[[5]]))arrows(rep(0,dim(T)[1]),rep(0,dim(T)[2]),T[,n1],T[,n2],0.1,col='red')
        }else{
          variable<-makevar(ans[[3]])
          grade<-unlist(variable$value)
          if(!is.null(grade)){
            tog<-typeof(grade)
            if(is.factor(grade))tog<-"factor"
            grade<-factor(grade)
            lev<-levels(grade)
            nl<-nlevels(grade)
            if(tog=="double")vcolor<-unlist(dovc(as.numeric(lev)))
            if(tog=="factor")vcolor<-unlist(dovc(as.character(lev)))
            if(tog=="character")vcolor<-unlist(dovc(as.character(lev)))
            if(tog=="integer")vcolor<-unlist(dovc(as.numeric(lev)))
            if(tog=="character" | tog=="factor"){
              dev.new(title="PCA loading plot scatter")
              op<-par(pty='s') 
              plot(T[,n1],T[,n2],xlab=x_lab,ylab=y_lab,
                   main=paste('Loading Plot (',as.character(round((V[n1]+V[n2])*100,1)),'% of total variance)',sep=''),type='n',xlim=Tlim,ylim=Tlim)
              text(T[,n1],T[,n2],tex,cex=siz,col=vcolor[grade]) 
              text(0,0,'+',cex=1.2,col='red')
              #grid()
              if(as.logical(ans[[5]]))arrows(rep(0,dim(T)[1]),rep(0,dim(T)[2]),T[,n1],T[,n2],0.1,col=vcolor[grade])
              legend("top", legend=lev,col=vcolor, cex=0.8,pch=18,ncol=min(length(lev),4),inset=c(0,-0.065),xpd=TRUE,bty = "n") 
            }else{
              dev.new(title="PCA loading plot scatter")
              op<-par(pty='s')
              layout(mat=matrix(c(1,2),nrow=1),widths = c(5,0.9))
              par(mar = c(4, 3, 4, 0))
              plot(T[,n1],T[,n2],xlab=x_lab,ylab=y_lab,
                   main=paste('Loading Plot (',as.character(round((V[n1]+V[n2])*100,1)),'% of total variance)',sep=''),
                   type='n',xlim=Tlim,ylim=Tlim)
              text(T[,n1],T[,n2],tex,cex=siz,col=vcolor[grade])
              text(0,0,'+',cex=1.2,col='red')
              #grid()
              if(as.logical(ans[[5]]))arrows(rep(0,dim(T)[1]),rep(0,dim(T)[2]),T[,n1],T[,n2],0.1,col=vcolor[grade])
              par(mar = c(2, 2, 4, 2))
              m<-as.numeric(lev)[order(as.numeric(lev),decreasing = FALSE)]
              if(variable$control==1)tl=variable$input
              if(variable$control==2)tl=colnames(eval(parse(text=variable$name),envir=.GlobalEnv))[as.numeric(sub("]","",sub(".*\\[,", "", sub("[[:digit:]]+","",sub(":","",sub("[[:digit:]]+","", variable$surname))))))]
              if(variable$control==3)tl=colnames(eval(parse(text=variable$name),envir=.GlobalEnv))[as.numeric(sub("]","",sub(".*\\[,", "", variable$surname)))]
              image(y=m,z=t(m), col=vcolor,
                    axes=FALSE, main=tl, cex.main=.8,ylab='')
              axis(4,cex.axis=0.8,mgp=c(0,0.5,0))
              rm(tl,m)
            }
            rm(tog,grade,lev,nl,vcolor)
          }
          }
          rm(n1,n2,tex,Tlim,y_lab)
        }
    }else{
        ans1<-inpboxek('*Components to be plotted (e.g.,1,3,5)','Header on the x axis?',vinp=c('1,2','TRUE'),title='Input')
        T<-PCA[[1]]@loadings
        at <- 1:length(rownames(T))
        tk <- TRUE
        # if(length(rownames(T))>15)tk <- FALSE
          if(length(( rownames(T)))>15){
            n<-length(rownames(T))
            m<-floor((n-1)/11)
            at=seq(1,n,m)
            rm(n,m)
          }
        assex <- 1:length(rownames(T))
        x_lab <- 'Variable Number'
        if(ans1[[2]]){
          assex <- rownames(T)
          x_lab <- ''
        }
        dev.new(title="PCA loading plot line") 
        vi<-as.numeric(unlist(str_split(ans1[[1]],',')))
        plot(T[,1],ylab='Loading value',xlab=x_lab,type='n',ylim=c(min(T[,vi]),max(T[,vi])),xaxt='n')
        axis(side=1, at=at,labels=assex[at],cex.axis=0.8,tick=tk)
        #grid()
        for(i in vi)lines(T[,i],col=i)
        legend("bottomleft",legend=as.character(vi),col=vi,lty=1)
        rm(i,vi)
        abline(0,0,lty=2) 
        rm(at,assex,x_lab,tk)
    }
      if (exists("ans1")) rm(ans1)
      if (exists("T")) rm(T)
      if (exists("op")) rm(op)
      if (exists("siz")) rm(siz)
      if (exists("V")) rm(V)
    }
}else{tk_messageBox(type=c("ok"),message='Run Model Computation First!',caption="Input Error")}


