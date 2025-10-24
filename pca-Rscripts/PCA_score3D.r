# Purpose   : draw a 3-D score plot.
# Input     : S, score matrix, by default equal to PCA$scores coming from PCA
#             matrix evaluation
#             c1, c2, c3 integer numbers of the three components that must be drawn
#             factor, vector with a code for each point indicating the cluster
#             label, vector of string with the names of each point, typically
#             an integer number series
# Output    : 3D plot with the points of the data set
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: CM
# Version   : 1.9
# Licence   : GPL2.1suppressWarnings(suppressPackageStartupMessages(library("lattice")))
require(lattice)
if(exists("PCA",envir=.GlobalEnv)){
  if(!exists('score3D.set'))score3D.set<-c('1','2','3','None')
  ans<-inpboxeeee(c('*Component on x-axis','*Component on y-axis','*Component on z-axis','Color Vector (e.g., A[,1])'),
                  score3D.set)
  if(!is.null(ans)){
    score3D.set<-ans
    c1<-as.numeric(ans[[1]])
    c2<-as.numeric(ans[[2]])
    c3<-as.numeric(ans[[3]])
    grade<-NULL
    if(as.character(ans[[4]])!='None'){
      variable<-makevar(ans[[4]])
      grade<-variable$value}
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
    }
    S<-PCA[[1]]@scores
    V<-PCA[[1]]@R2
    r<-nrow(S)
    siz=.9-log10(r)/10 # defines the size of the characters in the plots, based on the number of samples
    c<-nrow(PCA[[1]]@loadings)
    
    DeltaS1lim=0.01*(max(S[,c1])-min(S[,c1]))
    S1lim<-c(min(S[,c1])-DeltaS1lim,max(S[,c1])+DeltaS1lim)
    DeltaS2lim=0.01*(max(S[,c2])-min(S[,c2]))
    S2lim<-c(min(S[,c2])-DeltaS2lim,max(S[,c2])+DeltaS2lim)
    DeltaS3lim=0.01*(max(S[,c3])-min(S[,c3]))
    S3lim<-c(min(S[,c3])-DeltaS3lim,max(S[,c3])+DeltaS3lim)
    dev.new(title="PCA 3D score plot")

m<-min(c(S1lim,S2lim,S3lim))
M<-max(c(S1lim,S2lim,S3lim))

if(PCA$type=='pca'){
    xl<-paste('Comp. ',as.character(c1),' (',as.character(round(V[c1]*100,1)),'%)',sep='')
    yl<-paste('Comp. ',as.character(c2),' (',as.character(round(V[c2]*100,1)),'%)',sep='')
    zl<-paste('Comp. ',as.character(c3),' (',as.character(round(V[c3]*100,1)),'%)',sep='')}
else{
    xl<-paste('Fact. ',as.character(c1),' (',as.character(round(V[c1]*100,1)),'%)',sep='')
    yl<-paste('Fact. ',as.character(c2),' (',as.character(round(V[c2]*100,1)),'%)',sep='')
    zl<-paste('Fact. ',as.character(c3),' (',as.character(round(V[c3]*100,1)),'%)',sep='')}
    tl=paste('Score Plot (',as.character(round((V[c1]+V[c2]+V[c3])*100,1)),'% of total variance)',sep='')
    
    Data<-as.data.frame(S[,c(c1,c2,c3)]);colnames(Data)<-c("x","y","z")
    
    if(is.null(grade)){
      score.3D<-cloud(z~x*y,data = Data,screen = list(z = 30, x = -60),
                       xlim=c(m,M),ylim=c(m,M),zlim=c(m,M),xlab=list(xl,cex=0.8),ylab=list(yl,cex=0.8),zlab=list(zl,cex=0.8),
                       main=tl,cex=siz,col="black",pch=19);print(score.3D)
      }
    if(!is.null(grade)){
      if(tog=="character" | tog=="factor"){
        score.3D<-cloud(z~x*y,data = Data,screen = list(z = 30, x = -60),
                     xlim=c(m,M),ylim=c(m,M),zlim=c(m,M),xlab=list(xl,cex=0.8),ylab=list(yl,cex=0.8),zlab=list(zl,cex=0.8),
                     main=tl,cex=siz,col=vcolor[grade],pch=19,
                     key=list(columns=min(nl,4),cex=0.8,text=list(lev),points=list(pch=19,col=vcolor)));print(score.3D)
      }
      if(tog=="double" | tog=="integer"){

        if(variable$control==1)tl=paste('Score Plot (',as.character(round((V[c1]+V[c2]+V[c3])*100,1)),'% of total variance) \n color scale: ',variable$input,sep='')
        if(!variable$control==1)tl=paste('Score Plot (',as.character(round((V[c1]+V[c2]+V[c3])*100,1)),'% of total variance) \n color scale: ',
                                         colnames(eval(parse(text=variable$name),envir=.GlobalEnv))[as.numeric(sub("]","",sub(".*\\[,", "", variable$surname)))],sep='')

        score.3D<-cloud(z~x*y,data = Data,screen = list(z = 30, x = -60),col=vcolor[grade],
                         drape=TRUE,
                        at=seq(min(as.numeric(lev)),max(as.numeric(lev)),length.out=256),
                        #colorkey=list(labels=list(at=seq(min(as.numeric(lev)),max(as.numeric(lev)),
                        #                                 (max(as.numeric(lev)-min(as.numeric(lev))))/4),
                        #                          labels=round(seq(min(as.numeric(lev)),max(as.numeric(lev)),
                        #                                                  (max(as.numeric(lev)-min(as.numeric(lev))))/4),2))),
                        col.regions =colorpanel(256,low = "blue",high = "red"),
                        xlim=c(m,M),ylim=c(m,M),zlim=c(m,M),xlab=list(xl,cex=0.8),ylab=list(yl,cex=0.8),zlab=list(zl,cex=0.8),
                        main=tl,cex=siz,pch=19);print(score.3D)
      }
      
      rm(lev,nl,vcolor,tog)
      }
    rm(ans,c1,c2,c3,S,V,S1lim,S2lim,S3lim,DeltaS1lim,DeltaS2lim,DeltaS3lim,grade,xl,yl,zl,tl,
       siz,r,Data,c,m,M)   
  }
}else{
  tk_messageBox(type=c("ok"),message='Run Model Computation First!',caption="Input Error")}















