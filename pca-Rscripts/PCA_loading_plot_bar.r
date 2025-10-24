# Purpose   : draw as a bar plot the loading of each variable in a chosen 
#             component coming from the PCA analysis 
# Input     : T, loading matrix, by default equal to PCA$T coming from PCA
#             matrix evaluation
#             c1, number of componet that must be drawn
#             label, vector of string with the names of the variables
# Output    : bar plot to evaluate the weight of each variables in the 
#             c1 component
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 3.2
# Licence   : GPL2.1
#
plotco<-function(T,c1=1,label=NULL,col=NULL){
nr<-nrow(T)
if(is.null(label))label<-as.character(1:nr)
if(PCA$type=='pca'){
barplot(T[,c1],main=paste('Loading on Component',as.character(c1),sep=' '),
names.arg=as.character(label),cex.names=0.7,las=2,mgp=c(3, .4, 0),col=col)}
else{
barplot(T[,c1],main=paste('Loading on Factor',as.character(c1),sep=' '),
names.arg=as.character(label),cex.names=0.7,las=2,mgp=c(3, .4, 0),col=col)}
box(lty=1,col='red')
return()}
#
# write dialogbox 
#
if(exists("PCA",envir=.GlobalEnv)){
ans<-inpboxckee(c('Component Number','Column Names','External Vector with Variable Names (e.g., A[1,])','Color vector (e.g., A[1,])'),
                1:PCA$res@nPcs,c('0','FALSE','None','None'))
if(!is.null(ans)){
lb<-1:PCA$res@nVar
if(as.logical(ans[[2]]))lb<-names(as.data.frame(PCA[[2]]))
if(ans[[3]]!='None')lb<-eval(parse(text=ans[[3]]),envir=.GlobalEnv)
if(ans[[4]]!='None'){
  variable<-makevar(ans[[4]])
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
  dev.new(title="PCA loading plot bar") 
  plotco(PCA[[1]]@loadings,as.numeric(ans[[1]]),lb,vcolor[grade])
  legend("top", legend=lev,col=vcolor, cex=0.8,pch=18,ncol=min(length(lev),4),inset=c(0,-0.065),xpd=TRUE,bty = "n") 
    }else{
      dev.new(title="PCA loading plot bar")
      op<-par()
      layout(mat=matrix(c(1,2),nrow=1),widths = c(5,0.9))
      par(mar = c(4, 3, 4, 0))
      plotco(PCA[[1]]@loadings,as.numeric(ans[[1]]),lb,vcolor[grade])
      par(mar = c(2, 2, 4, 2))
      m<-as.numeric(lev)[order(as.numeric(lev),decreasing = FALSE)]
      if(variable$control==1)tl=variable$input
      if(variable$control==2)tl=colnames(eval(parse(text=variable$name),envir=.GlobalEnv))[as.numeric(sub("]","",sub(".*\\[,", "", sub("[[:digit:]]+","",sub(":","",sub("[[:digit:]]+","", variable$surname))))))]
      if(variable$control==3)tl=colnames(eval(parse(text=variable$name),envir=.GlobalEnv))[as.numeric(sub("]","",sub(".*\\[,", "", variable$surname)))]
      image(y=m,z=t(m), col=vcolor,
            axes=FALSE, main=tl, cex.main=.8,ylab='')
      axis(4,cex.axis=0.8,mgp=c(0,0.5,0))
      suppressWarnings(par(op))
      rm(m,tl,op)
    }
  rm(variable,grade,lev,nl,vcolor,tog)
  }
}else{
  plotco(PCA[[1]]@loadings,as.numeric(ans[[1]]),lb)
}
rm(lb,ans)
}
}else{
tk_messageBox(type=c("ok"),message='Run Matrix Evaluation First!',caption="Input Error")}
rm(plotco)
