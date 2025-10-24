# Purpose   : do a biplot  
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 2.2
# Licence   : GPL2.1
#
if(exists("PCA",envir=.GlobalEnv)){
if(!exists('biplot.set'))biplot.set<-c('1','2','None','FALSE','TRUE','FALSE')
ans<-inpboxeeekkk(c('*Component on x-axis','*Component on y-axis','External Vector with Variable Names (e.g., A[1,])'),
c('Column Names','Arrows','Row Names'),biplot.set)
if(!is.null(ans)){
    biplot.set<-ans
    tex<-as.character(1:PCA[[1]]@nObs)
    if(ans[[3]]!='None')tex<-eval(parse(text=ans[[3]]))
    if(ans[[6]])tex <- row.names(PCA$dataset)
    # draw score points
    c1<-as.numeric(ans[[1]])
    c2<-as.numeric(ans[[2]])
    S<-PCA[[1]]@scores
    V<-PCA[[1]]@R2
    Slim<-c(min(S[,c(c1,c2)]),max(S[,c(c1,c2)]))
    Slim<-c(sign(Slim[1])*max(abs(Slim)),sign(Slim[2])*max(abs(Slim)))
dev.new(title="PCA biplot")
if(PCA$type=='pca'){
    xl<-paste('Component ',as.character(c1),' (',as.character(round(V[c1]*100,1)),'% of variance)',sep='')
    yl<-paste('Component ',as.character(c2),' (',as.character(round(V[c2]*100,1)),'% of variance)',sep='')}
else{
    xl<-paste('Factor ',as.character(c1),' (',as.character(round(V[c1]*100,1)),'% of variance)',sep='')
    yl<-paste('Factor ',as.character(c2),' (',as.character(round(V[c2]*100,1)),'% of variance)',sep='')}
    tl=paste('Biplot (',as.character(round((V[c1]+V[c2])*100,1)),'% of total variance)',sep='')
    op<-par(pty='s')
    if(is.null(tex)){
         plot(S[,c(c1,c2)],xlim=Slim,ylim=Slim,pty='o',xlab=xl,ylab=yl,col='black')
    }else{
         plot(S[,c(c1,c2)],xlim=Slim,ylim=Slim,xlab=xl,ylab=yl,type='n')
    text(S[,c(c1,c2)],as.character(tex),col='black',cex=0.7)}
    par(op)
    # draw loading arrows
    par(new=TRUE)
    T<-PCA[[1]]@loadings
    tex<-1:nrow(T)
    if(as.logical(ans[[4]]))tex<-rownames(T)
    Tlim<-c(min(T[,c(c1,c2)]),max(T[,c(c1,c2)]))
    Tlim<-c(sign(Tlim[1])*max(abs(Tlim)),sign(Tlim[2])*max(abs(Tlim)))
    plot(T[,c(c1,c2)],axes=FALSE,type='n',xlim=Tlim,ylim=Tlim,pty='s',xlab=xl,ylab=yl)
    if(as.logical(ans[[5]]))arrows(rep(0,dim(T)[1]),rep(0,dim(T)[2]),T[,c1],T[,c2],col='red')
    text(T[,c1],T[,c2],as.character(tex),cex=0.7,col='red')
    axis(side=4)
    axis(side=3)
    par(new=FALSE)
    # draw centre and grid
    grid()
    text(0,0,'+',cex=1.2,col='red')
    title(main=tl,line=2.5)
	rm(S,T,V,tex,Tlim,Slim,c1,c2,xl,yl,op,tl,ans)
}}else{
tk_messageBox(type=c("ok"),message='Run Matrix Evaluation First!',caption="Input Error")}

