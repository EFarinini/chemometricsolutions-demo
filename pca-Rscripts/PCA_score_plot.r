# Purpose   : draw a score plot on the plane of two components.
# Input     : S, score matrix, by default equal to PCA$scores coming from PCA
#             matrix evaluation
#             c1, c2 integer numbers of the two components that must be drawn
#             factor, vector with a code for each point indicating the cluster
#             label, vector of string with the names of each point, typically
#             an integer number series-
# Output    : 2D plot with the points of the data set
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 1.9
# Licence   : GPL2.1

suppressWarnings(suppressPackageStartupMessages(library("lattice")))
require(lattice)
suppressWarnings(suppressPackageStartupMessages(library("latticeExtra")))
require(latticeExtra)

if(exists("PCA",envir=.GlobalEnv)){
if(!exists('score.set'))score.set<-c('1','2','None','None','None','FALSE','FALSE','FALSE')
ans<-inpboxeeeeekkk(c('*Component on x-axis','*Component on y-axis','Label Vector (e.g., A[,1])','Color Vector (e.g., A[,1])','Convex Hull Vector (e.g., A[,1])'),
                    c('Row Names','Ellipses','Line'),score.set,dis=FALSE)
if(!is.null(ans)){
score.set<-ans
c1<-as.numeric(ans[[1]])
c2<-as.numeric(ans[[2]])
tex<-NULL;grade<-NULL
if(as.logical(ans[[6]]))tex<-rownames(PCA$dataset)
if(as.character(ans[[3]])!='None')tex<-eval(parse(text=ans[[3]]),envir=.GlobalEnv)
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
if(!PCA$scale)c <- sum(apply(PCA$dataset,2,'var'))

DeltaS1lim=(max(S[,c1])-min(S[,c1]))
DeltaS2lim=(max(S[,c2])-min(S[,c2]))

if (DeltaS1lim>DeltaS2lim){
  Delta<-DeltaS1lim-DeltaS2lim
  S1lim<-c(min(S[,c1])-DeltaS1lim*0.05,max(S[,c1])+DeltaS1lim*0.05)
  S2lim<-c(min(S[,c2])-Delta/2-DeltaS1lim*0.05,max(S[,c2])+Delta/2+DeltaS1lim*0.05)
}
if (DeltaS2lim>DeltaS1lim){
  Delta<-DeltaS2lim-DeltaS1lim
  S1lim<-c(min(S[,c1])-Delta/2-DeltaS2lim*0.05,max(S[,c1])+Delta/2+DeltaS2lim*0.05)
  S2lim<-c(min(S[,c2])-DeltaS2lim*0.05,max(S[,c2])+DeltaS2lim*0.05)
}

dev.new(title="PCA score plot")

if(PCA$type=='pca'){
xl<-paste('Component ',as.character(c1),' (',as.character(round(V[c1]*100,1)),'% of variance)',sep='')
yl<-paste('Component ',as.character(c2),' (',as.character(round(V[c2]*100,1)),'% of variance)',sep='')}
else{
xl<-paste('Factor ',as.character(c1),' (',as.character(round(V[c1]*100,1)),'% of variance)',sep='')
yl<-paste('Factor ',as.character(c2),' (',as.character(round(V[c2]*100,1)),'% of variance)',sep='')}
tl=paste('Score Plot (',as.character(round((V[c1]+V[c2])*100,1)),'% of total variance)',sep='')

if(!is.null(grade)){
if(tog=="double" | tog=="integer"){
  if(variable$control==1)tl=paste('Score Plot (',as.character(round((V[c1]+V[c2])*100,1)),'% of total variance) \n color scale: ',variable$input,sep='')
  if(!variable$control==1)tl=paste('Score Plot (',as.character(round((V[c1]+V[c2])*100,1)),'% of total variance) \n color scale: ',
                                   colnames(eval(parse(text=variable$name),envir=.GlobalEnv))[as.numeric(sub("]","",sub(".*\\[,", "", variable$surname)))],sep='')}
}

panel.score <- function(x, y, ...) {
  panel.xyplot(x, y, ...)
  # panel.grid(h = -1, v = -1, lty = 3, col = "grey80")
  panel.text(x, y, ...)
  panel.text(0, 0, '+', cex = 1.2, col = 'red')
  
  # Add dotted lines at the center
  panel.abline(h = 0, lty = 3, col = 'grey80')
  panel.abline(v = 0, lty = 3, col = 'grey80')
}

panel.score.ell<-function(x,y,e1,e2,r,c,...){
  panel.xyplot(x,y,...)
  #panel.grid(h=-1, v=-1,lty = 3,col = "grey80")
  panel.text(x,y,...) 
  panel.text(0,0,'+',cex=1.2,col='red')
    # Add dotted lines at the center
  panel.abline(h = 0, lty = 3, col = 'grey80')
  panel.abline(v = 0, lty = 3, col = 'grey80')

    rad1=sqrt((e1*((r-1)/r)*c)*qf(.95,2,r-2)*2*(r^2-1)/(r*(r-2))); 
    rad2=sqrt((e2*((r-1)/r)*c)*qf(.95,2,r-2)*2*(r^2-1)/(r*(r-2)));
    theta <- seq(0, 2 * pi, length=1000)
    x <- rad1 * cos(theta)
    y <- rad2 * sin(theta)
    panel.xyplot(x,y, type = "l",col='red',xlab='',ylab='')
    
    rad1=sqrt((e1*((r-1)/r)*c)*qf(.99,2,r-2)*2*(r^2-1)/(r*(r-2))); 
    rad2=sqrt((e2*((r-1)/r)*c)*qf(.99,2,r-2)*2*(r^2-1)/(r*(r-2)));
    theta <- seq(0, 2 * pi, length=1000)
    x <- rad1 * cos(theta)
    y <- rad2 * sin(theta)
    panel.xyplot(x, y, type = "l",col='red',xlab='',ylab='',lty=2)
    
    rad1=sqrt((e1*((r-1)/r)*c)*qf(.999,2,r-2)*2*(r^2-1)/(r*(r-2))); 
    rad2=sqrt((e2*((r-1)/r)*c)*qf(.999,2,r-2)*2*(r^2-1)/(r*(r-2)));
    theta <- seq(0, 2 * pi, length=1000)
    x <- rad1 * cos(theta)
    y <- rad2 * sin(theta)
    panel.xyplot(x,y, type = "l",col='red',xlab='',ylab='',lty=3)
    
    rm(rad1,rad2,theta,x,y)
}

if(is.null(tex) & is.null(grade)){
  if(!as.logical(ans[[7]])){
    .G_<-xyplot(S[,c2]~S[,c1],xlim=S1lim,ylim=S2lim,
                  pty='o',xlab=xl,ylab=yl,main=tl,col='black',cex=siz,labels=NULL,
                  panel=panel.score)
    if(ans[[8]]).G_<-update(.G_,type='b')
    .G_<-update(.G_,asp=1)
   } else {
    .G_<-xyplot(S[,c2]~S[,c1],xlim=S1lim,ylim=S2lim,
                  pty='o',xlab=xl,ylab=yl,main=tl,col='black',cex=siz,
                  labels=NULL,panel=panel.score.ell,
                  e1=V[c1],e2=V[c2],r=r,c=c,
                  sub="Ellipses: critical T^2 value at p=0.05, 0.01 and 0.001",
                  par.settings = list(par.sub.text = list(cex = 0.6,col = "red")))
    if(ans[[8]]).G_<-update(.G_,type='b')
    .G_<-update(.G_,asp=1)
   }
}

if(!is.null(tex)& is.null(grade)){
  if(!as.logical(ans[[7]])){
   .G_<-xyplot(S[,c2]~S[,c1],xlim=S1lim,ylim=S2lim,xlab=xl,ylab=yl,main=tl,
                 type='n',
                 labels=tex,cex=siz,panel=panel.score)
   if(ans[[8]]).G_<-update(.G_,type='l')
   .G_<-update(.G_,asp=1)
   } else{
   .G_<-xyplot(S[,c2]~S[,c1],xlim=S1lim,ylim=S2lim,xlab=xl,ylab=yl,main=tl,
                  type='n',
                  labels=tex,cex=siz,panel=panel.score.ell,
                  e1=V[c1],e2=V[c2],r=r,c=c,
                  sub="Ellipses: critical T^2 value at p=0.05, 0.01 and 0.001",
                  par.settings = list(par.sub.text = list(cex = 0.6,col = "red")))
   if(ans[[8]]).G_<-update(.G_,type='l')
   .G_<-update(.G_,asp=1)
  }
}

if(is.null(tex)& !is.null(grade)){
  if(tog=="character" | tog=="factor"){
    if(!as.logical(ans[[7]])){
      .G_<-xyplot(S[,c2]~S[,c1],xlim=S1lim,ylim=S2lim,xlab=xl,ylab=yl,main=tl,
                    col=vcolor[grade],pch=19,cex=siz,labels=NULL,
                    key=list(columns=min(nl,4),cex=0.8,text=list(lev),
                    points=list(pch=19,col=vcolor)),panel=panel.score)
      if(ans[[8]]).G_<-update(.G_,type='b')
      .G_<-update(.G_,asp=1)
    } else {
      .G_<-xyplot(S[,c2]~S[,c1],xlim=S1lim,ylim=S2lim,xlab=xl,ylab=yl,main=tl,
                    col=vcolor[grade],pch=19,cex=siz,labels=NULL,
                    key=list(columns=min(nl,4),cex=0.8,text=list(lev),
                    points=list(pch=19,col=vcolor)),panel=panel.score.ell,
                    e1=V[c1],e2=V[c2],r=r,c=c,
                    sub="Ellipses: critical T^2 value at p=0.05, 0.01 and 0.001",
                    par.settings = list(par.sub.text = list(cex = 0.6,col = "red")))
      if(ans[[8]]).G_<-update(.G_,type='b')
      .G_<-update(.G_,asp=1)
    }
  }
  
  if(tog=="double" | tog=="integer"){
    
    panel.levelplot.points<-function (x, y, z, subscripts = TRUE, at = pretty(z), shrink, 
                                      labels, label.style, contour, region, pch = 21, col.symbol = "#00000044", 
                                      ..., col.regions = regions$col, fill = NULL){
      regions <- trellis.par.get("regions")
      zcol <- level.colors(z, at, col.regions, colors = TRUE)
      x <- x[subscripts]
      y <- y[subscripts]
      zcol <- zcol[subscripts]
      panel.xyplot(x, y, fill = zcol, pch = pch, col.symbol = col.symbol, 
                   ...)
      #panel.grid(h=-1, v=-1,lty = 3,col = "grey")
      panel.text(0,0,'+',cex=1.2,col='red')
  # Add dotted lines at the center
  panel.abline(h = 0, lty = 3, col = 'grey80')
  panel.abline(v = 0, lty = 3, col = 'grey80')
    }
    
    panel.levelplot.points.ell<-function (x, y, z,e1,e2,cl,row, subscripts = TRUE, at = pretty(z), shrink, 
                                      labels, label.style, contour, region, pch = 21, col.symbol = "#00000044", 
                                      col.regions = regions$col, fill = NULL,...){
      regions <- trellis.par.get("regions")
      zcol <- level.colors(z, at, col.regions, colors = TRUE)
      x <- x[subscripts]
      y <- y[subscripts]
      zcol <- zcol[subscripts]
      panel.xyplot(x, y, fill = zcol, pch = pch, col.symbol = col.symbol, 
                   ...)
      #panel.grid(h=-1, v=-1,lty = 3,col = "grey")
      panel.text(0,0,'+',cex=1.2,col='red')
  # Add dotted lines at the center
  panel.abline(h = 0, lty = 3, col = 'grey80')
  panel.abline(v = 0, lty = 3, col = 'grey80')
      
      rad1=sqrt((e1*((row-1)/row)*cl)*qf(.95,2,row-2)*2*(row^2-1)/(row*(row-2))); 
      rad2=sqrt((e2*((row-1)/row)*cl)*qf(.95,2,row-2)*2*(row^2-1)/(row*(row-2)));
      theta <- seq(0, 2 * pi, length=1000)
      x <- rad1 * cos(theta)
      y <- rad2 * sin(theta)
      panel.xyplot(x,y, type = "l",col='red',xlab='',ylab='')
      
      rad1=sqrt((e1*((row-1)/row)*cl)*qf(.99,2,row-2)*2*(row^2-1)/(row*(row-2))); 
      rad2=sqrt((e2*((row-1)/row)*cl)*qf(.99,2,row-2)*2*(row^2-1)/(row*(row-2)));
      theta <- seq(0, 2 * pi, length=1000)
      x <- rad1 * cos(theta)
      y <- rad2 * sin(theta)
      panel.xyplot(x, y, type = "l",col='red',xlab='',ylab='',lty=2)
      
      rad1=sqrt((e1*((row-1)/row)*cl)*qf(.999,2,row-2)*2*(row^2-1)/(row*(row-2))); 
      rad2=sqrt((e2*((row-1)/row)*cl)*qf(.999,2,row-2)*2*(row^2-1)/(row*(row-2)));
      theta <- seq(0, 2 * pi, length=1000)
      x <- rad1 * cos(theta)
      y <- rad2 * sin(theta)
      panel.xyplot(x,y, type = "l",col='red',xlab='',ylab='',lty=3)
      
      rm(rad1,rad2,theta,x,y)
    }
    
    if(!as.logical(ans[[7]])){
      
      .G_<-levelplot(variable$value ~S[,c1]*S[,c2],xlim=S1lim,ylim=S2lim,
                       xlab=xl,ylab=yl,main=tl,col=vcolor[grade],pch=19,cex=siz,labels=NULL,
                       panel = panel.levelplot.points, col.regions = colorpanel(256,low = "blue",high = "red"),
                       at=seq(min(as.numeric(lev)),max(as.numeric(lev)),length.out=256),
                     )
      if(ans[[8]]).G_<-update(.G_,type='b')
      .G_<-update(.G_,asp=1)
    } else {
      .G_<-levelplot(variable$value ~S[,c1]*S[,c2],
                     e1=V[c1],e2=V[c2],row=r,cl=c,
                     xlim=S1lim,ylim=S2lim,xlab=xl,ylab=yl,main=tl,col=vcolor[grade],pch=19,cex=siz,
              panel = panel.levelplot.points.ell, col.regions = colorpanel(256,low = "blue",high = "red"),
              at=seq(min(as.numeric(lev)),max(as.numeric(lev)),length.out=256),
              sub="Ellipses: critical T^2 value at p=0.05, 0.01 and 0.001",
              par.settings = list(par.sub.text = list(cex = 0.6,col = "red")))
      if(ans[[8]]).G_<-update(.G_,type='b')
      .G_<-update(.G_,asp=1)
    }
    if (exists("panel.levelplot.points")) rm(panel.levelplot.points)
    if (exists("panel.levelplot.points.ell")) rm(panel.levelplot.points.ell)
  }
rm(lev,nl,vcolor)
}

if(!is.null(tex)& !is.null(grade)){
  if(tog=="character" | tog=="factor"){
    if(!as.logical(ans[[7]])){
      .G_<-xyplot(S[,c2]~S[,c1],xlim=S1lim,ylim=S2lim,xlab=xl,ylab=yl,labels=tex,
                    main=tl,col=vcolor[grade],type='n',cex=siz,panel=panel.score,
                    key=list(columns=min(nl,4),cex=0.8,text=list(lev),points=list(pch=19,col=vcolor)))
      if(ans[[8]]).G_<-update(.G_,type='l')
      .G_<-update(.G_,asp=1)
    } else {
      .G_<-xyplot(S[,c2]~S[,c1],xlim=S1lim,ylim=S2lim,xlab=xl,ylab=yl,labels=tex,
                    main=tl,col=vcolor[grade],type='n',cex=siz,panel=panel.score.ell,
                    key=list(columns=min(nl,4),cex=0.8,text=list(lev),points=list(pch=19,col=vcolor)),
                    e1=V[c1],e2=V[c2],r=r,c=c,
                    sub="Ellipses: critical T^2 value at p=0.05, 0.01 and 0.001",
                    par.settings = list(par.sub.text = list(cex = 0.6,col = "red")))
      if(ans[[8]]).G_<-update(.G_,type='l')
      .G_<-update(.G_,asp=1)
    }
  }
  
  if(tog=="double" | tog=="integer"){
    panel.levelplot.points<-function (x, y, z, subscripts = TRUE, at = pretty(z), shrink, 
                                      labels, label.style, contour, region, pch = 21, col.symbol = "#00000044", 
                                      ..., col.regions = regions$col, fill = NULL){
      regions <- trellis.par.get("regions")
      zcol <- level.colors(z, at, col.regions, colors = TRUE)
      x <- x[subscripts]
      y <- y[subscripts]
      zcol <- zcol[subscripts]
      panel.xyplot(x, y, fill = zcol, pch = pch, col.symbol = col.symbol, 
                   ...)
      panel.text(x,y,col=zcol,labels,...)
      #panel.grid(h=-1, v=-1,lty = 3,col = "grey")
      panel.text(0,0,'+',cex=1.2,col='red')
  # Add dotted lines at the center
  panel.abline(h = 0, lty = 3, col = 'grey80')
  panel.abline(v = 0, lty = 3, col = 'grey80')
    }
    
    panel.levelplot.points.ell<-function (x, y, z, e1,e2,row,cl, subscripts = TRUE, at = pretty(z), shrink, 
                                      labels, label.style, contour, region, pch = 21, col.symbol = "#00000044", 
                                      ..., col.regions = regions$col, fill = NULL){
      regions <- trellis.par.get("regions")
      zcol <- level.colors(z, at, col.regions, colors = TRUE)
      x <- x[subscripts]
      y <- y[subscripts]
      zcol <- zcol[subscripts]
      panel.xyplot(x, y, fill = zcol, pch = pch, col.symbol = col.symbol, 
                   ...)
      panel.text(x,y,col=zcol,labels,...)
      #panel.grid(h=-1, v=-1,lty = 3,col = "grey")
      panel.text(0,0,'+',cex=1.2,col='red')
  # Add dotted lines at the center
  panel.abline(h = 0, lty = 3, col = 'grey80')
  panel.abline(v = 0, lty = 3, col = 'grey80')
      
      rad1=sqrt((e1*((row-1)/row)*cl)*qf(.95,2,row-2)*2*(row^2-1)/(row*(row-2))); 
      rad2=sqrt((e2*((row-1)/row)*cl)*qf(.95,2,row-2)*2*(row^2-1)/(row*(row-2)));
      theta <- seq(0, 2 * pi, length=1000)
      x <- rad1 * cos(theta)
      y <- rad2 * sin(theta)
      panel.xyplot(x,y, type = "l",col='red',xlab='',ylab='')
      
      rad1=sqrt((e1*((row-1)/row)*cl)*qf(.99,2,row-2)*2*(row^2-1)/(row*(row-2))); 
      rad2=sqrt((e2*((row-1)/row)*cl)*qf(.99,2,row-2)*2*(row^2-1)/(row*(row-2)));
      theta <- seq(0, 2 * pi, length=1000)
      x <- rad1 * cos(theta)
      y <- rad2 * sin(theta)
      panel.xyplot(x, y, type = "l",col='red',xlab='',ylab='',lty=2)
      
      rad1=sqrt((e1*((row-1)/row)*cl)*qf(.999,2,row-2)*2*(row^2-1)/(row*(row-2))); 
      rad2=sqrt((e2*((row-1)/row)*cl)*qf(.999,2,row-2)*2*(row^2-1)/(row*(row-2)));
      theta <- seq(0, 2 * pi, length=1000)
      x <- rad1 * cos(theta)
      y <- rad2 * sin(theta)
      panel.xyplot(x,y, type = "l",col='red',xlab='',ylab='',lty=3)
      
      rm(rad1,rad2,theta,x,y)
    }
    
    if(!as.logical(ans[[7]])){
      .G_<-levelplot(variable$value ~S[,c1]*S[,c2],xlim=S1lim,ylim=S2lim,xlab=xl,ylab=yl,main=tl,col=vcolor[grade],type='n',cex=siz,
                       panel = panel.levelplot.points, col.regions = colorpanel(256,low = "blue",high = "red"),labels=tex,
                       at=seq(min(as.numeric(lev)),max(as.numeric(lev)),length.out=256),
                     )
      if(ans[[8]]).G_<-update(.G_,type='l')
      .G_<-update(.G_,asp=1)
    } else {
      .G_<-levelplot(variable$value ~S[,c1]*S[,c2],xlim=S1lim,ylim=S2lim,xlab=xl,ylab=yl,main=tl,col=vcolor[grade],type='n',cex=siz,
                       panel = panel.levelplot.points.ell, col.regions = colorpanel(256,low = "blue",high = "red"),
                       at=seq(min(as.numeric(lev)),max(as.numeric(lev)),length.out=256),
                       e1=V[c1],e2=V[c2],row=r,cl=c,labels=tex,
                       sub="Ellipses: critical T^2 value at p=0.05, 0.01 and 0.001",
                       par.settings = list(par.sub.text = list(cex = 0.6,col = "red")))
      if(ans[[8]]).G_<-update(.G_,type='l')
      .G_<-update(.G_,asp=1)
    }
    if (exists("panel.levelplot.points")) rm(panel.levelplot.points)
    if (exists("panel.levelplot.points.ell")) rm(panel.levelplot.points.ell)
  }
rm(lev,nl,vcolor)
}

if(as.character(ans[[5]])=='None')print(.G_)
if(as.character(ans[[5]])!='None'){
  variable<-makevar(ans[[5]])   
  grade<-variable$value
  if(!is.null(grade)){
  tog<-typeof(grade)
  if(is.factor(grade))tog<-"factor"
  grade<-factor(grade)
  lev<-levels(grade)
  if(tog=="factor"|tog=="character"){
    vcolor<-unlist(dovc(as.character(lev)))
    # vcolor[lev=='-']="#00000000"
    if(ans[4]!='None'& ans[4][[1]]==ans[5][[1]]){######
      # if(ans[4]!='None'){
      variable_c<-makevar(ans[[4]])   
      grade<-variable_c$value
      rm(variable_c)
      tog <- typeof(grade)
      if(tog=='factor'|tog=='character'){
        grade<-factor(grade)
        vcolor<-unlist(dovc(as.character(levels(grade))))
        vcolor[!levels(grade)%in%lev]="#00000000"
        lev<-levels(grade)
      }
    }
    vcolor[lev=='-']="#00000000"
    i=0
    A<-".G_"
    X <- list(NULL)
    CH <- list(NULL)
    print("The matrix of Convex hull areas is saved in cha")
    CHA <- as.data.frame(NULL)
    
    for(l in lev){
      i=i+1
      X[[i]]<-S[eval(parse(text=paste("variable$value<-",variable$input)))==l,c(c1,c2)]
      if(!is.matrix(X[[i]]))
        next
      hpts <- chull(X[[i]])
      hpts<- c(hpts, hpts[1])
      CH[[i]] <- hpts
      CHA[i,1] <- l
      CHA[i,2] <- signif(sp::Polygon(X[[i]][hpts,],hole=F)@area,4)
      # if(l!='-' & sp::Polygon(X[[i]][hpts,],hole=F)@area>0)print(paste0(l,': ',signif(sp::Polygon(X[[i]][hpts,],hole=F)@area,4)),quote=FALSE)
      A<-paste0(A,"+layer(panel.lines(x=X[[",i,"]][CH[[",i,"]],1],y=X[[",i,"]][CH[[",i,"]],2],col=vcolor[",which(l==lev),"]))")
    }
    
    colnames(CHA) <- c('level','area')
    CHA <- CHA[CHA$level!='-',]
    CHA <- CHA[CHA$area>0,]
    assign('cha',CHA,envir = .GlobalEnv)
    rm(CHA)
    
    print(eval(parse(text=A)))
  }else{
    
    print('No Convex Hull ')
    print(.G_)
  }
}}

ans<-inpboxe(title="Score plot", vlabel="Save as",inp="scores")
if (!is.null(ans)){
 assign(ans[[1]],.G_,envir = .GlobalEnv) 
}

if(exists("tog"))rm(tog)
rm(panel.score,panel.score.ell)
rm(c,r,c1,c2,V,S1lim,S2lim)
rm(.G_,siz,ans,S,DeltaS1lim,DeltaS2lim,Delta,tex,grade,xl,yl,tl)
rm_obj<-c('A','CH','hpts','i','l','lev','vcolor','X')
for (o in 1:length(rm_obj))if(exists(rm_obj[o]))rm(list=rm_obj[o])
rm(rm_obj,o)
# }
}}else{
tk_messageBox(type=c("ok"),message='Run Model Computation First!',caption="Input Error")}




