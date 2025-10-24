# Purpose   : plot the experimental values vs. the CV predicted values
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 1.1
# Licence   : GPL2.1
#
if(exists('DOE',envir=.GlobalEnv)){
if(DOE$loY){
 dev.new(title=paste("experimental vs CV predicted",DOE$Yname))
y<-DOE$y
 nr<-nrow(DOE$x)
 siz=.9-log10(nr)/10 # defines the size of the characters in the plots, based on the number of samples
 predcv<-DOE$predcv
 minval<-min(y,predcv)
 maxval<-max(y,predcv)
 dl<-c(minval-(maxval-minval)*0.05,maxval+(maxval-minval)*0.05)
 plot(y,predcv,type='n',col='red',xlim=dl,ylim=dl,xlab='Experimental Value',
 ylab='CV Predicted Value',main=paste("experimental vs CV predicted",DOE$Yname))
 for(i in 1:nr){text(y[i],predcv[i],as.character(i),col='red',cex=siz)}
 abline(a=0,b=1,col='green',lty=1)
 grid()
 rm(y,predcv,minval,maxval,dl,i,nr,siz)
}else{
tk_messageBox(type=c("ok"),message='Missing Y!',
caption="Input Error")}
}else{
tk_messageBox(type=c("ok"),message='Run Model Evaluation First in DOE!',
caption="Input Error")}
