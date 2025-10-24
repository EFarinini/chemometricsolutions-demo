# Purpose   : plot experimental vs. fitted from OLS object
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 1.1
# Licence   : GPL2.1
#
if(exists('DOE',envir=.GlobalEnv)){
if(DOE$loY){
dev.new(title=paste("experimental vs fitted",DOE$Yname))

 y<-DOE$y
 nr<-nrow(DOE$x)
 siz=.9-log10(nr)/10 # defines the size of the characters in the plots, based on the number of samples
 pred<-DOE$pred
 minval<-min(y,pred)
 maxval<-max(y,pred)
 dl<-c(minval-(maxval-minval)*0.05,maxval+(maxval-minval)*0.05)
 plot(y,pred,type='n',col='red',xlim=dl,ylim=dl,xlab='Experimental Value',
 ylab='Fitted Value',main=paste("Experimental vs. Fitted",DOE$Yname))
 abline(a=0,b=1,col='green',lty=1)
 grid()
 for(i in 1:nr){
  text(y[i],pred[i],as.character(i),col='red',cex=siz)}
 rm(i,y,nr,pred,minval,maxval,dl,siz)
}else{
tk_messageBox(type=c("ok"),message='Missing Y!',
caption="Input Error")}
}else{
tk_messageBox(type=c("ok"),message='Run Model Evaluation First in DOE!',
caption="Input Error")}
