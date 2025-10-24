# Purpose   : plot the residuals from OLS object
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 1.1
# Licence   : GPL2.1
#
if(exists('DOE',envir=.GlobalEnv)){
if(DOE$loY){
 dev.new(title=paste("residuals in fitting",DOE$Yname))
 y<-DOE$y
 nr<-nrow(DOE$x)
 pred<-DOE$pred
 plot(1:nr,pred-y,col='red',xlim=c(0.5,(nr+0.5)),type='p',xlab='Sample Number',
 ylab='Residual in Fitting',main=paste("Residuals in Fitting",DOE$Yname))
 abline(h=0,col='green',lty=2)
 grid()
 rm(y,nr,pred)
}else{
tk_messageBox(type=c("ok"),message='Missing Y!',
caption="Input Error")}
}else{
tk_messageBox(type=c("ok"),message='Run Model Evaluation First in DOE!',
caption="Input Error")}
