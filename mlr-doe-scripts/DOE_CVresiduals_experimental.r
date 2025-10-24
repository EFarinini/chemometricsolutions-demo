# Purpose   : plot residuals in CV
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 1.1
# Licence   : GPL2.1
#
if (exists('DOE',envir=.GlobalEnv)){
if(DOE$loY){
 dev.new(title=paste("CV residuals",DOE$Yname))
 y<-DOE$y
 nr<-nrow(DOE$x)
 predcv<-DOE$predcv
 rescv<-DOE$rescv
 minval<-min(0,rescv)
 maxval<-max(0,rescv)
 dl<-c(minval-(maxval-minval)*0.05,maxval+(maxval-minval)*0.05)
 op<-par(pty='s',mfrow=c(1,2))
 plot(y,rescv,col='red',ylim=dl,type='p',xlab='Experimental Value',cex.main=0.8,
 ylab='Residual in CV',main=paste("CV residuals",DOE$Yname))
 abline(h=0,col='green',lty=2)
 grid()
 plot(1:nr,predcv-y,col='red',xlim=c(0.5,(nr+0.5)),type='p',xlab='Sample Number',cex.main=0.8,
 ylab='Residual in CV',main=paste("CV residuals",DOE$Yname))
 abline(h=0,col='green',lty=2)
 grid()
 par(op)
 rm(y,dl,minval,maxval,rescv,predcv,nr,op)
}else{
tk_messageBox(type=c("ok"),message='Missing Y!',
caption="Input Error")}
}else{
tk_messageBox(type=c("ok"),message='Run Model Evaluation First in DOE!',caption="Input Error")}
