# Purpose   : Predict the response of new points.
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 1.5
# Licence   : GPL2.1
#
if (exists('DOE',envir=.GlobalEnv)){
if(DOE$loY){

if(!exists('doe.pred.set'))doe.pred.set<-c(previous.name,'all','','None','TRUE','FALSE')  
ans<-inpboxeeeerr(c('*Matrix Name with experiments to be predicted','*Rows to be selected (e.g., 1:10,15)',
                  '*X-Variables to be selected (e.g., 1:4,8)','Y-Variable to be selected (e.g., 9)'),
                  c("experim. var. from residuals","experim. var. from indep. estimation"),
                  doe.pred.set)


if(!is.null(ans)){
 previous.name<-ans[[1]]
 doe.pred.set<-ans
 M_<-eval(parse(text=ans[[1]]),envir=.GlobalEnv)
 
 loY<-TRUE
 if(ans[[4]]!='None'){Y_<-M_[,as.integer(ans[[4]])]
 }else{loY<-FALSE;Y_<-rep(0,nrow(M_))}
 if((ans[[2]]!='all')&(ans[[3]]!='all'))M_<-M_[givedim(ans[[2]]),givedim(ans[[3]])]
 if((ans[[2]]!='all')&(ans[[3]]=='all'))M_<-M_[givedim(ans[[2]]),]
 if((ans[[2]]=='all')&(ans[[3]]!='all'))M_<-M_[,givedim(ans[[3]])]

 if(ans[[2]]!='all')Y_<-Y_[givedim(ans[[2]])]

 if((typeof(M_)=='double')|(typeof(M_)=='list')){
   
if(class(M_)=="numeric"){M_<- as.data.frame((t(as.matrix(c(Y_,M_)))))  # righe=1 e con="all"
}else{
M_<-data.frame(cbind(Y_,M_))}

 naM<-names(M_)
 naM<-naM[-1]
 nr<-nrow(M_)
 nc<-ncol(M_)-1
 nv<-DOE$nv
 b<-DOE$b
 m<-DOE$m
 coeff<-DOE$coeff
 s<-DOE$rmsef
 df<-DOE$dof
 if(ans[[6]]){
   ans<-inpboxee(c("experimental standard deviation ","degrees of freedom "),c("",""))
   s<-as.numeric(ans[[1]])
   df<-as.numeric(ans[[2]])}
 
 if(nv==nc){
 x<-M_[,-1]
 y<-M_[,1]
 z<-0
 a<-nc
 for(j1 in 1:(nc-1)){
    for(j2 in (j1+1):nc){
     z<-z+1
     if(m[j1,j2]==1){
      a<-a+1
      x<-cbind(x,x[,j1]*x[,j2])}}}
 z<-nc*(nc-1)/2
 for(j1 in 1:nc){
      z<-z+1
      if(m[j1,j1]==1){
        a<-a+1
        x<-cbind(x,x[,j1]^2)}}
 if(coeff[length(coeff)]==1){
  a<-a+1
  x<-cbind(rep(1,nr),x)}

 if(length(b)==7)x<-cbind(x,x[,1]*x[,2]*x[,3])

 var.pred<-as.matrix(x)%*%b
 var.pred<-as.data.frame(var.pred)
 colnames(var.pred)='pred' 
 
 lev<-round(diag(as.matrix(x)%*%DOE$disper%*%t(as.matrix(x))),3)
 var.pred<-cbind(var.pred,lev)
 if(df!=0){
    quant<-qt(p = 0.025,lower.tail = FALSE,df = df)
    lower<-var.pred$pred-quant*sqrt(lev)*s
    upper<-var.pred$pred+quant*sqrt(lev)*s
    var.pred<-cbind(var.pred,lower,upper)
 }

 if(loY)var.pred<-cbind(var.pred,y=y,res=var.pred$pred-y)
 
 if(loY){
   colnames(var.pred)[5]=paste0(DOE$Yname," exp")
   dev.new(title=paste("DoE predictions",DOE$Yname))
   op<-par(pty='s',mfrow=c(1,2))
   predcv<-DOE$predcv
   if(DOE$dof==0)predcv<-var.pred$pred
   xl<-c(min(min(y),min(var.pred$pred),min(predcv)),
   max(max(y),max(var.pred$pred),max(predcv)));yl<-xl
   plot(y,var.pred$pred,xlab='Experimental Value',ylab='Predicted Value',
   asp=1,xlim=xl,ylim=yl,main=paste("Predictions",DOE$Yname))
   lines(par('usr')[1:2],par('usr')[3:4],col='red');grid()
   
   suppressWarnings({yl<-c(min((var.pred$pred-y),min(predcv-DOE$y)),
                           max(max(var.pred$pred-y),max(predcv-DOE$y)))})
   
   
   yl<-c(min((var.pred$pred-y),min(predcv-DOE$y)),
   max(max(var.pred$pred-y),max(predcv-DOE$y)))
   plot(1:nr,var.pred$pred-y,xlab='Object Number',ylab='Residuals',ylim=yl,main=paste("Predictions",DOE$Yname))
   abline(h=0,col="red");grid()
   par(op)
   rm(xl,yl,op,predcv)
 }
 
 
 colnames(var.pred)[1]=paste0(DOE$Yname," pred")
 print(var.pred)
 assign(paste0(DOE$Yname,".pred"),var.pred)
 print(paste('The value is saved in:',paste0(DOE$Yname,".pred")))
 
 
 
 rm(s,df,nr,nc,nv,M_,Y_,naM,ans,y,x,j1,j2,a,z,lev,b,coeff,loY,m,var.pred)
 if(exists("lower"))rm(lower)
 if(exists("upper"))rm(upper)
 if(exists("quant"))rm(quant)}}
}else{
tk_messageBox(type=c("ok"),message='Wrong dimension in new vector !',caption="Input Error")}
}else{
tk_messageBox(type=c("ok"),message='Missing Y!',caption="Input Error")}
}else{
tk_messageBox(type=c("ok"),message='Run Model Evaluation First in DOE!',caption="Input Error")}
