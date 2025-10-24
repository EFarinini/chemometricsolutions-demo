# Purpose    : Multiple Linear Regression function
# Input      : YX dataset
# Authors    : R. Leardi, C. Melzi, G. Polotti
# Programmers: GMP-CM
# Version    : 3.1
# Licence    : GPL2.1

require(RGtk2Extras)
require(matrixcalc)
require(MASS)
if(!exists('doe.set'))doe.set<-c(previous.name,'all','','None','TRUE','FALSE')
ans<-inpboxe4k2(c('*Matrix Name','*Rows to be selected (e.g., 1:10,15)',
'*X-Variables to be selected (e.g., 1:4,8)','Y-Variable to be selected (e.g., 9)'),
c('Quadratic Terms','Cubic Term'),doe.set)
if(!is.null(ans)){
 previous.name<-ans[[1]]
 doe.set<-ans
 DOE<-list()
 M_<-eval(parse(text=ans[[1]]),envir=.GlobalEnv)
 loY<-TRUE
 naY<-NULL
 if(ans[[4]]!='None'){Y_<-M_[,as.integer(ans[[4]])];naY<-colnames(M_)[as.integer(ans[[4]])]
 }else{loY<-FALSE;Y_<-rep(0,nrow(M_))}
 if((ans[[2]]!='all')&(ans[[3]]!='all'))M_<-M_[givedim(ans[[2]]),givedim(ans[[3]])]
 if((ans[[2]]!='all')&(ans[[3]]=='all'))M_<-M_[givedim(ans[[2]]),]
 if((ans[[2]]=='all')&(ans[[3]]!='all'))M_<-M_[,givedim(ans[[3]])]
 if(ans[[2]]!='all')Y_<-Y_[givedim(ans[[2]])]
 
 if((typeof(M_)=='double')|(typeof(M_)=='list')){
   
 M_<-data.frame(cbind(Y_,M_))
 naM<-names(M_)
 naM<-naM[-1]
 nr<-nrow(M_)
 nc<-ncol(M_)-1
 #
 # build system model matrices
 #
 x<-M_[,-1]
 y<-M_[,1]
 m<-matrix(0,nc,nc)
 
 for(l in 5:100){
   if(length(unique(substr(naM,1,l)))==length(substr(naM,1,l))) break()
   }
 
 if(as.logical(ans[[5]])){
   m<-matrix(rep(1,nc*nc),nc,nc)
   diag(m)<-rep(1,nc)
   m<-as.data.frame(m)
   
   names(m)<-substr(naM,1,l)
   row.names(m)<-names(m)
   m<-as.data.frame(upper.triangle(as.matrix(m)))

   for (i in 1:nc){
     for (j in 1 : nc){
		if (i > j) m[i,j]=""}}

   for (i in 1:nc){
	if (length(unique(x[,i]))==2) m[i,i]=""
	if (length(unique(x[,i]))==2 & min(unique(x[,i]))== 0 & max(unique(x[,i])) == 1){
		for (j in 1:nc) {
			m[i,j]=""
			m[j,i]=""}}
   }
   
   diag(m)<-rep("",nc)

   m<-dfedit(m,dataset.name=deparse(substitute(items)),autosize=FALSE,
   size=c(110*nc,40*nc),editable=TRUE,update=TRUE,modal=TRUE)
   } 

   coeff<-rep(0,nc+nc*(nc-1)/2+1)
   z<-0
   a<-nc
   for(j1 in 1:(nc-1)){
    for(j2 in (j1+1):nc){
     z<-z+1
     if(m[j1,j2]==1){
      coeff[z]<-coeff[z]+1
      a<-a+1
      x<-cbind(x,x[,j1]*x[,j2])}}}
    z<-nc*(nc-1)/2
   for(j1 in 1:nc){
      z<-z+1
      if(m[j1,j1]==1){
        coeff[z]<-coeff[z]+1
        a<-a+1
        x<-cbind(x,x[,j1]^2)}}

 x<-as.matrix(x)
 
 var<-factor(substr(naM,1,l),levels=substr(naM,1,l)) 
 M<-matrix(levels(interaction(var,var,sep="*")),length(var))
 
 for(i in 1:length(var)){
   for(j in 1:length(var)){
     if(j<i)M[i,j]<-''
     if(j==i)M[i,j]<-paste(var[i],'^2',sep='')
   }
 }
 
 for(i in 1:length(var)){
   for(j in 1:length(var)){
     if(m[i,j]!=1)M[i,j]<-''
   }
 }
 
 quad<-diag(M)[which(diag(M)!="")]
 diag(M)<-""
 M<-t(M)
 inter<-M[which(M!="")]
 
 coeff_name<-c(substr(naM,1,l),inter,quad)
 colnames(x)<-coeff_name

 if(ans[[6]]){
   x<-cbind(x,x[,1]*x[,2]*x[,3])
   colnames(x)[ncol(x)]<-paste0(var[1],'*',var[2],'*',var[3])
   coeff_name<-c(coeff_name,paste0(var[1],'*',var[2],'*',var[3]))
 }

 ncx<-ncol(x)
 
 if(qr(x)$rank<ncx){
   tk_messageBox(type=c('ok'),
               message=paste("Program aborted because the model matrix is rank deficient!"),caption='Input Error')
               rm(coeff,j1,j2,loY,m,M_,naM,nc,ncx,nr,x,y,Y_,z)
               rm(ans,a,DOE)
               if (exists("i"))rm(i)
               if(exists("j"))rm(j)
 }else{
 
 cat(' ',"\n")
 cat('*************************  Model Solution  *******************************',"\n")
 inf<-t(x)%*%x
 disper<-solve(inf)
 cat(' ',"\n")
 cat('Dispersion Matrix',"\n")
 print(round(disper,4))
 tr<-matrix.trace(disper)
 cat(' ',"\n")
 cat('Trace',"\n")
 cat(round(tr,4),"\n")
 xcc<-x-matrix(rep(1,nr),nr,1)%*%apply(x,2,'mean')
 inf1<-apply(xcc^2,2,sum)*diag(disper)
 cat(' ',"\n")
 cat('Leverage of the Experimental Points',"\n")
 lev<-diag(x%*%disper%*%t(x))
 print(format(as.vector(t(lev)),digit=4),quote=FALSE)
 cat(' ',"\n")
 cat('Maximum leverage',"\n")
 cat(format(max(lev),digit=4),"\n")
 dof<-nr-ncx
 ans<-inpboxrr(c("Experimental variance estimated from:"),
               c("residuals","independent measurements"))
 if(!is.null(ans)){
   if(ans[[2]]){
   ans<-inpboxee(c("experimental standard deviation ","degrees of freedom"),c("",""))
   if(!is.null(ans)){
   rmsef_exp<-as.numeric(ans[[1]])
   dof_exp<-as.numeric(ans[[2]])}}
 }
 
if(loY){
   b<-disper%*%t(x)%*%y
   tb_coeff <- b
   colnames(tb_coeff) <- c('Coefficients')
   cat(' ',"\n")
   if (!exists("dof_exp")){
     cat('Degrees of freedom (regression)',"\n")
     cat((dof),"\n")
   } else {
     cat('Degrees of freedom (independent measurements)',"\n")
     cat((dof_exp),"\n")
   }
   
   if(dof==0 & exists("dof_exp")){
     varcoeff<-rmsef_exp^2*diag(disper)
     sdcoeff<-sqrt(varcoeff)
     tb_coeff <- cbind(tb_coeff,round(sdcoeff,4))
     colnames(tb_coeff)[2] <- c('Std.dev.')
     tb_coeff <- cbind(tb_coeff,round(qt(0.975,dof_exp)*sdcoeff,digit=4))
     colnames(tb_coeff)[3] <- c('Conf.Int.')
     t<-abs(b/sdcoeff)
     sig<-(1-pt(t,dof_exp))*2
     tb_coeff <- cbind(tb_coeff,round(as.vector(sig),4))
     colnames(tb_coeff)[4] <- c('p-value')
     cat('','\n')
     print(tb_coeff,quote=FALSE)
     cat('','\n')
   }
   if(dof==0 & !exists("dof_exp")){
     print(tb_coeff,quote=FALSE)
     cat('','\n')
   }

   if(dof>0){
     pred<-x%*%b
     varres<-sum((y-pred)^2)/dof
     rmsef<-sqrt(varres)
     
    if(exists("rmsef_exp")){
      varcoeff<-rmsef_exp^2*diag(disper)
    } else {
      varcoeff<-varres*diag(disper)
    }
     
     sdcoeff<-sqrt(varcoeff)
     names(sdcoeff)<-coeff_name
     tb_coeff <- cbind(tb_coeff,round(sdcoeff,4))
     colnames(tb_coeff)[2] <- c('Std.dev.')
     
     if(exists("dof_exp")){
       tb_coeff <- cbind(tb_coeff,round(qt(0.975,dof_exp)*sdcoeff,digit=4))
       colnames(tb_coeff)[3] <- c('Conf.Int.')
     } else {
       tb_coeff <- cbind(tb_coeff,round(qt(0.975,dof)*sdcoeff,digit=4))
       colnames(tb_coeff)[3] <- c('Conf.Int.')
     }
     
     t<-abs(b/sdcoeff)
     if(exists("dof_exp")){
       sig<-(1-pt(t,dof_exp))*2
     } else {
       sig<-(1-pt(t,dof))*2
     }
     
     names(sig)<-coeff_name
     tb_coeff <- cbind(tb_coeff,round(t(sig)[,],4))
     colnames(tb_coeff)[4] <- c('p-value')
     cat('',"\n")
     print(tb_coeff,quote=FALSE)
     cat('',"\n")
     
     cat('Variance of Y',"\n")
     vary<-sd(y)^2
     cat(format(vary,digit=4),'\n')
     cat('',"\n")
     cat('Standard deviation of the residuals',"\n")
     cat(format(rmsef,digit=4),'\n')
     cat('',"\n")
     cat('% Explained Variance',"\n")
     cat(round((1-varres/vary)*100,2),'\n')
     cat('',"\n")
     
     cat('Fitted Values',"\n")
     print(format(as.vector(pred),digit=4),quote=FALSE)
     cat('',"\n")
     cat('Residuals',"\n")
     print(format(as.vector(pred-y),digit=4),quote=FALSE)
     cat('',"\n")
     predcv<-rep(0,nr)
     bcr<-matrix(0,nr,ncx)
     for(i in 1:nr){
       xcv<-x[-i,]
       ycv<-y[-i]
       bcv<-ginv(t(xcv)%*%xcv)%*%t(xcv)%*%ycv
       bcr[i,]<-t(bcv)
       predcv[i]<-x[i,]%*%bcv}
     cat('',"\n")
     
     varrescv=sum((y-predcv)^2)/nr
     rmsecv<-sqrt(varrescv)
     
     cat('RMSECV',"\n")
     cat(format(rmsecv,digit=4),'\n')
     cat('',"\n")
     cat('% CV Explained Variance',"\n")
     cat(round((1-varrescv/vary)*100,2),'\n')
     cat('',"\n")
     
     
     cat('CV Values',"\n")
     print(format(predcv,digit=4),quote=FALSE)
     cat('',"\n")
     cat('CV Residuals',"\n")
     rescv<-predcv-y
     print(format(as.vector(rescv),digit=4),quote=FALSE)
     

     
     bmat<-t(b%*%matrix(1,1,nr))
     res<-(bcr-bmat)^2
     cat('',"\n")
     cat('Std.dev. of the coefficients according to resampling',"\n")
     sdres<-sqrt(apply(res,2,sum)*nr/(nr-1))
     print(format(sdres,digit=4),quote=FALSE)
     cat('',"\n")
     cat('Significance of the coefficients according to resampling',"\n")
     t<-abs(b/sdres)
     print(round((t(1-pt(t,nr))[,]*2),4),quote=FALSE)
   }
   if(dof==0){
     cat('0 Degrees of Freedom: no diagnostic plots allowed',"\n")}
   if(dof<0){
     cat('Negative Degree of Freedom: Calculation Ends',"\n")}
}else{
  if(exists("dof_exp")){
    cat('',"\n")
    cat('Degrees of freedom (independent measurements)',"\n")
    cat((dof_exp),'\n')
    varcoeff<-rmsef_exp^2*diag(disper)
    sdcoeff<-sqrt(varcoeff)
    cat('',"\n")
    cat('Std.dev. of coefficients:',"\n")
    cat(format(sdcoeff,digit=4),'\n')
    cat('',"\n")
    cat('Semiamplitude of the confidence intervals',"\n")
    cat(format(qt(0.975,dof_exp)*sdcoeff,digit=4),'\n')
    cat('',"\n")
    }
  }
 
 cat('',"\n")
 cat('**************************************************************************',"\n")
 #
 # save results in DOE object
 #
 DOE$name<-naM
 DOE$Yname<-naY
 DOE$x<-x
 DOE$y<-y
 DOE$nv<-nc
 DOE$m<-m
 DOE$coeff<-coeff
 DOE$inf<-inf
 DOE$disper<-disper
 DOE$tr<-tr
 DOE$lev<-lev
 DOE$loY<-loY
 if(exists("rmsef_exp"))DOE$rmsef_exp<-rmsef_exp
 if(exists("dof_exp"))DOE$dof_exp<-dof_exp

 if(loY){
   DOE$b<-b
   DOE$dof<-dof
   if(dof==0 & exists("dof_exp")){
    DOE$dof<-dof_exp
    DOE$sig<-sig
    DOE$sdcoeff<-sdcoeff}
   if(dof>0){
     if(exists("dof_exp"))DOE$dof<-dof_exp
     DOE$pred<-pred
     DOE$varres<-varres
     DOE$rmsef<-rmsef
     DOE$varcoeff<-varcoeff
     DOE$sig<-sig
     DOE$vary<-vary
     DOE$predcv<-predcv
     DOE$rescv<-rescv
     DOE$varrescv<-varrescv
     DOE$rmsecv<-rmsecv
     DOE$sdres<-sdres
     DOE$sdcoeff<-sdcoeff
     rm(pred,varres,varcoeff,sdcoeff,t,sig,vary,predcv,rescv,varrescv,rmsecv,
     sdres,xcv,ycv,bmat,res,bcr,bcv,b,dof)}

}
rm(nr)
rm(ans,M_,Y_,a,z,coeff,x,y,inf,disper,tr,xcc,inf1,lev,j1,j2,m,naM,naY,nc,ncx,loY)}}
if (exists("rmsef_exp"))rm(rmsef_exp)
if (exists("dof_exp"))rm(dof_exp) 
if (exists("sdcoeff")) rm(sdcoeff)
if (exists("sig")) rm(sig)
if (exists("i")) rm(i)
if (exists("b")) rm(b)
if (exists("dof")) rm(dof)
if (exists("j")) rm(j)
if(exists("rmsef"))rm(rmsef)
if (exists("i")) rm(i)
if (exists("varcoeff")) rm(varcoeff) 
 
 if (exists("coeff_name")) rm(coeff_name)
 if (exists("var")) rm(var)
 if (exists("inter")) rm(inter)
 if (exists("quad")) rm(quad)
 if (exists("M")) rm(M)
 if (exists("l")) rm(l)
 if (!is.function(t)) rm(t)
 if (exists("tb_coeff")) rm(tb_coeff)
 if (exists("M")) rm(M)
 if (exists("l")) rm(l)
 }
