# Purpose    : plot a bar plot of the coefficients
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmers: GMP-CM
# Version    : 3.1
# Licence    : GPL2.1

if(exists("DOE",envir=.GlobalEnv)){
if(DOE$loY){
 dev.new(title=paste("coefficients",DOE$Yname))
 x<-DOE$x
 b<-DOE$b
 b_type <- rownames(DOE$b)
 b_type[grepl("b0", rownames(DOE$b))] <- 'Int'
 num_ <- sapply(b_type, function(x) {
   sum(strsplit(x, "")[[1]] == "*")
 })
 b_type[num_>1]='Q_.'
 b_type[grepl("\\*", b_type)] <- 'I_.'
 b_type[grepl("\\^", b_type)] <- 'Q_.'
 b_type[!b_type%in%c("Int","I_.","Q_.")] <- 'L_.'
 b_type <- as.factor(b_type)
 dof<-DOE$dof
 sig<-DOE$sig
 sdcoeff<-DOE$sdcoeff
 nr<-nrow(x)
 nc<-ncol(x)
 iv<-1
 if(sum(x[,1]==1)==nr)iv<-2
 if(dof==0){
 barplot(b[iv:nc],space=0,col=c("white","red","green","cyan")[match(b_type[iv:nc], c('Int','L_.','I_.','Q_.'))],main=paste("Coefficients",DOE$Yname),names.arg=attr(b,'dimnames')[[1]][iv:nc],las=2,cex.names=0.8)
 box(lty=2)
 rm(x,b,dof,sig,sdcoeff,nr,nc,iv)}else{
 interv<-qt(0.975,dof)*sdcoeff
 llim<-b-interv
 ulim<-b+interv
 min<-min(b[iv:nc],llim[iv:nc])
 max<-max(b[iv:nc],ulim[iv:nc])
 if(sign(min)==sign(max)){
   if(sign(min)==-1)max=0
   if(sign(min)==1)min=0
 }
 barplot(b[iv:nc],space=0,col=c("white","red","green","cyan")[match(b_type[iv:nc], c('Int','L_.','I_.','Q_.'))],main=paste("Coefficients",DOE$Yname),names.arg=attr(b,'dimnames')[[1]][iv:nc],las=2,cex.names=0.8,
 ylim=c(min,max))
 box(lty=2)
 s3<-(sig<=0.001)
 for(i in iv:nc){
  if(s3[i])text((i-iv)+0.5,b[i],'***',cex=2)}
 s2<-(sig>0.001)&(sig<=0.01)
 for(i in iv:nc){
  if(s2[i])text((i-iv)+0.5,b[i],'**',cex=2)}
 s1<-(sig>0.01)&(sig<=0.05)
 for(i in iv:nc){
  if(s1[i])text((i-iv)+0.5,b[i],'*',cex=2)}
 for(i in iv:nc){
  segments((i-iv)+0.5,llim[i],(i-iv)+0.5,ulim[i],col='black')
  segments((i-iv-0.2)+0.5,llim[i],(i-iv+0.2)+0.5,llim[i],col='black')
  segments((i-iv-0.2)+0.5,ulim[i],(i-iv+0.2)+0.5,ulim[i],col='black')}
 rm(iv,interv,llim,ulim,x,b,b_type,dof,nr,nc,num_,i,sig,sdcoeff,s1,s2,s3,min,max)}
}else{
tk_messageBox(type=c("ok"),message='Missing Y!',
caption="Input Error")}
}else{
tk_messageBox(type=c("ok"),message='Run Model Evaluation First in DOE!',
caption="Input Error")}
