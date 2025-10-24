# Purpose   : Plot of the surface response from a OLS object. Contour plot.
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 1.6
# Licence   : GPL2.1
#
require(graphics)
require(RGtk2Extras)
if(exists('DOE',envir=.GlobalEnv)){
if(DOE$loY){
ans<-inpboxee(c('*Minimum value of range','*Maximum value of range'),c('-1','1'))
minrange<-as.numeric(ans[[1]])
maxrange<-as.numeric(ans[[2]])
nv<-DOE$nv
coeff<-DOE$coeff
b<-DOE$b
z<-0
nstep<-30
st<-(maxrange-minrange)/nstep
lab<-seq(minrange,maxrange,by=st)
r<-(nstep+1)^2
c<-nv
gr<-matrix(0,r,c)
ans<-inpboxcc(c('*Index of the variable on X-axis','*Index of the variable on Y-axis'),
as.character(1:nv),as.character(1:nv))
v1<-as.numeric(ans[[1]])
v2<-as.numeric(ans[[2]])
a<-0
x<-NULL
vv<-rep(0,nv)
vn <- rep(0,nv)
for(ij in seq(minrange,maxrange,by=st)){
    for(y in seq(minrange,maxrange,by=st)){
        a<-a+1
        gr[a,v1]=ij
        gr[a,v2]=y}}
for(i in 1:nv){
  if((i!=v1)&(i!=v2)){
     ans<-gnfib(paste('*Value of variable ',i,' (',DOE$name[i],') ? ',sep=''),'0')
     x<-as.numeric(ans[[1]])
     .x_0 <- x
     vv[i]<-x 
     if(x!=0){
       gr[,i]<-gr[,i]+x}}}
.vv_0<-vv
vv<-as.data.frame(t(vv[-c(v1,v2)])) 
colnames(vv)<-DOE$name[-c(v1,v2)] 
row.names(vv)<-c('')
text.subtl <- NULL
if(nv>2){
  text.subtl <-  paste0(colnames(vv)[1],'=',vv[1])
  if(length(vv)>1){
    for(i in 2:length(vv)){
      text.subtl <- paste0(text.subtl,', ', names(vv)[i],'=',vv[i])
    }
  }
}
a<-c
for(j1 in 1:(c-1)){
   for(j2 in (j1+1):c){
       z<-z+1
       if(coeff[z]==1){
       a<-a+1
       gr<-cbind(gr,gr[,j1]*gr[,j2])}}}
for(j1 in 1:c){
   z<-z+1
   if(coeff[z]==1){
       a<-a+1
       gr<-cbind(gr,gr[,j1]^2)}}
z<-z+1
if(coeff[z]==1){
  a<-a+1
  gr<-cbind(rep(1,r),gr)}
ir<-1
risp<-data.frame(Y=gr%*%b,gr)
risp<-risp[,c(1,v1+2,v2+2)]
names(risp)<-c("z","x","y")
dev.new(title=paste("response surface",DOE$Yname))
if(abs((max(risp)-min(risp))/max(risp))>0.01){
resp.3D<-lattice::wireframe(z~x*y,data=risp,drape=TRUE,col.regions = colorRampPalette(c("yellow","green","blue"))(256),
                     at=seq(min(risp$z),max(risp$z),length.out=256),
                     main=paste("Response Surface",DOE$Yname),
                     par.settings=list(
                       par.sub.text = list(font = 1,cex=1,
                                           just = "left", 
                                           x = grid::unit(5, "mm"))),
                     sub=text.subtl,cex.main=1,xlab=DOE$name[v1],
                     ylab=DOE$name[v2],zlab=paste(DOE$Yname)) 
print(resp.3D)
dev.new(title=paste("contour plot",DOE$Yname))
resp.cont<-lattice::contourplot(z~x*y,data=risp,cuts=15,main=paste("Contour Plot",DOE$Yname),
                                
                                
                                par.settings=list(
                                  par.sub.text = list(font = 1,cex=1,
                                                      just = "left", 
                                                      x = grid::unit(5, "mm"))),
                                sub=text.subtl,cex.main=1,
                                xlab=DOE$name[v1],ylab=DOE$name[v2],col='blue',labels=list(col="blue",cex=1))
resp.cont<-update(resp.cont,asp=1)
print(resp.cont)
}else{
print('3D plot impossible: third variable apparently constant')}
source(paste(script.dir,"DOE_CI_surface.r",sep="/"))
rm_obj<-c('i','nv','coeff','minrange','maxrange','r','v1','v2','vn',
          'vv','lab','st',
          'nstep','a','j1','j2','ij','x','y','z','gr','risp','b','ir','.x_0')
for (o in 1:length(rm_obj))if(exists(rm_obj[o]))rm(list=rm_obj[o])
rm(rm_obj,o)
}else{
tk_messageBox(type=c("ok"),message='Missing Y!',caption="Input Error")}
}else{
tk_messageBox(type=c("ok"),message='Run Model Computation First in DOE!',caption="Input Error")}

