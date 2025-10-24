# Purpose   : Plot of the surface response from a OLS object. Contour plot.
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 1.3
# Licence   : GPL2.1
#
if(exists('DOE',envir=.GlobalEnv)){
require(graphics)
require(RGtk2Extras)
ans<-inpboxee(c('*Minimum value of range','*Maximum value of range'),c('-1','1'))
minrange<-as.numeric(ans[[1]])
maxrange<-as.numeric(ans[[2]])
nv<-DOE$nv
coeff<-DOE$coeff
disper<-DOE$disper
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
a<-0;x<-NULL
vv<-rep(0,nv)
for(ij in seq(minrange,maxrange,by=st)){
    for(y in seq(minrange,maxrange,by=st)){
        a<-a+1
        gr[a,v1]=ij
        gr[a,v2]=y}}
for(i in 1:nv){
  if((i!=v1)&(i!=v2)){
     ans<-gnfib(paste('*Value of variable ',i,' (',DOE$name[i],') ? ',sep=''),'0')
     x<-as.numeric(ans[[1]])
     vv[i]<-x
      if(x!=0){
       gr[,i]<-gr[,i]+x}}}
vv<-as.data.frame(t(vv[-c(v1,v2)])) 
colnames(vv)<-DOE$name[-c(v1,v2)] 
row.names(vv)<-c('')
text.subtl <- NULL
if(nv>2){
  text.subtl <-  paste(colnames(vv)[1],'=',vv[1])
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

lev<-data.frame(Y=diag(gr%*%disper%*%t(gr)),gr)
lev<-lev[,c(1,v1+2,v2+2)]
names(lev)<-c("z","x","y")

print(paste('Leverage: min. ',format(min(lev$z),digits=4),' average ',
format(mean(lev$z),digits=4),' max. ',format(max(lev$z),digits=4),sep=''))
dev.new(title="leverage surface")
zlab<-NULL
if(!is.null(x)){zlab<-paste('Resp.at',format(x,digits=4))}

lev.3D<-lattice::wireframe(z~x*y,data=lev,drape=TRUE,col.regions = colorRampPalette(c("yellow","green","blue"))(256),
                           at=seq(min(lev$z),max(lev$z),length.out=256),
                           #colorkey=list(labels=list(at=seq(min(as.numeric(lev$z)),max(as.numeric(lev$z)),
                           #                                     (max(as.numeric(lev$z)-min(as.numeric(lev$z))))/4),
                           #                          labels=round(seq(min(as.numeric(lev$z)),max(as.numeric(lev$z)),
                           #                                           (max(as.numeric(lev$z)-min(as.numeric(lev$z))))/4),2))),
                           main="Leverage Surface",
                           par.settings=list(
                             par.sub.text = list(font = 1,cex=0.7,
                                                 just = "left", 
                                                 x = grid::unit(5, "mm"))),
                           sub=text.subtl,
                           cex.main=0.8,xlab=DOE$name[v1],
                           ylab=DOE$name[v2],zlab=paste('Leverage')) 

print(lev.3D)

dev.new(title="leverage contour plot")
lev.cont<-lattice::contourplot(z~x*y,data=lev,cuts=15,main="Leverage Contour Plot",
                               par.settings=list(
                                 par.sub.text = list(font = 1,cex=0.7,
                                                     just = "left", 
                                                     x = grid::unit(5, "mm"))),
                               sub=text.subtl,
                               cex.main=0.8,
                                 xlab=DOE$name[v1],ylab=DOE$name[v2],col='green',labels=list(col="green",cex=0.9))
lev.cont<-update(lev.cont,asp=1)
print(lev.cont)

# vv<-as.data.frame(t(vv[-c(v1,v2)]))
# colnames(vv)<-DOE$name[-c(v1,v2)]
# row.names(vv)<-c('')
if(nv>2)print(vv)

rm(i,zlab)
if(exists("text.subtl"))rm(text.subtl)
rm(nv,coeff,minrange,maxrange,r,c,ans,v1,v2,vv,lab,st,nstep,a,j1,j2,ij,x,y,z,gr,
lev,disper,ir)}else{
tk_messageBox(type=c("ok"),message='Run Model Computation First in DOE!',caption="Input Error")}
