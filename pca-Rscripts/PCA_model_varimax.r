# Purpose   : perform varimax rotation
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 1.4
# Licence   : GPL2.1
#
require(pcaMethods)
if(exists('ch.script.dir'))source(paste(ch.script.dir,"PCA_model_PCA.r",sep="/"))
if(exists('script.dir'))source(paste(script.dir,"PCA_model_PCA.r",sep="/"))
if(!is.null(ans)){
if(exists('ch.script.dir'))source(paste(ch.script.dir,"PCA_variance_plot.r",sep="/"))
if(exists('script.dir'))source(paste(script.dir,"PCA_variance_plot.r",sep="/"))
ans<-inpboxc('*Number of components for Varimax rotation',2:PCA$res@nPcs,-1)
ncomp<-ans[[1]]+1
if(!is.null(ans)){
prl<-t(PCA$res@loadings[,1:ncomp])
go<-1
while(go==1){
for(i in 1:(ncomp-1)){
 for(j in (i+1):ncomp){
    lo<-prl[c(i,j),]
    rotb<-0
    sim<-sum(lo^4)
    simmax<-sim
    for (rot in seq(-90,90,0.1)){
     rm<-c(cos(rot*pi/180),-sin(rot*pi/180),sin(rot*pi/180),cos(rot*pi/180))
     rm<-matrix(rm,2,2)
     lo2<-rm%*%lo
     sim2<-sum(lo2^4)
     if(sim2>simmax){
       lob<-lo2
       simmax<-sim2
       rotb<-rot}}
     if(rotb!=0){
       go<-1
       prl[i,]<-lob[1,]
       prl[j,]<-lob[2,]}else{go<-0}}}}
prs<-PCA$res@completeObs%*%t(prl)
vp<-apply(prs^2,2,sum)/sum(apply(PCA$res@completeObs^2,2,sum))
ivp<-sort(vp,decreasing=TRUE,index.return=TRUE)$ix
vp<-sort(vp,decreasing=TRUE,index.return=TRUE)$x
PCA$res@loadings<-PCA$res@loadings[,1:ncomp]
PCA$res@scores<-PCA$res@scores[,1:ncomp]
name.pca<-colnames(PCA$res@loadings)
PCA$res@loadings<-t(prl[ivp,])
PCA$res@scores<-prs[,ivp]
name.pca<-gsub("PC", "Factor ", name.pca)
colnames(PCA$res@loadings)<-name.pca
colnames(PCA$res@scores)<-name.pca
names(vp)<-name.pca
PCA$res@nPcs<-ncomp
PCA$res@R2<-vp
PCA$res@sDev<-sqrt(PCA$res@R2)
PCA$res@R2cum<-cumsum(PCA$res@R2)
PCA$type<-'varimax'
print('*****',quote=FALSE)
print(paste('VARIMAX: Variance explained by ',ncomp,' components: ',round(PCA$res@R2cum[PCA$res@nPcs]*100,2),'%',sep=''),quote=FALSE)
print('% Variance explained by each component:',quote=FALSE)
print(round(PCA$res@R2*100,2),quote=FALSE)
rm(ncomp,prs,vp,ivp,prl,go,rotb,simmax,sim,sim2,lo,lo2,rm,rot,i,j,ans,name.pca)}}

