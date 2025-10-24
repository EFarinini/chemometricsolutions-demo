# Purpose   : PCA using Nipals algorithm
# Input     : M dataframe with only numeric values
#             nc integer,number of component to evaluate
#             bc boolean TRUE center A before evaluate PCA
#             bs  boolean TRUE scale A before evaluate PCA
#             iteration  maximum number of iterations in Nipals
#             tolerance  tolerance limit for convergence in Nipals
# Output    : PCA object with
#                       [1] Original Data set 
#                       [2] standard deviation for each component (square root
#                           of eigenvalue
#                       [3] total variance of the dataset (sigma tot)
#                       [4] matrix of loadings (eigenvectors)
#                       [5] matrix of scores (new coordinates)
#                       [6] logical value : TRUE if the data set is centered
#                       [7] locical value : TRUE if the data set is scaled
#                       [8] integer: number of variables (observations) 
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP - CM
# Version   : 2.5
# Licence   : GPL2.1
#

suppressWarnings(suppressPackageStartupMessages(library("pcaMethods")))
library(pcaMethods)
PCA<-list()
if(!exists('pca.set'))pca.set<-c(previous.name,'all','all','5','TRUE','TRUE')
ans<-inpboxe4k2(c('*Matrix Name','*Rows (e.g., "1:10,15" or "15:end")',
'*Columns (e.g., "1:3,7" or "3:end")','*Number of Components'),c('Centered','Scaled'),pca.set)
if(!is.null(ans)){
previous.name<-ans[[1]]
pca.set<-ans
M_<-eval(parse(text=ans[[1]]),envir=.GlobalEnv)

.a <- strsplit(unlist(strsplit(ans[[2]], ',')),':')
if((ans[[2]]!='all')& .a[[length(.a)]][length(.a[[length(.a)]])]=='end'){
  .a[[length(.a)]][2]=nrow(M_)
  for(i in 1:length(.a)).a[[i]]<-paste(.a[[i]],collapse=':')
  ans[[2]]<-paste(unlist(.a),collapse=',')
}
.a <- strsplit(unlist(strsplit(ans[[3]], ',')),':')
if((ans[[3]]!='all')& .a[[length(.a)]][length(.a[[length(.a)]])]=='end'){
  .a[[length(.a)]][2]=ncol(M_)
  for(i in 1:length(.a)).a[[i]]<-paste(.a[[i]],collapse=':')
  ans[[3]]<-paste(unlist(.a),collapse=',')
}
rm(.a)

if((ans[[2]]!='all')&(ans[[3]]!='all'))M_<-M_[givedim(ans[[2]]),givedim(ans[[3]])]
if((ans[[2]]!='all')&(ans[[3]]=='all'))M_<-M_[givedim(ans[[2]]),]
if((ans[[2]]=='all')&(ans[[3]]!='all'))M_<-M_[,givedim(ans[[3]])]
if((typeof(M_)=='double')|(typeof(M_)=='list')){
nNA<-sum(is.na(M_))
if(nNA>0){
mess<-paste(as.character(nNA),'missing data present - This is a WARNING message, not an error message')
tk_messageBox(type=c("ok"),message=mess,caption="Input Error")}
ncom<-as.numeric(ans[[4]])
if((ncom>ncol(M_))|(ncom<1)){
tk_messageBox(type=c("ok"),message='Wrong component number !',caption="Input Error")
}else{
sgt<-as.integer(ans[[4]])
if(!ans[[6]])sgt<-sum(apply(M_,2,var))
ccs<-'none';if(ans[[6]])ccs<-'uv'
md<-prep(M_,scale=ccs,center=ans[[5]],simple=FALSE,rev=FALSE)
res<-pca(md$data,method="nipals",nPcs=as.numeric(ans[[4]]),scale=ccs,center=ans[[5]])
PCA$res<-res
PCA$dataset<-prep(PCA$res@completeObs,scale=md$scale,center=md$center,reverse=TRUE)
PCA$dataset<-as.data.frame(PCA$dataset)
PCA$center<-ans[[5]]
PCA$scale<-ans[[6]]
PCA$centered<-md$center
PCA$scaled<-md$scale
PCA$sgt<-sgt
PCA$type<-'pca'
previous.name<-ans[[1]]
print('Note: Data are saved in the PCA object, write PCA to see all',quote=FALSE)
print(paste('Variance explained by ',res@nPcs,' components: ',
round(res@R2cum[res@nPcs]*100,2),'%',sep=''),quote=FALSE)
print('% Variance explained by each component:',quote=FALSE)
print(round(res@R2*100,2),quote=FALSE)
if (exists("mess")) rm(mess)
rm(M_,nNA,ccs,ncom,md,res,sgt)}}
}
