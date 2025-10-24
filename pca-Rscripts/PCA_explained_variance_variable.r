# Purpose   : fraction of variance explained by each variable
# Input     : integer, number of principal components
# Output    : numerical none, plot of Explained variance vs variable names
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 3.1
# Licence   : GPL2.1
#
if(exists("PCA",envir=.GlobalEnv)){
require(chemometrics)
ans<-inpboxc('Number of Components:',as.character(1:PCA$res@nPcs),-1)
if(!is.null(ans)){
if(PCA[[1]]@scaled=='uv')scale<-TRUE else scale<-FALSE
dev.new(title="variance of each variable")
pcaVarexpl(PCA$dataset,a=ans[[1]],scale=scale,center=PCA[[1]]@centered,las=2,cex.names=0.7,mgp=c(3, .4, 0),
main='Variance of each Variable explained'); 
rm(ans,scale)}}else{
tk_messageBox(type=c("ok"),message='Run Matrix Evaluation First!',caption="Input Error")}
