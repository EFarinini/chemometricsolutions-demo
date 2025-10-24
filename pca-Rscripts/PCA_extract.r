# Purpose   : extract some relevant features from a PCA object
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 1.1
# Licence   : GPL2.1
#
if(exists("PCA",envir=.GlobalEnv)){
nl<-inpboxc('Extract:',c('Loadings','Scores','Expl. Variance','% Expl. Var.','Cum. % Expl. Var.'))
if(!is.null(nl)){
if(nl[[1]]==1){loadings<-t(PCA[[1]]@loadings);print('Values saved and exported in "loadings"',quote=FALSE)}
if(nl[[1]]==2){scores<-PCA[[1]]@scores;print('Values saved and exported in "scores"',quote=FALSE)}
if(nl[[1]]==3){expvar<-PCA[[1]]@sDev^2;print('Values saved and exported in "expvar"',quote=FALSE)}
if(nl[[1]]==4){percexpvar<-PCA[[1]]@R2*100;names(percexpvar)=colnames(PCA[[1]]@scores);print('Values saved and exported in "percexpvar"',quote=FALSE)}
if(nl[[1]]==5){cpercexpvar<-PCA[[1]]@R2cum*100;names(cpercexpvar)=colnames(PCA[[1]]@scores);print('Values saved and exported in "cpercexpvar"',quote=FALSE)}
}}else{tk_messageBox(type=c("ok"),message='Run Matrix Evaluation First in PCA!',
caption="Input Error")}
if(nl[[1]]==1) {values="loadings";col=TRUE;row=TRUE}
if(nl[[1]]==2) {values="scores";col=TRUE;row=TRUE}
if(nl[[1]]==3) {values="expvar";col=FALSE;row=TRUE}
if(nl[[1]]==4) {values="percexpvar";col=FALSE;row=TRUE}
if(nl[[1]]==5) {values="cpercexpvar";col=FALSE;row=TRUE}
ans=list(values,".","NA",col,row)
variable<-NULL
if(!is.null(ans))variable<-makevar(ans[[1]])
if(!is.null(variable)){
	M<-eval(parse(text=ans[[1]]),envir=.GlobalEnv)
	if((as.logical(ans[[4]]))&(as.logical(ans[[5]]))){
		mt<-dimnames(M)[[2]]
		if(is.null(mt))mt<-as.character(1:ncol(M))
		mt<-matrix(c(' ',mt),1,ncol(M)+1)
		write.table(mt,paste(variable$name,'.txt',sep=''),sep="\t",quote=TRUE,row.names=FALSE,col.names=FALSE)
write.table(M,paste(variable$name,'.txt',sep=''),sep="\t",quote=TRUE,append=TRUE,dec=as.character(ans[[2]]),na=as.character(ans[[3]]),row.names=TRUE,col.names=FALSE)
		rm(mt)
	}
	if((as.logical(ans[[4]]))&(!as.logical(ans[[5]]))){
write.table(M,paste(variable$name,'.txt',sep=''),sep="\t",quote=TRUE,dec=as.character(ans[[2]]),na=as.character(ans[[3]]),row.names=FALSE,col.names=TRUE)
	}
	if((!as.logical(ans[[4]]))&(as.logical(ans[[5]]))){
write.table(M,paste(variable$name,'.txt',sep=''),sep="\t",quote=TRUE,dec=as.character(ans[[2]]),na=as.character(ans[[3]]),row.names=TRUE,col.names=FALSE)
	}
	if((!as.logical(ans[[4]]))&(!as.logical(ans[[5]]))){
write.table(M,paste(variable$name,'.txt',sep=''),sep="\t",quote=TRUE,dec=as.character(ans[[2]]),na=as.character(ans[[3]]),row.names=as.logical(ans[[5]]),col.names=as.logical(ans[[4]]))
	}
	assign('previous.name',variable$name,envir=.GlobalEnv,inherits=FALSE)
}else{
	tk_messageBox(type=c("ok"),message='The Variable does not exist!',caption="Input Error")
}
rm(values,ans,row,col,M,nl)
