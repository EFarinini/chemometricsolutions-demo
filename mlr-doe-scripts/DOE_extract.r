# Purpose   : extract some relevant vectors and matrices from a DOE object
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 1.0
# Licence   : GPL2.1
#
if(exists("DOE",envir=.GlobalEnv)){
	get("DOE",envir=.GlobalEnv)
	ans<-inpboxc('Extract Matrix:',c('Dispersion Matrix','Coefficients','Fitted Values','Residuals','CV predicted','CV Residuals'))
	if(!is.null(ans)){
		if(ans[[1]]==1){
			Disp<-DOE$disper
			assign('Disp',Disp,envir=.GlobalEnv)
			print('Values saved and exported in "Disp"',quote=FALSE)
		}
		if(ans[[1]]==2){
			Coeff<-t(DOE$b)
			# colnames(Coeff) <- ''
			rownames(Coeff) <- ''
			assign('Coeff',Coeff,envir=.GlobalEnv)
			print('Values saved and exported in "Coeff"',quote=FALSE)
		}
		if(ans[[1]]==3){
			Fitted<-DOE$pred
			colnames(Fitted) <- 'Fitted'
			rownames(Fitted) <- 1:length(DOE$pred)
			assign('Fitted',Fitted,envir=.GlobalEnv)
			print('Values saved and exported in "Fitted"',quote=FALSE)
		}
	  if(ans[[1]]==4){
	    Res<-DOE$pred-DOE$y
	    colnames(Res) <- 'Residuals'
	    rownames(Res) <- 1:length(DOE$pred)
	    assign('Res',Res,envir=.GlobalEnv)
	    print('Values saved and exported in "Res"',quote=FALSE)
	  }
		if(ans[[1]]==5){
			CVpred<-as.matrix(DOE$predcv)
			colnames(CVpred) <- 'CVpred'
			rownames(CVpred) <- 1:length(DOE$predcv)
			assign('CVpred',CVpred,envir=.GlobalEnv)
			print('Values saved and exported in "CVpred"',quote=FALSE)
		}
		if(ans[[1]]==6){
			CVres<-as.matrix(DOE$rescv)
			colnames(CVres) <- 'CVres'
			rownames(CVres) <- 1:length(DOE$rescv)
			assign('CVres',CVres,envir=.GlobalEnv)
			print('Values saved and exported in "CVres"',quote=FALSE)
		}
	}
}else{
	tk_messageBox(type=c("ok"),message='Run Model Evaluation First in DOE!',caption="Input Error")
}


if(ans[[1]]==1) {values="Disp";col=FALSE;row=FALSE}
if(ans[[1]]==2) {values="Coeff";col=FALSE;row=FALSE}
if(ans[[1]]==3) {values="Fitted";col=FALSE;row=FALSE}
if(ans[[1]]==4) {values="Res";col=FALSE;row=FALSE}
if(ans[[1]]==5) {values="CVpred";col=FALSE;row=FALSE}
if(ans[[1]]==6) {values="CVres";col=FALSE;row=FALSE}

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

rm(values,ans,row,col)








