# Purpose   : function doptadd  D-optimal design by which the best experiments to be added
#             the input matrix is the matrix of candidates points
#             the input matrix is the matrix of already made experiments
#             the output matrix contains the codes of the selected experiments
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 1.4
# Licence   : GPL2.1
#
require(MASS)
require(graphics)
require(pracma)
require(matrixcalc)
require(RGtk2Extras)
matmod<-function(xexp,lI,lHT){
	tot<-xexp
	nr<-nrow(xexp)
	nc<-ncol(xexp)
	m<-matrix(rep(0,nc*nc),nc,nc)
	if(as.logical(lHT)){
		m<-matrix(rep(1,nc*nc),nc,nc)
		diag(m)<-rep(0,nc)
		m<-as.data.frame(m)
		names(m)<-names(xexp)
		row.names(m)<-names(m)
		m<-as.data.frame(upper.triangle(as.matrix(m)))
 
    for (i in 1:nc){
	     for (j in 1 : nc){
	       	if (i > j) m[i,j]=""}}

    for (i in 1:nc){
	    if (length(unique(xexp[,i]))==2) m[i,i]=""
          if (length(unique(xexp[,i]))>2) m[i,i]=1
	    if (length(unique(xexp[,i]))==2 & min(unique(xexp[,i]))== 0 & max(unique(xexp[,i])) == 1){
		   for (j in 1:nc) {
		    	m[i,j]=""
		    	m[j,i]=""}}
    }

 	if(!ans[[3]]) diag(m)<-rep("",nc)
		
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
				tot<-cbind(tot,tot[,j1]*tot[,j2])
			}
		}
	}
	z<-nc*(nc-1)/2
	for(j1 in 1:nc){
		z<-z+1
		if(m[j1,j1]==1){
			coeff[z]<-coeff[z]+1
			a<-a+1
			tot<-cbind(tot,tot[,j1]^2)
		}
	}
	if(as.logical(lI)){
		coeff[nc+nc*(nc-1)/2+1]<-1
		a<-a+1
		tot<-cbind(rep(1,nr),tot)
	}
	tot<-as.data.frame(tot)
	names(tot)<-as.character(1:ncol(tot))
	rownames(tot)<-as.character(1:nrow(tot))
	return(tot)
}
ans<-inpboxeekk(c('*Matrix with Performed Experiments','*Matrix with Candidate Points'),
c('Intercept','Higher Terms'),c('','','TRUE','TRUE'))
int=ans[[3]]
if(!is.null(ans)){
	previous.name<-ans[[1]]
	xori<-eval(parse(text=ans[[1]]),envir=.GlobalEnv)
	x<-eval(parse(text=ans[[2]]),envir=.GlobalEnv)
	r<-nrow(x)
	co<-ncol(x)
	ror<-nrow(xori)
	# ror<-nrow(unique(xori))
	tot<-rbind(xori,x)
	tot<-matmod(tot,ans[[3]],ans[[4]])
	maxinfl<-NULL
	rtot<-nrow(tot)
	co<-ncol(tot)
	xori<-as.matrix(tot[1:ror,])
	x<-as.matrix(tot[(ror+1):rtot,])
	print(paste('The Model has ',co,' coefficients',sep=''),quote=FALSE)
	print(paste(ror,' experiments have already been performed',sep=''),quote=FALSE)
	nnt<-co-ror
	if(nnt<1)nnt<-1
	ans<-inpboxeeee(c('*Lower Number of Experiments','*Upper Number of Experiments','*Incremental Step','*Number of trials'),c(nnt,r-1,1,10))
	l<-as.numeric(ans[[1]])
	if(l+ror<co){
		l<-co
		print(paste('You cannot have less experiments than coefficients - Lowest number of Experiments will be',co,sep=''),quote=FALSE)
	}
	h<-as.numeric(ans[[2]])
      ne<-as.numeric(ans[[3]])
	nt<-as.numeric(ans[[4]])
	nlog<-seq(from=l,to=h,by=ne)
	expt<-matrix(0,length(nlog),h)
	rownames(expt)<-seq(from=l,to=h,by=ne)
	logmt<-matrix(0,2,length(nlog))
	logmt[1,]<-nlog
	w<-0

dev.new(title="normalized determinant")
vec_rk <- NULL

	for(n in seq(from=l,to=h,by=ne)){
		w<-w+1
		maxt<-0
		for(j in 1:nt){
			miss<-0
			xin<-NULL
			dmax<-0
			o<-randperm(1:r)
			while (miss<5){
				xin<-x[o[1:n],]
				xin<-matrix(xin,nrow=n,ncol=ncol(x))
				xout<-as.matrix(x[o[(n+1):r],])
				if(ncol(xout)==1)xout<-t(xout)
				totin<-rbind(xori,xin)
				d<-det(t(totin)%*%totin)
				if(d>dmax){
					dmax<-d
					if(d>maxt){
						mexp<-o[1:n]
						mexp<-as.vector(mexp)
						maxt<-d
					}
				}else{
					miss<-miss+1
				}
				levin<-diag(xin%*%ginv(t(totin)%*%totin)%*%t(xin))
				levout<-diag(xout%*%ginv(t(totin)%*%totin)%*%t(xout))
				j1<-o[which.min(levin)]
				j2<-o[which.max(levout)+n]
				o[which.min(levin)]<-j2
				o[which.max(levout)+n]<-j1
			}
		}

if(qr(totin)$rank<co){
  vec_rk <- c(vec_rk,n)
  next
}

		if(miss<=5){
			mexp<-mexp[order(mexp)]
			print('',quote=FALSE)
                        print(paste('Solution with ',n,' added experiments',sep=''),quote=FALSE)
			print('Selected Points: ',quote=FALSE)
			print(as.vector(mexp))
			print(paste('Log(det):',round(log10(maxt),4)),quote=FALSE)
			totin<-rbind(xori,x[mexp,])
			logm<-log10(det(t(totin)%*%totin)/nrow(totin)^co)
			print(paste('Log(M):',round(logm,4)),quote=FALSE)
			expt[w,1:n]<-mexp
			logmt[2,w]<-logm
	#       computing inflation factors
if (int){
			sel<-as.matrix(totin)
			rsel<-nrow(sel)
			csel<-ncol(sel)
			xcc<-sel-matrix(1,rsel,1)%*%apply(sel,2,mean)
			infl<-apply((xcc^2),2,sum)*diag(pinv(t(sel)%*%sel))
			maxinfl<-c(maxinfl,max(infl))
			print('Inflation Factors:',quote=FALSE)
			print(as.vector(round(infl,4)),quote=FALSE)
rm(sel,rsel,csel,infl,xcc)
}	

			if(is.null(vec_rk))	{
			  plot(logmt[1,1:w],logmt[2,1:w],col='red',type='b',xlab='Number of Additional Experiments',ylab='log(Normalized Determinant)')
			}else{
			  plot(logmt[1,1:w][-(vec_rk-nnt+1)],logmt[2,1:w][-(vec_rk-nnt+1)],col='red',type='b',xlab='Number of Additional Experiments',ylab='log(Normalized Determinant)')
			}

			grid()
		}else{
			print('The matrix is almost singular, I cannot evaluate the determinant.',quote=FALSE)
			break
		}
	}
	if(miss<=5){
		if(int){
		dev.new(title="maximum VIF")
		n1 <- length(logmt[1,1:w])
		n2 <- length(maxinfl)
		if(is.null(vec_rk)){
		  plot(logmt[1,1:w],maxinfl,ylim=c(1,max(maxinfl,8)),col='red',
		       type='b',xlab='Number of Experiments',ylab='Maximum Inflation Factor')
		}else{
		  plot(logmt[1,1:w][-(vec_rk-nnt+1)],maxinfl,ylim=c(1,max(maxinfl,8)),col='red',
		       type='b',xlab='Number of Experiments',ylab='Maximum Inflation Factor')
		}
		grid()
		abline(h=4,lty=2,col='red')
		abline(h=8,lty=2,col='red')
		rm(n1,n2)
		}
		 print('',quote=FALSE)
		 print('The matrix is saved in expt',quote=FALSE)
		 print('Type expt on the console to see it',quote=FALSE)
		 
		 if(!is.null(vec_rk))expt <- expt[-which(row.names(expt)==vec_rk),]

		 rm(x,nlog,maxinfl,logmt,logm,mexp,o,j1,j2,levout,levin,co,r,n,w,ne,nt,d,dmax,h,j,l,matmod,maxt,nnt,ror,rtot,tot,totin,xin,xori,xout,miss,vec_rk)
	}

}
rm(ans,int)
