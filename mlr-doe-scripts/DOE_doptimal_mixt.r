# Purpose   : function dopt selects a D-optimal design
#             the input matrix is the matrix of candidates points
#             the output matrix contains the codes of the selected experiments
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 1.5
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

		#if(!ans[[2]]) diag(m)<-rep("",nc) #####
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
	

	tot<-as.data.frame(tot)
	names(tot)<-as.character(1:ncol(tot))
	rownames(tot)<-as.character(1:nrow(tot))
	return(tot)
}
previous.name<-"cp"
ans<-inpboxekk('*Matrix with Candidate Points','Model with :',c('Quadratic Terms','Cubic Term'))

int=FALSE ########################da cancellare

if(!is.null(ans))variable<-makevar(ans[[1]])
if((typeof(variable$value)=='double')|(typeof(variable$value)=='list')){
 x<-eval(parse(text=ans[[1]]),envir=.GlobalEnv)
 
 x<-matmod(x,FALSE,ans[[2]])##########################☻
 if(ans[[3]]){
   x<-cbind(x,x[,1]*x[,2]*x[,3])
  # colnames(x)[ncol(x)]<-paste0(var[1],'*',var[2],'*',var[3])
  # coeff_name<-c(coeff_name,paste0(var[1],'*',var[2],'*',var[3]))
 }
 
 x<-as.matrix(x)

 maxinfl<-NULL
 r<-nrow(x)
 co<-ncol(x)
 print(paste('The Model has ',co,' coefficients',sep=''),quote=FALSE)
 ans<-inpboxeeee(c('* Lower Number of Experiments','* Upper Number of Experiments','* Incremental Step','* Number of trials'),c(co,r-1,1,10))
 l<-as.numeric(ans[[1]])
if(l<co){
 l<-co
 print(paste('You cannot have less experiments than coefficients - Lowest number of Experiments will be',co,sep=''),quote=FALSE)}
h<-as.numeric(ans[[2]])
if(h>(r-1)){
 h<-r-1
 print(paste('The Experimental Matrix must have a subset of the candidate points - Highest number of Experiments will be ',r-1,sep=''),quote=FALSE)}
 ne<-as.numeric(ans[[3]])
 nt<-as.numeric(ans[[4]])
 nlog<-seq(from=l,to=h,by=ne)
 expt<-matrix(0,length(nlog),h)
 rownames(expt)<-seq(from=l,to=h,by=ne)
 logmt<-matrix(0,2,length(nlog))
 logmt[1,]<-nlog
 w<-0

 dev.new(title="normalized determinant") 

 for(n in seq(from=l,to=h,by=ne)){
   w<-w+1
   maxt<-0
   for(j in 1:nt){
     miss<-0
     xin<-NULL
     dmax<-0
     o<-randperm(1:r)
     while (miss<5){
       xin<-as.matrix(x[o[1:n],])
       xout<-as.matrix(x[o[(n+1):r],])
       if(ncol(xout)==1)xout<-t(xout)
       d<-det(t(xin)%*%xin)
       if(d>dmax){
         dmax<-d
         if(d>maxt){
           mexp<-o[1:n]
           maxt<-d}
       }else{
           miss<-miss+1}
       levin<-diag(xin%*%ginv(t(xin)%*%xin)%*%t(xin))
       levout<-diag(xout%*%ginv(t(xin)%*%xin)%*%t(xout))
       j1<-o[which.min(levin)]
       j2<-o[which.max(levout)+n]
       o[which.min(levin)]<-j2
       o[which.max(levout)+n]<-j1}
   }
      mexp<-mexp[order(mexp)]
	  print('',quote=FALSE)
      print(paste('Solution with ',n,' experiments',sep=''),quote=FALSE)
      print('Selected Points: ',quote=FALSE)
      print(as.vector(mexp))

      print(paste('Log(det):',round(log10(maxt),4)),quote=FALSE)
      logm<-log10(det(t(x[mexp,])%*%x[mexp,])/n^co)
      print(paste('Log(M):',round(logm,4)),quote=FALSE)
      expt[w,1:n]<-mexp
      logmt[2,w]<-logm

      plot(logmt[1,1:w],logmt[2,1:w],col='red',type='b',xlab='Number of Experiments',
      ylab='log(Normalized Determinant)');grid()
 }

 print('',quote=FALSE)
 print('The matrix is saved in expt',quote=FALSE)
 print('Type expt on the console to see it',quote=FALSE)
 rm(x,nlog,maxinfl,logmt,logm,mexp,o,j1,j2,levout,levin,miss,co,r,n,w,ne,nt)
}

rm(ans,variable,d,dmax,h,j,l,matmod,maxt,xin,xout)
