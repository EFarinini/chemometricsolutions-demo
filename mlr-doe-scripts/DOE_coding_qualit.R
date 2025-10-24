# Purpose   : transformation of qualitative variables at more than two levels into dummy variables coding
# Input     : experimental matrix with the qualitative variables being one column each
#             the levels of the qualitative variables must be alphanumeric
# Output    : experimental matrix with each qualitative variable transformed into n-1 columns
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: CM
# Version   : 1.3
# Licence   : GPL2.1

if(!exists('cq.set'))cq.set<-c(previous.name,'','','')
ans0<-inpboxeee('*Matrix name','*First column','*Last column',vinp=cq.set)

if(!is.null(ans0)){
  if(ans0[[2]]!='' & ans0[[3]]!=''){
    previous.name<-ans0[[1]]
    cq.set<-ans0
    var.trans<-eval(parse(text=ans0[[1]]),envir=.GlobalEnv)
    M_<-var.trans[,ans0[[2]]:ans0[[3]],drop=FALSE]
    m<-nrow(M_)
    n<-ncol(M_)
    Y<-list(NULL)
    for(i in 1:n){
      if(is.factor(M_[,i]) | is.character(M_[,i])){
        M_[,i]<-as.factor(M_[,i])
        k<-length(levels(M_[,i]))
        X<-matrix(rep(0,m*k),nrow = m,ncol = k)
        for(j in 1:k)X[,j]<-as.numeric(M_[,i]==levels(M_[,i])[j])
        X<-as.data.frame(X)
        colnames(X)<-paste0(colnames(M_)[i],'.',levels(M_[,i]))
        nl<-inpboxc('*Implicit Level:',levels(M_[,i]),inp=0)
        if(is.null(nl))stop(call. =FALSE,'Choose Implicit Level')
        if(!is.null(nl))Y[[i]]<-X[,-nl[[1]],drop=FALSE]
      }else{
        Y[[i]]<-M_[,i,drop=FALSE]
      }
    }
    M_.cod<-Y[[1]]
    if(n>1)for(i in 2:n)M_.cod<-cbind.data.frame(M_.cod,Y[[i]])
    
    if(ans0[[2]]==1&ans0[[3]]<ncol(var.trans))M_.cod<-cbind.data.frame(M_.cod,
                                                            var.trans[,(as.numeric(ans0[[3]])+1):ncol(var.trans),
                                                                      drop=FALSE])
    if(ans0[[2]]>1&ans0[[3]]<ncol(var.trans))M_.cod<-cbind.data.frame(var.trans[,1:(as.numeric(ans0[[2]])-1),drop=FALSE],M_.cod,
                                                          var.trans[,(as.numeric(ans0[[3]])+1):ncol(var.trans),
                                                                    drop=FALSE])
    if(ans0[[2]]>1&ans0[[3]]==ncol(var.trans))M_.cod<-cbind.data.frame(var.trans[,1:(as.numeric(ans0[[2]])-1),drop=FALSE],M_.cod)
    
    assign(paste(ans0[[1]],'.cod',sep=''),M_.cod,envir=.GlobalEnv)
    print(paste("The matrix of candidate points is saved in",paste(ans0[[1]],'.cod',sep='')))
    rm_obj<-c('M_','M_.cod','m','n','Y','X','i','j','k','var.trans','ans0','nl','ans')
    for (o in 1:length(rm_obj))if(exists(rm_obj[o]))rm(list=rm_obj[o])
    rm(rm_obj,o)
  }
}

