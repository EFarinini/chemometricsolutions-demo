# Purpose   : detect the non dominated points according to Pareto optimality
# Input     : data set containing the responses on which the non dominated points will be selected
# Output    : list of the non dominated points 
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: CM
# Version   : 3.1
# Licence   : GPL2.1

suppressWarnings(suppressPackageStartupMessages(library("rPref")))
library(rPref)
if(!exists('pareto.set'))pareto.set<-c(previous.name,'all')
ans<-inpboxee(c('*Matrix Name','*Variables to be selected (e.g., 1:4,8)'),
              pareto.set)
if (!is.null(ans)){
  previous.name<-ans[[1]]
  pareto.set<-ans
  M_<-eval(parse(text=ans[[1]]),envir=.GlobalEnv)
  if(ans[[2]]!="all")M_<-M_[,givedim(ans[[2]])];M_i<-M_
    frm<-"high(M_[,1])"
    ans<-inpboxr2k(colnames(M_)[1],c("max","min"),'Target',FALSE)
    if(is.null(ans)){
        opt <- options(show.error.messages=FALSE)
        on.exit(options(opt))
        if(exists('frm'))rm(frm)
        if(exists('M_'))rm(M_)
        if(exists('i'))rm(i)
        if(exists('M_i'))rm(M_i)
        if(exists('opt'))rm(opt)
        stop()
    }
    if(ans[[3]]==FALSE){
      if(ans[[1]]==FALSE)frm<-"low(M_[,1])"
    }else{
      ans<-gnfib(paste('*Target value? ',sep=''),'')
      M_[,1]<-abs(M_[,1]-as.numeric(ans))
      frm<-"low(M_[,1])"
    }
    for (i in 2:ncol(M_)){
      ans<-inpboxr2k(colnames(M_)[i],c("max","min"),'Target',FALSE)
      if(is.null(ans)){
        opt <- options(show.error.messages=FALSE)
        on.exit(options(opt))
        if(exists('frm'))rm(frm)
        if(exists('M_'))rm(M_)
        if(exists('i'))rm(i)
        if(exists('M_i'))rm(M_i)
        if(exists('opt'))rm(opt)
        stop()
      }
      if(ans[[3]]==FALSE){
        if(ans[[1]]==TRUE){
          frm<-paste(frm,"*high(M_[,",i,"])",sep="")
        }else{
          frm<-paste(frm,"*low(M_[,",i,"])",sep="")
        }
      }else{
        ans<-gnfib(paste('*Target value? ',sep=''),'')
        M_[,i]<-abs(M_[,i]-as.numeric(ans))
        frm<-paste(frm,"*low(M_[,",i,"])",sep="")
      }
    }   
frm<-eval(parse(text=frm))
#nondom <- psel(M_,frm)
nondom<-M_i[row.names(psel(M_,frm)),]
print(nondom)
print('The list of the non dominated points is saved in: nondom',quote=FALSE)

if(exists('frm'))rm(frm)
if(exists('M_'))rm(M_)
if(exists('i'))rm(i)
if(exists('M_i'))rm(M_i)
if(exists('opt'))rm(opt)

}


