previous.name<-'cp'
variable<-gvfib(lb='*Matrix with Candidate Points')
if(length(variable$input)!=0){
  CP<-as.data.frame(variable$value)
  if(exists('expt')){
    m<-min(which(expt[1,]==0))-1
    M<-ncol(expt)
    ans<-inpboxc(c('',''),rownames(expt),inp=0)
    if(!is.null(ans)){
      r_cp<-expt[ans[[1]],][expt[ans[[1]],]!=0]
      print(CP[r_cp,])
      print(paste('The D-optimal design is saved in', paste0("sel",length(r_cp))),quote=FALSE)
      assign(x = paste0("sel",length(r_cp)),value =CP[r_cp,] )
    }
    rm('ans')
  }else{
    tk_messageBox(type=c("ok"),message='Run D-optimal First!',caption="Input Error")}
}

rm_obj<-c('m','M','r_cp','variable','CP')
for (o in 1:length(rm_obj))if(exists(rm_obj[o]))rm(list=rm_obj[o])
rm(rm_obj,o)