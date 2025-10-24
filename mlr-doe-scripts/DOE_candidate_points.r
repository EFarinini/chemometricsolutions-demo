# Authors   : R. Leardi, C. Melzi, G. Polotti


ans<-gnfib(paste('*Number of variables? ',sep=''),'2')
if (!is.null(ans)){
   n_var<-as.numeric(ans)

for(i in 1:n_var){
ans<-inpboxe1k1(paste('*Levels of variable ',i,'? (e.g., "10,15,20")',sep=''),'The variable is numeric',c('',TRUE))
tv_num<-ans[[2]]
 if(!is.null(ans)){
    if(tv_num)lv<-as.numeric(unlist(strsplit(ans[[1]],",")))
    if(!tv_num)lv<-(unlist(strsplit(ans[[1]],",")))
   assign(paste("x",i,sep=""),lv)
 }else{
   stop(paste("Missing levels of variable x"),i,"!",sep="")
 }
}

e_g<-"expand.grid(x1"
c_n<-"x1"
for(i in 2:n_var){
  e_g<-paste(e_g,",x",i,sep="")
  c_n<-paste(c_n,",x",i,sep="")
}
e_g<-paste(e_g,")",sep="")

cp<-eval(parse(text=e_g))
colnames(cp)<-unlist(strsplit(c_n,","))
print("The matrix of candidate points is saved in cp")

rm(list = unlist(strsplit(c_n,",")))
rm(n_var,ans,e_g,c_n,lv,i)
if(exists('tv_num'))rm(tv_num)
}



