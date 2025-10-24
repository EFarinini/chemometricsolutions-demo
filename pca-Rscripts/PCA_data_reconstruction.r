# Purpose   : Data reconstruction 
# Input     : matrix with NA
# Output    : matrix without NA
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 3.2
# Licence   : GPL2.1
#
suppressWarnings(suppressPackageStartupMessages(library("Rcpp")))
require(pcaMethods)
ans<-inpboxe4k2(c('* Matrix Name','* Rows to be selected (e.g., "1:10,15" or "15:end")',
'* Columns to be selected (e.g., "1:3,7" or "7:end")',"* Max. number of Components for reconstruction" )
,c('Centered','Scaled'),c(previous.name,'all','all','10','TRUE','TRUE'))
if(!is.null(ans)){
M_<-eval(parse(text=ans[[1]]),envir=.GlobalEnv)
M_0<-M_
if((ans[[2]]!='all')& strsplit(unlist(strsplit(ans[[2]], ',')),':')[[1]][2]=='end')
  ans[[2]]<-paste(strsplit(unlist(strsplit(ans[[2]], ',')),':')[[1]][1],':',nrow(M_))
if((ans[[3]]!='all')& strsplit(unlist(strsplit(ans[[3]], ',')),':')[[1]][2]=='end')
  ans[[3]]<-paste(strsplit(unlist(strsplit(ans[[3]], ',')),':')[[1]][1],':',ncol(M_))
ans_0<-ans
if((ans[[2]]!='all')&(ans[[3]]!='all'))M_<-M_[givedim(ans[[2]]),givedim(ans[[3]])]
if((ans[[2]]!='all')&(ans[[3]]=='all'))M_<-M_[givedim(ans[[2]]),]
if((ans[[2]]=='all')&(ans[[3]]!='all'))M_<-M_[,givedim(ans[[3]])]
if((typeof(M_)=='double')|(typeof(M_)=='list')){
  previous.name<-ans[[1]]
  M.na<-is.na(M_)
  if(sum(is.na(M_))!=0){
    sc<-"none"
    if(as.logical(ans[[6]]))sc<-"uv"
	pre<-as.logical(ans[[5]])
    npc<-min(as.numeric(ans[[4]]),dim(M_))
    res<-pca(M_,method="nipals",center=pre,scale=sc,nPcs=npc)
    V_<-res@R2*100
dev.new(title="scree plot")
    plot(V_,xlab='Component Number',ylab="% Explained Variance",
    main='% Explained Variance',ylim=c(0,max(V_)*1.2),type='n')
    for(i in 1:length(V_)){
      if(V_[i]!=0)points(i,V_[i],col='blue')
    }
    lines(1:i,V_[1:i]);grid()
    ans<-gnfib('* Number of Components for reconstruction',as.character(npc))
    if(!is.null(ans))npc<-as.numeric(ans[[1]])
    M.rec<-fitted(res,nPcs=npc,pre=pre,post=TRUE)
    M.rec[!M.na]<-M_[!M.na]
    M.rec<-as.data.frame(M.rec)
    names(M.rec)<-names(M_)
    row.names(M.rec)<-row.names(M_)
    if((ans_0[[2]]!='all')&(ans_0[[3]]!='all'))M_0[givedim(ans_0[[2]]),givedim(ans_0[[3]])]<-M.rec
    if((ans_0[[2]]!='all')&(ans_0[[3]]=='all'))M_0[givedim(ans_0[[2]]),]<-M.rec
    if((ans_0[[2]]=='all')&(ans_0[[3]]!='all'))M_0[,givedim(ans_0[[3]])]<-M.rec
    if((ans_0[[2]]=='all')&(ans_0[[3]]=='all'))M_0<-M.rec
    assign('M.rec',M_0,envir=.GlobalEnv)
    eval(parse(text=paste(previous.name,'.full','<-M.rec',sep='')),envir=.GlobalEnv)
    print(paste('Matrix with recomputed data is saved in: ',previous.name,'.full',sep=''))
    rm(npc,res,sc,pre,M.na,M.rec,i,V_,M_,M_0,ans_0)
    }else{
    tk_messageBox(type=c("ok"),message='No Missing Data!',caption='Input Error')}
    rm(ans)}
}
