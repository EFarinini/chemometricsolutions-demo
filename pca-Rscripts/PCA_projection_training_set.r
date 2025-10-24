# Purpose   : draw a new dataset in the plane score of a PCA object
# Input     : M_ new dataset
#             PCA object provide by several routines correlated
#             c1_,c2_ integer numbers of the two componets that must be drawn
#             lbd, vector of string with the names of the variables
#             bc, bs binary choices for centering and scaling
# Output    : 2D plot with old data in black and new data in red
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 1.8
# Licence   : GPL2.1

if (exists("PCA",envir=.GlobalEnv)){
  if(!exists('pca.prj.set'))pca.prj.set<-c(previous.name,'all','all','None','1','2','None','FALSE','FALSE')
  ans<-inpboxe7k2(c('*External Data Set','*Rows to be selected (e.g., "1:10,15" or "15:end")','*Columns to be selected (e.g., "1:3,7" or "7:end")',
                    'Label Vector for external set (e.g., A[,1])','*Component on x-axis','*Component on y-axis','Label Vector for training set (e.g., A[,1])',
                    'Row Names','Ellipses'),pca.prj.set)
  if(!is.null(ans)){
    previous.name<-ans[[1]]
    pca.prj.set<-ans
    # if((as.logical(ans[[9]]))&(!PCA$center | !PCA$scale)){
    #   ans<-NULL
    #   tk_messageBox(type=c("ok"),message='No Ellipse without Autoscale!',caption="Input Error")
    # }
    c1_<-as.integer(ans[[5]])
    c2_<-as.integer(ans[[6]])
    lb_<-NULL
    #if(as.logical(ans[[8]]))lb_<-row.names(PCA$dataset)
    if(ans[[7]]!='None')lb_<-eval(parse(text=ans[[7]]))
    M_<-eval(parse(text=ans[[1]]))
    
    if((ans[[2]]!='all' & grepl(':', ans[[2]], fixed = TRUE)) & strsplit(unlist(strsplit(ans[[2]], ',')),':')[[1]][2]=='end')
      ans[[2]]<-paste(strsplit(unlist(strsplit(ans[[2]], ',')),':')[[1]][1],':',nrow(M_))
    if((ans[[3]]!='all' & grepl(':', ans[[3]], fixed = TRUE) ) & strsplit(unlist(strsplit(ans[[3]], ',')),':')[[1]][2]=='end')
      ans[[3]]<-paste(strsplit(unlist(strsplit(ans[[3]], ',')),':')[[1]][1],':',ncol(M_))
    if((ans[[2]]!='all')&(ans[[3]]!='all'))M_<-M_[givedim(ans[[2]]),givedim(ans[[3]])]
    if((ans[[2]]!='all')&(ans[[3]]=='all'))M_<-M_[givedim(ans[[2]]),]
    if((ans[[2]]=='all')&(ans[[3]]!='all'))M_<-M_[,givedim(ans[[3]])]
    
    if(sum(is.na(M_))!=0){
      print('>>NA found: remove them before evaluation<<')
    }else{
      
      lbd<-NULL
      if(as.logical(ans[[8]]))lbd<-row.names(M_)
      if(ans[[4]]!='None')lbd<-eval(parse(text=ans[[4]]))
      
      # standard score evaluation
      S_<-PCA$res@scores
      v1_<-PCA$res@R2[c1_]*100
      v2_<-PCA$res@R2[c2_]*100
      r<-nrow(S_)
      c<-nrow(PCA[[1]]@loadings)
      if(!PCA$scale)c <- sum(apply(PCA$dataset,2,'var'))
      
      yn.lb<-TRUE
      if(is.null(lb_))yn.lb<-FALSE
      if(yn.lb){
        if(length(lb_)!=nrow(S_)){
          tk_messageBox(type=c("ok"),message='Wrong Score Label Dimension !',caption="Input Error")}}
      # new dataset evaluation
      T_<-PCA$res@loadings
      unity<-matrix(rep(1,nrow(M_)),nrow(M_),1)
      if(PCA$center)M_<-M_-(unity%*%PCA$centered)
      if(PCA$scale)M_<-M_/(unity%*%PCA$scaled)
      D_<-as.matrix(M_) %*% T_
      # plot standard score plot in the new scale 
      
      # Slim<-c(min(S_[,c(c1_,c2_)],D_[,c(c1_,c2_)]),max(S_[,c(c1_,c2_)],D_[,c(c1_,c2_)]))
      

      DeltaS1lim=(max(S_[,c1_],D_[,c1_])-min(S_[,c1_],D_[,c1_]))
      DeltaS2lim=(max(S_[,c2_],D_[,c2_])-min(S_[,c2_],D_[,c2_]))
      
      if (DeltaS1lim>DeltaS2lim){
        Delta<-DeltaS1lim-DeltaS2lim
        S1lim<-c(min(S_[,c1_],D_[,c1_])-DeltaS1lim*0.05,max(S_[,c1_],D_[,c1_])+DeltaS1lim*0.05)
        S2lim<-c(min(S_[,c2_],D_[,c2_])-Delta/2-DeltaS1lim*0.05,max(S_[,c2_],D_[,c2_])+Delta/2+DeltaS1lim*0.05)
      }
      if (DeltaS2lim>DeltaS1lim){
        Delta<-DeltaS2lim-DeltaS1lim
        S1lim<-c(min(S_[,c1_],D_[,c1_])-Delta/2-DeltaS2lim*0.05,max(S_[,c1_],D_[,c1_])+Delta/2+DeltaS2lim*0.05)
        S2lim<-c(min(S_[,c2_],D_[,c2_])-DeltaS2lim*0.05,max(S_[,c2_],D_[,c2_])+DeltaS2lim*0.05)
      }

      if(PCA$type=='pca'){
        xl_<-paste('Component ',as.character(c1_),' (',as.character(round(v1_,1)),'% of variance)',sep='')
        yl_<-paste('Component ',as.character(c2_),' (',as.character(round(v2_,1)),'% of variance)',sep='')
      }else{
        xl_<-paste('Factor ',as.character(c1_),' (',as.character(round(v1_,1)),'% of variance)',sep='')
        yl_<-paste('Factor ',as.character(c2_),' (',as.character(round(v2_,1)),'% of variance)',sep='')
      }
      tl_=paste('Score Plot (',as.character(round((v1_+v2_),1)),'% of total variance)',sep='')
      
      dev.new(title="PCA score plot projection")
      op<-par(pty='s')
      if(!yn.lb){
        plot(S_[,c(c1_,c2_)],xlim=S1lim,ylim=S2lim,pty='o',xlab=xl_,ylab=yl_,col='black',cex=0.4)}
      if(yn.lb){
        plot(S_[,c(c1_,c2_)],xlim=S1lim,ylim=S2lim,xlab=xl_,ylab=yl_,type='n')
        text(S_[,c(c1_,c2_)],as.character(lb_),col='black',cex=0.6)
      }
      grid()
      par(new=TRUE)
      
      if(as.logical(ans[[9]])){
        title(main=tl_,sub='Training: black - External: red - Ellipses: critical T^2 value at p=0.05, 0.01 and 0.001',cex.main=1.2,font.main=2,
              col.main="black",cex.sub=0.6,font.sub=2,col.sub="red")
        
        op<-par(pty='s')
        par(new=TRUE)
        
        rad1=sqrt((v1_/100*((r-1)/r)*c)*qf(.95,2,r-2)*2*(r^2-1)/(r*(r-2))); 
        rad2=sqrt((v2_/100*((r-1)/r)*c)*qf(.95,2,r-2)*2*(r^2-1)/(r*(r-2)));
        theta <- seq(0, 2 * pi, length=1000)
        x <- rad1 * cos(theta)
        y <- rad2 * sin(theta)
        plot(x, y, type = "l",col='black',xlim=S1lim,ylim=S2lim,xlab='',ylab='')
        par(new=TRUE)
        rad1=sqrt((v1_/100*((r-1)/r)*c)*qf(.99,2,r-2)*2*(r^2-1)/(r*(r-2))); 
        rad2=sqrt((v2_/100*((r-1)/r)*c)*qf(.99,2,r-2)*2*(r^2-1)/(r*(r-2)));
        theta <- seq(0, 2 * pi, length=1000)
        x <- rad1 * cos(theta)
        y <- rad2 * sin(theta)
        plot(x, y, type = "l",col='black',xlim=S1lim,ylim=S2lim,xlab='',ylab='',lty=2)
        par(new=TRUE)
        rad1=sqrt((v1_/100*((r-1)/r)*c)*qf(.999,2,r-2)*2*(r^2-1)/(r*(r-2))); 
        rad2=sqrt((v2_/100*((r-1)/r)*c)*qf(.999,2,r-2)*2*(r^2-1)/(r*(r-2)));
        theta <- seq(0, 2 * pi, length=1000)
        x <- rad1 * cos(theta)
        y <- rad2 * sin(theta)
        plot(x, y, type = "l",col='black',xlim=S1lim,ylim=S2lim,xlab='',ylab='',lty=3)
        par(new=TRUE)
        
      }else{
        title(main=tl_,sub='Training: black - External: red',cex.main=1.2,font.main=2,
              col.main="black",cex.sub=0.6,font.sub=2,col.sub="red")
      }
      # new dataset plot
      ynld<-TRUE
      nd<-nrow(D_)
      if(is.null(lbd))ynld<-FALSE 
      if(ynld){
        if(length(lbd)!=nd){
          tk_messageBox(type=c("ok"),message='Wrong Dataset Label Dimension !',caption="Input Error")}}
      if(!ynld)points(D_[,c1_],D_[,c2_],col='red')
      if(ynld){points(D_[,c1_],D_[,c2_],type='n')
        text(D_[,c1_],D_[,c2_],as.character(lbd),col='red',cex=0.7)}
      par(op)
      # save new coordinates
      assign('scores.ext',D_,envir=.GlobalEnv)
      print('The scores of the external set are saved in matrix scores.ext')
      rm(ans,c1_,c2_,lb_,lbd,M_,S_,v1_,v2_,yn.lb,ynld,unity,D_,T_,S1lim,S2lim,xl_,yl_,tl_,nd,op)}}
}else{
  tk_messageBox(type=c("ok"),message='Run Matrix Evaluation First!',caption="Input Error")}
  
  
  
  
  
  

