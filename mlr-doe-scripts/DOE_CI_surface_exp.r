# Purpose   : Plot of Semiamplitude of the 0.05 confidence interval from a OLS object. Contour plot.
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 1.3
# Licence   : GPL2.1
#
if(exists('DOE',envir=.GlobalEnv)){
  ans<-inpboxrr(c("Prediction variance estimated from:"),
                c("residuals","independent measurements"))
  if(!is.null(ans)){
    res<-ans[[1]]
    if(ans[[2]]){
      ans<-inpboxee(c("Prediction standard deviation","degrees of freedom"),c("",""))
      if(!is.null(ans)){
        s<-as.numeric(ans[[1]])
        dof<-as.numeric(ans[[2]])}}
    if(res){
      ans<-inpboxrr(c("Experimental variance estimated from:"),
                    c("residuals","independent measurements"))
      if(!is.null(ans)){
        res_e<-ans[[1]]
        if(ans[[2]]){
          ans<-inpboxee(c("experimental standard deviation","degrees of freedom"),c("",""))
          if(!is.null(ans)){
            s_e<-as.numeric(ans[[1]])
            dof_e<-as.numeric(ans[[2]])
          }}
      }else{stop(call.=FALSE,domain = 'NA')}
      }else{
      res_e=FALSE
      s_e=s
      dof_e=dof
    }
      n_exp=inpboxe("How many replicates",1,'Input')
      n_exp=as.numeric(n_exp)
      if(DOE$loY==FALSE & res){
        tk_messageBox(type=c("ok"),message='Missing Y!',caption="Input Error")
      }else{
        if(res){
          dof<-DOE$dof
          s<-DOE$rmsef}
        if(res_e){
          dof_e<-DOE$dof
          s_e<-DOE$rmsef}
        if(dof>0&dof_e>0){
          require(graphics)
          require(RGtk2Extras)
          if(!exists('minrange',envir=.GlobalEnv)){
            ans<-inpboxee(c('*Minimum value of range','*Maximum value of range'),c('-1','1'))
            minrange<-as.numeric(ans[[1]])
            maxrange<-as.numeric(ans[[2]])
          }
          nv<-DOE$nv
          coeff<-DOE$coeff
          disper<-DOE$disper
          z<-0
          nstep<-30
          st<-(maxrange-minrange)/nstep
          lab<-seq(minrange,maxrange,by=st)
          r<-(nstep+1)^2
          c<-nv
          gr<-matrix(0,r,c)
          if(!exists('v1',envir=.GlobalEnv)){
            ans<-inpboxcc(c('Index of the variable on X-axis','Index of the variable on Y-axis'),
                          as.character(1:nv),as.character(1:nv))
            v1<-as.numeric(ans[[1]])
            v2<-as.numeric(ans[[2]])
          }
          a<-0;x<-NULL
          vv<-rep(0,nv)
          for(ij in seq(minrange,maxrange,by=st)){
            for(y in seq(minrange,maxrange,by=st)){
              a<-a+1
              gr[a,v1]=ij
              gr[a,v2]=y}}
          for(i in 1:nv){
            if((i!=v1)&(i!=v2)){
              if(!exists('.vv_0',envir=.GlobalEnv)){
                ans<-gnfib(paste('*Value of variable ',i,' (',DOE$name[i],') ? ',sep=''),'0')
                x<-as.numeric(ans[[1]])
              }else{
                x <- .vv_0[i]
              }
              vv[i]<-x
              if(x!=0){
                gr[,i]<-gr[,i]+x}}}
          vv<-as.data.frame(t(vv[-c(v1,v2)])) 
          colnames(vv)<-DOE$name[-c(v1,v2)] 
          row.names(vv)<-c('')
          if(!exists('text.subtl')){
            text.subtl <- NULL
            if(nv>2){
              text.subtl <-  paste0(colnames(vv)[1],'=',vv[1])
              if(length(vv)>1){
                for(i in 2:length(vv)){
                  text.subtl <- paste0(text.subtl,', ', names(vv)[i],'=',vv[i])
                }
                
              }
            }
          }
          a<-c
          for(j1 in 1:(c-1)){
            for(j2 in (j1+1):c){
              z<-z+1
              if(coeff[z]==1){
                a<-a+1
                gr<-cbind(gr,gr[,j1]*gr[,j2])}}}
          for(j1 in 1:c){
            z<-z+1
            if(coeff[z]==1){
              a<-a+1
              gr<-cbind(gr,gr[,j1]^2)}}
          z<-z+1
          if(coeff[z]==1){
            a<-a+1
            gr<-cbind(rep(1,r),gr)}
          ir<-1
          q<-qt(0.975,dof)
          q_e<-qt(0.975,dof_e)
          lev<-data.frame(Y=sqrt((q*s*sqrt(diag(gr%*%disper%*%t(gr))))^2+(q_e*s_e*sqrt(1/n_exp))^2),gr)
          lev<-lev[,c(1,v1+2,v2+2)]
          names(lev)<-c("z","x","y")
          print(paste('Semiamplitude ',n_exp,' replicates',': min. ',format(min(lev$z),digits=4),' average ',
                      format(mean(lev$z),digits=4),' max. ',format(max(lev$z),digits=4),sep=''))
          dev.new(title=paste("exper. conf. int. surface",DOE$Yname))
          zlab<-NULL
          if(!is.null(x)){zlab<-paste('Resp.at',format(x,digits=4))}
          confint.3D<-lattice::wireframe(z~x*y,data=lev,drape=TRUE,col.regions = colorRampPalette(c("yellow","green","blue"))(256),
                                         at=seq(min(lev$z),max(lev$z),length.out=256),
                                         main=paste("Semiampl. exper. conf. int.",DOE$Yname),
                                         par.settings=list(
                                           par.sub.text = list(font = 1,cex=0.7,
                                                               just = "left", 
                                                               x = grid::unit(5, "mm"))),
                                        sub=paste(text.subtl,paste(n_exp,' replicates')), 
                                        cex.main=0.8,xlab=DOE$name[v1],
                                         ylab=DOE$name[v2],zlab=paste('conf. int.')) 
          print(confint.3D)
          dev.new(title=paste("exper. conf. int. contour plot",DOE$Yname))
          confint.cont<-lattice::contourplot(z~x*y,data=lev,cuts=15,main=paste("Semiampl. experim. conf. int.",DOE$Yname),
                                             par.settings=list(
                                               par.sub.text = list(font = 1,cex=0.7,
                                                                   just = "left", 
                                                                   x = grid::unit(5, "mm"))),
                                             sub=paste(text.subtl,paste(n_exp,' replicates')),
                                             cex.main=0.9,xlab=DOE$name[v1],ylab=DOE$name[v2],
                                             col='red',labels=list(col="red",cex=0.9))
          confint.cont<-update(confint.cont,asp=1)
          print(confint.cont)
          if(nv>2)print(vv)
          rm(nv,coeff,minrange,maxrange,r,c,ans,v1,v2,lab,st,nstep,a,j1,j2,ij,x,y,z,gr,
             lev,disper,ir,vv)
        }else{
          tk_messageBox(type=c("ok"),message='No degrees of freedom!',caption="Input Error")
        }
        if(!is.function(q))rm(q)
        if(exists("s"))rm(s)
        if(exists("dof"))rm(dof)
        if(exists("res"))rm(res)
        if(exists("s_e"))rm(s_e)
        if(exists("dof_e"))rm(dof_e)
        if(exists("res_e"))rm(res_e)
        if(exists("q_e"))rm(q_e)
        if(exists("i"))rm(i)
        if(exists("zlab"))rm(zlab)
        if(exists("text.subtl"))rm(text.subtl)
        if(exists(".vv_0"))rm(.vv_0)
        if(exists("n_exp"))rm(n_exp)
      }}
}else{
  tk_messageBox(type=c("ok"),message='Run Model Computation First in DOE!',caption="Input Error")}
        
        
        
        
        
        
        
        
        
        
        
        

 

    
    








