# Authors   : R. Leardi, C. Melzi, G. Polotti



if(exists('DOE',envir=.GlobalEnv)){
  ans<-inpboxrr(c("Experimental variance estimated from:"),
                c("residuals","independent measurements"))
  if(!is.null(ans)){
    res<-ans[[1]]
    if(ans[[2]]){
      ans<-inpboxee(c("experimental standard deviation ","degrees of freedom"),c("",""))
      if(!is.null(ans)){
        s<-as.numeric(ans[[1]])
        dof<-as.numeric(ans[[2]])}}
    if(DOE$loY==FALSE & res){
      tk_messageBox(type=c("ok"),message='Missing Y!',caption="Input Error")
    }else{
      if(res){
        dof<-DOE$dof
        s<-DOE$rmsef}
      if(dof>0){
        ## Generates triangle
        trian <- expand.grid(base=seq(0,1,l=100), high=seq(0,sin(pi/3),l=87))
        trian <- subset(trian, (base*sin(pi/3)*2)>high)
        trian <- subset(trian, ((1-base)*sin(pi/3)*2)>high)
        ## Generates triangle in R^3 X1+X2+X3=1
        new=data.frame(X1=1-trian$base-trian$high/sqrt(3))
        new$X2=trian$high*2/sqrt(3)
        new$X3=trian$base-trian$high/sqrt(3)
        ## Creates model matrix on X1+X2*X3=1 (new)
        P=DOE$x
        n=ncol(P)
        M=DOE$m
        m=ncol(M)
        names_=DOE$name
        if (names_[1]=="X0" & names_[2]=="X1" & names_[3]=="X2") names_[1:3]=c("Comp.1","Comp.2","Comp.3")
        names_coeff<-colnames(DOE$x)
        form<-paste0(names_coeff,collapse = '+')
        form<-paste0('~-1+',form)
        form<-formula(form)
        names(new)<-names_coeff[1:3]
        ## Design in coord (b,h)
        b=1-P[,1]-P[,2]/2
        h=(sqrt(3)/2)*P[,2]
        X=model.matrix(form,data = new)
        ## CI_surface on triangle
        Q=X%*%solve(t(P)%*%P)%*%t(X)
        q_<-qt(0.975,dof)
        Lev=data.frame(trian,"L"=q_*s*sqrt(diag(Q)))
        ## Creates function grid and labels on the axes
        grade.trellis <- function(from=0.2, to=0.8, step=0.2, col=1, lty=2, lwd=0.5){
          x1 <- seq(from, to, step)
          x2 <- x1/2
          y2 <- x1*sqrt(3)/2
          x3 <- (1-x1)*0.5+x1
          y3 <- sqrt(3)/2-x1*sqrt(3)/2
          lattice::panel.segments(x1, 0, x2, y2, col=col, lty=lty, lwd=lwd)
          # lattice::panel.text(x1, 0, label=x1, pos=1)
          lattice::panel.segments(x1, 0, x3, y3, col=col, lty=lty, lwd=lwd)
          # lattice::panel.text(x2, y2, label=rev(x1), pos=2)
          lattice::panel.segments(x2, y2, 1-x2, y2, col=col, lty=lty, lwd=lwd)
          # lattice::panel.text(x3, y3, label=rev(x1), pos=4)
        }
        ## Generates isoleverage plot
        p=lattice::contourplot(L~base*high, Lev,col="red",cuts = 10,aspect = sqrt(3)/2,label=list(col="black",cex=0.8),
                               region = TRUE,col.regions = colorRampPalette(c("yellow","green","blue")),
                               main=paste("Semiampl. of the confidence interval",DOE$Yname),
                               par.settings=list(axis.line=list(col=NA), axis.text=list(col="black")),xlab=NULL, ylab=NULL,
                               scales=list(x=list(alternating=0),y=list(alternating=0)),
                               xlim=c(-0.1,1.1), ylim=c(-0.1,0.96))
        dev.new(title=paste("confidence interval contour plot mixture",DOE$Yname))
        print(p)
        lattice::trellis.focus("panel", 1, 1, highlight=FALSE)
        lattice::panel.segments(c(0,0,0.5), c(0,0,sqrt(3)/2), c(1,1/2,1), c(0,sqrt(3)/2,0))
        lattice::panel.xyplot(x=b,y=h, col="red",pch=20,cex=1.75)
        grade.trellis()
        lattice::panel.text(.53+(nchar(names_[2])-1)/100,0.92,label=names_[2],pos=2)
        lattice::panel.text((nchar(names_[1])-1)/50,-0.05,label=names_[1],pos=2)
        lattice::panel.text(1.01-(nchar(names_[3])-1)/50,-0.05,label=names_[3],pos=4)
        lattice::trellis.unfocus()
        rm(P,n,M,m,form,trian,new,Q,Lev,grade.trellis,b,h,p,X,names_coeff)
      }else{
        tk_messageBox(type=c("ok"),message='No degrees of freedom!',caption="Input Error")
      }
      if (exists("txt")) rm(txt)
      if (exists("i")) rm(i)
      if (exists("names_")) rm(names_)
      if(exists("q_"))rm(q_)
      if(exists("s"))rm(s)
      if(exists("dof"))rm(dof)
      if(exists("res"))rm(res)
}}
}else{
  tk_messageBox(type=c("ok"),message='Run Model Computation First in DOE!',caption="Input Error")}
