# Authors   : R. Leardi, C. Melzi, G. Polotti
ans=inpboxeee(tentry = "Lower constraints",tentry1 = "Upper constraints",tentry2 = "Grid steps",
              vinp = c("c(0,0,0)","c(1,1,1)","c(0.05,0.05,0.05)"))

if(!is.null(ans)){

vinc.inf=eval(parse(text=ans[[1]]))
vinc.sup=eval(parse(text=ans[[2]]))
delta=eval(parse(text=ans[[3]]))

p=list(NULL)
y=list(NULL)
for (i in 1:3){
  p[[i]]=seq(vinc.inf[i],vinc.sup[i],delta[i])
  y[i]=paste("X",i,sep="")
}
Dom=expand.grid(p)
  
  q=c(NULL)
  for(i in 1:length(delta)){
    for(j in 1:10){
      if (delta[i]*10^j>=1 & !delta[i]*10^(j-1)>=1) q[i]=j
    }
  }
  
  q=max(q)
  cond=round(apply(Dom,1,sum),q)
  
  cp=Dom[cond==1,]

 if (nrow(cp)==0) stop("inconsistent constraints",call.=FALSE)

  
 rm(ans,p,y,Dom,q)
 colnames(cp)=c("X1","X2","X3")
 rownames(cp)=c(1:nrow(cp))

 ## Generates triangle for the plot (on a plane, two coordinates)
 trian <- expand.grid(base=seq(0,1,l=100), high=seq(0,sin(pi/3),l=87))
 trian <- subset(trian, (base*sin(pi/3)*2)>high)
 trian <- subset(trian, ((1-base)*sin(pi/3)*2)>high)
 
 ## Generates triangle in R^3 X1+X2+X3=1
 new=data.frame(X1=1-trian$base-trian$high/sqrt(3))
 new$X2=trian$high*2/sqrt(3)
 new$X3=trian$base-trian$high/sqrt(3)
 
 ## Builds data.frame triangle in 2 (base,high), 3 (X1,X2,X3) variables e column condition (cond)
 cond=new$X1>=vinc.inf[1] & new$X2>=vinc.inf[2] & new$X3>=vinc.inf[3] & new$X1<=vinc.sup[1] & new$X2<=vinc.sup[2] & new$X3<=vinc.sup[3]
 trian.new.cond=cbind(trian,new,cond)
 
 ## Builds triangle in 2 (base,high), 3 (X1,X2,X3) variables satisfying constraints
 trian.cond=trian.new.cond[trian.new.cond$cond==TRUE,1:2]
 new.cond=trian.new.cond[trian.new.cond$cond==TRUE,3:5]
 
 ## Creates function set grid e axis labels
 grade.trellis <- function(from=0.2, to=0.8, step=0.2, col=1, lty=2, lwd=0.5){
   x1 <- seq(from, to, step)
   x2 <- x1/2
   y2 <- x1*sqrt(3)/2
   x3 <- (1-x1)*0.5+x1
   y3 <- sqrt(3)/2-x1*sqrt(3)/2
   lattice::panel.segments(x1, 0, x2, y2, col=col, lty=lty, lwd=lwd)
   lattice::panel.text(x1, 0, label=x1, pos=1)
   lattice::panel.segments(x1, 0, x3, y3, col=col, lty=lty, lwd=lwd)
   lattice::panel.text(x2, y2, label=rev(x1), pos=2)
   lattice::panel.segments(x2, y2, 1-x2, y2, col=col, lty=lty, lwd=lwd)
   lattice::panel.text(x3, y3, label=rev(x1), pos=4)
 }
 
 
 q=lattice::xyplot(trian.cond$high~trian.cond$base,par.settings=list(axis.line=list(col=NA),
                   axis.text=list(col=NA)),xlab=NULL, ylab=NULL, pch=19,cex=0.1,col="gray47",
                   xlim=c(-0.1,1.1), ylim=c(-0.1,0.96),aspect = sqrt(3)/2)
 dev.new(title="mixture domain")
 print(q)
 
 lattice::trellis.focus("panel", 1, 1, highlight=FALSE)
 lattice::panel.segments(c(0,0,0.5), c(0,0,sqrt(3)/2), c(1,1/2,1), c(0,sqrt(3)/2,0))
 grade.trellis()
 lattice::panel.text(.55,0.92,label="X2",pos=2)
 lattice::panel.text(0,-0.05,label="X1",pos=2)
 lattice::panel.text(1,-0.05,label="X3",pos=4)
 lattice::trellis.unfocus()
 
 rm(q,grade.trellis,cond,delta,i,j,new,new.cond,trian,trian.cond,trian.new.cond)
 
print('The list of candidate points is saved in "cp"',quote=FALSE)}