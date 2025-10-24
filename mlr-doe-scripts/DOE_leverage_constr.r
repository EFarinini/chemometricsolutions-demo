# Authors   : R. Leardi, C. Melzi, G. Polotti
## Generates triangle
trian <- expand.grid(base=seq(0,1,l=100*2), high=seq(0,sin(pi/3),l=87*2))
trian <- subset(trian, (base*sin(pi/3)*2)>high)
trian <- subset(trian, ((1-base)*sin(pi/3)*2)>high)

## Generates triangle in R^3 X1+X2+X3=1
new=data.frame(X1=1-trian$base-trian$high/sqrt(3))
new$X2=trian$high*2/sqrt(3)
new$X3=trian$base-trian$high/sqrt(3)

## Generates data.frame triangle in 2 (base,high), 3 (X1,X2,X3) variables and column conditions (cond)
cond=new$X1>=vinc.inf[1] & new$X2>=vinc.inf[2] & new$X3>=vinc.inf[3] & new$X1<=vinc.sup[1] & new$X2<=vinc.sup[2] & new$X3<=vinc.sup[3]
trian.new.cond=cbind(trian,new,cond)

## Generates triangle in 2 (base,high), 3 (X1,X2,X3) variables satisfying contraints
trian.cond=trian.new.cond[trian.new.cond$cond==TRUE,1:2]
new.cond=trian.new.cond[trian.new.cond$cond==TRUE,3:5]

## Creates model matrix on X1+X2*X3=1 satisfying contraints (new.cond)

P=DOE$x
n=ncol(P)
M=DOE$m
m=ncol(M)

names=DOE$name
if (names[1]=="X0" & names[2]=="X1" & names[3]=="X2") names[1:3]=c("Comp.1","Comp.2","Comp.3")

names_coeff<-colnames(DOE$x)
form<-paste0(names_coeff,collapse = '+')
form<-paste0('~-1+',form)
form<-formula(form)
names(new)<-names_coeff[1:3]
names(new.cond)<-names_coeff[1:3]


## Design in coord (b,h)
b=1-P[,1]-P[,2]/2
h=(sqrt(3)/2)*P[,2]

X=model.matrix(form,data = new.cond)

## Leverage on trian.cond 
Q=X%*%solve(t(P)%*%P)%*%t(X)
Lev=data.frame(trian.cond,"L"=diag(Q))

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


## Values to define graph limits
b1=min(trian.cond$base)
b2=max(trian.cond$base)
h1=min(trian.cond$high)
h2=max(trian.cond$high)

## Generates isoleverage plot

p=lattice::contourplot(L~base*high, Lev, col = "red", cuts = 10,aspect = ((h2-h1)+.1)/((b2-b1)+.1),region = TRUE,col.regions = colorRampPalette(c("yellow","green","blue")),label=list(col="black",cex=0.9),
                       main="Leverage Plot",
                       par.settings=list(axis.line=list(col=NA), axis.text=list(col="black")),xlab=NULL, ylab=NULL,
                       scales=list(x=list(alternating=0),y=list(alternating=0)),
                       xlim=c(b1-0.05,b2+0.05), ylim=c(h1-0.05,h2+0.05))
dev.new(title="leverage mixture (zoom)")
print(p)

lattice::trellis.focus("panel", 1, 1, highlight=FALSE)
lattice::panel.segments(c(0,0,0.5), c(0,0,sqrt(3)/2), c(1,1/2,1), c(0,sqrt(3)/2,0))
lattice::panel.xyplot(x=b,y=h, col="red",pch=20,cex=1.75)
grade.trellis()
lattice::panel.text(.54+(nchar(names[2])-1)/100,0.92,label=names[2],pos=2)
lattice::panel.text((nchar(names[1])-1)/50,-0.05,label=names[1],pos=2)
lattice::panel.text(1-(nchar(names[3])-1)/50,-0.05,label=names[3],pos=4)
lattice::trellis.unfocus()

p=lattice::contourplot(L~base*high, Lev, col = "red", cuts = 10,aspect = sqrt(3)/2,region = TRUE,col.regions = colorRampPalette(c("yellow","green","blue")),label=list(col="black",cex=0.8),
                       main="Leverage Plot",
                       par.settings=list(axis.line=list(col=NA), axis.text=list(col="black")),xlab=NULL, ylab=NULL,
                       scales=list(x=list(alternating=0),y=list(alternating=0)),
                       xlim=c(-0.1,1.1), ylim=c(-0.1,0.96))

dev.new(title="leverage mixture")  
print(p)
  
lattice::trellis.focus("panel", 1, 1, highlight=FALSE)
lattice::panel.segments(c(0,0,0.5), c(0,0,sqrt(3)/2), c(1,1/2,1), c(0,sqrt(3)/2,0))
lattice::panel.xyplot(x=b,y=h, col="red",pch=20,cex=1.75)
grade.trellis()
lattice::panel.text(.53+(nchar(names[2])-1)/100,0.92,label=names[2],pos=2)
lattice::panel.text((nchar(names[1])-1)/50,-0.05,label=names[1],pos=2)
lattice::panel.text(1.01-(nchar(names[3])-1)/50,-0.05,label=names[3],pos=4)
lattice::trellis.unfocus()
    
rm(P,n,M,m,form,trian,new,trian.cond,new.cond,Q,Lev,grade.trellis,b,h,b1,b2,h1,h2,p,trian.new.cond,X,cond)
if (exists("txt")) rm(txt)
if (exists("i")) rm(i)
if (exists("names")) rm(names)