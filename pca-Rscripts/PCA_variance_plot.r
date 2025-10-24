# Purpose   : plot the variance of each component.
# Input     : V, vector of variance
# Output    : plot of variance (blue) as a function of vector position.
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 3.0
# Licence   : GPL2.1
#
if (exists("PCA",envir=.GlobalEnv)){
dev.new(title="scree plot")
op<-par(pty='s')
V<-PCA[[1]]@R2*100
xlab<-'Component Number'
if(PCA$type=='varimax')xlab<-'Factor Number'
plot(V,xlab=xlab,ylab="% Explained Variance",
main='Scree plot',ylim=c(0,max(V)*1.1),type='n')
for(i in 1:length(V)){
if(V[i]!=0){points(i,V[i],col='blue') }}
lines(1:i,V[1:i])
grid();par(op);rm(V,i,op)}else {
tk_messageBox(type=c("ok"),message='Run Matrix Evaluation First!',caption="Input Error")}
