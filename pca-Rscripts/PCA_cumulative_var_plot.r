# Purpose   : plot the cumulative variance vector from PCA matrix evaluation.
#             Single  cumulative variance points are displayed
# Input     : V of variance
# Output    : plot of cumulated variance as a function of number of components. 
# Authors   : R. Leardi, C. Melzi, G. Polotti
# Programmer: GMP
# Version   : 3.1
# Licence   : GPL2.1
#
if (exists("PCA",envir=.GlobalEnv)){
dev.new(title="cumulative variance")
op<-par(pty='s')
V<-PCA[[1]]@R2cum*100
xlab<-'Component Number'
if(PCA$type=='varimax')xlab<-'Factor Number'
plot(V,xlab=xlab,ylab="% Explained Variance",
main='% Explained Variance',ylim=c(0,100),type='n')
for(i in 1:length(V)){
if(V[i]!=0){points(i,V[i],col='red') }}
lines(1:i,V[1:i])
grid();par(op);rm(V)}else {
tk_messageBox(type=c("ok"),message='Run Matrix Evaluation First!',caption="Input Error")}
