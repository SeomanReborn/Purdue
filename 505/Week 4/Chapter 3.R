# Data from Table 3.1
Claim=1:40
Days=c(48,41,35,36,37,26,36,46,35,47,
       35,34,36,42,43,36,56,32,46,30,
       37,43,17,26,28,27,45,33,22,27,
       16,22,33,30,24,23,22,30,31,17)

# Numerical summaries

n=length(Days)
n

n/2; n/2 + 1

sort(Days)
sort(Days)[20:21]
mean(sort(Days)[20:21])
median(Days)

?quantile

quantile(Days,.10)

quarts=quantile(Days,c(.25,.50,.75))
quarts

quarts[3]-quarts[1]
IQR(Days)

quarts[1]-1.5*IQR(Days)
quarts[3]+1.5*IQR(Days)

sum(Days)/length(Days)
mean(Days)

sum((Days-mean(Days))^2)/(length(Days)-1)
(sum(Days^2)-sum(Days)^2/length(Days))/(length(Days)-1)
var(Days)

sqrt(var(Days))
sd(Days)

ex1=1:5
ex2=ex1+10
ex3=ex1*10
ex4=ex1; ex4[5]=50
rbind(ex1,ex2,ex3,ex4)

mean(ex1); mean(ex2); mean(ex3); mean(ex4)

median(ex1); median(ex2); median(ex3); median(ex4)

IQR(ex1); IQR(ex2); IQR(ex3); IQR(ex4)

sd(ex1); sd(ex2); sd(ex3); sd(ex4)

# Visual summaries

stem(Days)

?stem

stem(Days, scale=.5)

plot(y=Days, x=Claim)          # Figure 3.2
plot(y=Days, x=Claim, type='l')
plot(Days~Claim, type='b')


# Data from Table 3.2
Thickness=c(438,413,444,468,445,472,474,454,455,449,
            450,450,450,459,466,470,457,441,450,445,
            487,430,446,450,456,433,455,459,423,455,
            451,437,444,453,434,454,448,435,432,441,
            452,465,466,473,471,464,478,446,459,464,
            441,444,458,454,437,443,465,435,444,457,
            444,471,471,458,459,449,462,460,445,437,
            461,453,452,438,445,435,454,428,454,434,
            432,431,455,447,454,435,425,449,449,452,
            471,458,445,463,423,451,440,442,441,439)

hist(Thickness)

?hist

hist(Thickness, breaks=15)


# Data from Table 3.4
Diameter=c(120.5,120.9,120.3,121.3,
           120.4,120.2,120.1,120.5,
           120.7,121.1,120.9,120.8)


boxplot(Diameter, horizontal=TRUE)


?mtcars

summary(mtcars$mpg)
stem(mtcars$mpg)
hist(mtcars$mpg)
boxplot(mtcars$mpg)
boxplot(mpg~cyl, data=mtcars, col=2:4)


# Hypergeometric distribution

N=52  # cards in deck
D=13  # Hearts in deck
n=5   # cards drawn

x=0:min(n,D)

probs=choose(D,x)*choose(N-D,n-x)/choose(N,n)
cbind(x,probs)

?dhyper
dhyper(x, m=D, n=N-D, k=n)

# P(at least 3 Hearts) =
sum(probs[4:6])

D=4   # Kings in deck

x=0:min(n,D)

# P(fewer than 2 Kings) =
sum(dhyper(x, m=D, n=N-D, k=n)[1:2])


# Binomial distribution

n=150
p=0.90

x=0:n

choose(n,x)*p^x*(1-p)^(n-x)
probs=dbinom(x, size=n, prob=p)
probs

plot(probs~x, type='h')

#P(between 3 and 6, inclusive, successes) =
sum(probs[4:7])


# Poisson distribution

?dpois

lambda=4

x=0:alot

lambda^x*exp(-lambda)/factorial(x)

probs=dpois(x, lambda)
probs

plot(probs~x, type='h')

# P(2 or fewer defects) =
sum(probs[1:3])

# P(more than 10 defects) =
1 - sum(probs[1:11])


# Negative Binomial & Geometric distributions

r=1
p=1/6

?dnbinom   # ?dgeom
# R defines the variable x as the number 
#  of FAILURES until the r^th success,
#  not the number of trials.
#  (R's x) + r = (book's x)

x=0:alot

dnbinom(x, size=r, prob=p)
dgeom(x, prob=p)

r=3
dnbinom(x, size=r, prob=p)


# Normal distribution

# Example 3.7

mu=40
sigma=2

?pnorm
# p____ gives cumulative probabilities F(x)
# d____ gives the height of the f(x) curve

# P(x <= 35) =
pnorm(35, mean=mu, sd=sigma)    # needs sd, not var

# P(x > 41) =
1 - pnorm(41, mu, sigma)

# Example 3.8

mu=0.2508        #0.2500
sigma=0.0005

spec1=0.2500-0.0015
spec2=0.2500+0.0015

pnorm(spec2, mu, sigma) - pnorm(spec1, mu, sigma)

# Example 3.9

mu=10
sigma=sqrt(9)

qnorm(.95, mu, sigma)


# Getting the values in Figure 1.12

sig=1:6

# a (Normal centered at target)
(pnorm(sig)-pnorm(-sig))*100   # Percentage Inside Specs
(1-(pnorm(sig)-pnorm(-sig)))*1000000   # ppm Defective

# b (Normal with mean shifted by 1.5sigma from target
(pnorm(sig, -1.5)-pnorm(-sig, -1.5))*100   # Percentage Inside Specs
(1-(pnorm(sig, -1.5)-pnorm(-sig, -1.5)))*1000000   # ppm Defective


# Central limit theorem with dice

par(mfrow=c(2,2))     # plot in 2x2 grid

# A single die roll

n=10000
plot(table(sample(1:6,n,replace=T))/n, type='h', lwd=3, xlim=c(1,6),
     main='1 Roll', xlab='', ylab='Prob')

# Average of two die rolls
twos=(sample(1:6,n,replace=T)+sample(1:6,n,replace=T))/2
plot(table(twos)/length(twos), type='h', lwd=3, xlim=c(1,6),
     main='Average of 2 Rolls', xlab='', ylab='Prob')

# Average of four die rolls
fours=colMeans(matrix(sample(1:6,n*4,replace=T),nrow=4))
plot(table(fours)/length(fours), type='h', lwd=3, xlim=c(1,6),
     main='Average of 4 Rolls', xlab='', ylab='Prob')

# Average of twelve die rolls
twelves=colMeans(matrix(sample(1:6,n*12,replace=T),nrow=12))
plot(table(twelves)/length(twelves), type='h', lwd=3, xlim=c(1,6),
     main='Average of 12 Rolls', xlab='', ylab='Prob')


# Lognormal, Example 3.10

theta=6
omega=1.2

# P(x>500) =
1-plnorm(500, meanlog=theta, sdlog=omega)
1-pnorm(log(500), theta, omega)     
    # in R, log() is ln by default

qlnorm(.01, theta, omega)    
    # the book rounds qnorm(.01) to -2.33


# Exponential distribution

lambda=10^-4

1/lambda  # mean

# P(x < 10000) =
1 - exp(-lambda*10000)
pexp(10000, rate=lambda)

# P(x > 20000) =
1- (1 - exp(-lambda*20000))
1 - pexp(20000, rate=lambda)


# Gamma distribution, Example 3.11

r=2

# P(x > 20000) =
1 - pgamma(20000, shape=r, rate=lambda)


# Weibull distribution, Example 3.12

beta=1/2
theta=5000

# P(x > 20000)
1-(1-exp(-(20000/theta)^beta))
1-pweibull(20000, shape=beta, scale=theta)


# Probability plots, Example 3.13
octane=c(88.9, 87.0, 90.0, 88.2, 87.2, 87.4, 87.8, 89.7, 86.0, 89.6)

qqnorm(octane, datax=TRUE)
qqline(octane, datax=TRUE)

sort(qqnorm(octane)$x)   # R's quantiles

n=length(octane)
j=1:n
(j-0.5)/n
qnorm((j-0.5)/n)         # book's quantiles

qqplot(octane, qnorm((j-0.5)/n))


# Binomial Approximation to Hypergeometric

N=200
D=5
n=10

n/N   # <= 0.1

round(dhyper(0:min(D,n), m=D, n=N-D, k=n), 4)

round(dbinom(0:n, size=n, prob=D/N), 4)


# Poisson Approximation to Binomial

n=100
p=0.025

round(dbinom(0:n, size=n, prob=p), 4)

round(dpois(0:n, lambda=n*p), 4)


# Normal Approximation to Binomial

n=40
p=0.5

mu=n*p
sigma=sqrt(n*p*(1-p))

pnorm(18.5, mu, sigma)-pnorm(17.5, mu, sigma)
dbinom(18, n, p)

1-pnorm(24.5, mu, sigma)
sum(dbinom(25:n, n, p))

plot(x=0:n, y=dbinom(0:n, n, p), type='h', lwd=2)
curve(dnorm(x, mu, sigma), col='purple', lwd=2, add=TRUE)