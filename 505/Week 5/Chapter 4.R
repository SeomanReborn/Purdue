# Sampling distributions and bias

set.seed(1)

samples=matrix(rpois(10000*5, lambda=4), ncol=5)

head(samples)

means=rowMeans(samples)
head(means)
mean(means)

vars=apply(samples, 1, var)
head(vars)
mean(vars)

sds=apply(samples, 1, sd)
head(sds)
mean(sds)

mean(sds)/0.94

R=function(x) diff(range(x))
ranges=apply(samples, 1, R)
head(ranges)
mean(ranges)/2.326


# Chi-Square distribution

pchisq(2, df=5)


# t distribution

pt(2, df=8)


# F distribution

pf(2, df1=2, df2=10)


# Confidence Intervals

# One sample t-based CI for mu
# Table 4.1 / Example 4.4
Viscosity=c(3193,3124,3153,3145,3093,
            3466,3355,2979,3182,3227,
            3256,3332,3204,3282,3170)

df=length(Viscosity)-1
df

qt(.975, df)
pt(2.144787, df)-pt(-2.144787, df)

mean(Viscosity)+c(-1,1)*qt(.975, df)*
       sd(Viscosity)/sqrt(length(Viscosity))

t.test(Viscosity, conf.level=0.95)$conf.int

t.test(Viscosity, conf.level=0.80)$conf.int

t.test(Viscosity, conf.level=0.90, 
       alternative='greater')$conf.int

# Two-sample t-based CI for mu_x - mu_y
t.test(x=, y=, conf.level=)$conf.int
t.test(x=, y=, var.equal=TRUE, conf.level=)$conf.int

# Paired t-based CI for mu_(x-y)
t.test(x=, y=, paired=TRUE, conf.level=)$conf.int

# One sample chi-squared-based CI for variance
var(Viscosity)*df/qchisq(c(.975,.025), df)

install.packages('EnvStats')
library(EnvStats)
varTest(Viscosity, conf.level=0.95)$conf.int

# Two sample F-based CI for ratio of variances
var.test(x=, y=, conf.level=)$conf.int

# One sample z-based CI for proportion
# Example 4.6
x=15; n=80
x/n + qnorm(c(.025,.975))*sqrt(x/n*(1-x/n)/n) 
prop.test(x, n)$conf.int

# Two sample z-based CI for difference in props
prop.test(x=c(x1, x2), n=c(n1, n2), conf.level=)$conf.int


# Hypothesis Tests

(mean(Viscosity)-3200)/
  (sd(Viscosity)/sqrt(length(Viscosity)))
2*(1-pt(0.35345, df))

t.test(Viscosity, mu=3200)

t.test(Viscosity, mu=3200, alternative='greater')

varTest(Viscosity, sigma.squared=)

x=15; n=80
prop.test(x, n, p=.1800)


# ANOVA

# Table 4.4
Concentration=sort(rep(c(5,10,15,20),6))
Strength=c(7,8,15,11,9,10,
           12,17,13,18,19,15,
           14,18,19,17,16,18,
           19,25,22,23,18,20)

boxplot(Strength~Concentration, col=terrain.colors(4))

Conc=factor(Concentration)
# Especially if your grouping variable is
#   numeric, make sure it is saved as a factor!

paper=lm(Strength~Conc)

anova(paper)

names(paper)
paper$residuals

Strength[1]
mean(Strength[Concentration==5])

qqnorm(paper$residuals)


# Regression

# https://allisonhorst.github.io/palmerpenguins/
install.packages('palmerpenguins')
library(palmerpenguins)

summary(penguins)

attach(na.omit(penguins))

plot(bill_depth_mm~bill_length_mm, pch=16)

reg1=lm(bill_depth_mm~bill_length_mm)

reg1

abline(reg1)

summary(reg1)

anova(reg1)

plot(reg1)


plot(bill_depth_mm~bill_length_mm,
     pch=as.numeric(species)+14, col=species)

reg2=lm(bill_depth_mm~bill_length_mm+species)
                # species is saved as a factor

summary(reg2)

abline(reg2$coef[1:2], lwd=2)
abline(a=reg2$coef[1]+reg2$coef[3],
       b=reg2$coef[2], col=2, lwd=2)
abline(a=reg2$coef[1]+reg2$coef[4],
       b=reg2$coef[2], col=3, lwd=2)


# Table 4.10 / Example 4.13
Applications=c(80,93,100,82,90,99,81,96,
               94,93,97,95,100,85,86,87)
Loans=c(8,9,10,12,11,8,8,10,
        12,11,13,11,8,12,9,12)
Cost=c(2256,2340,2426,2293,2330,2368,2250,2409,
       2364,2379,2440,2364,2404,2317,2309,2328)

reg3=lm(Cost~Applications+Loans)

summary(reg3)

#install.packages('rgl')
library(rgl)
plot3d(Cost~Applications+Loans, size=8, col=4)
planes3d(reg3$coef[2], reg3$coef[3], -1, 
         reg3$coef[1], col=2, alpha=0.5)