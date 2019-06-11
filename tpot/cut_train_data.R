train.dat <- read.csv('BrainAGE_train.csv')
str(train.dat)
head(train.dat[1:5])
my.cov <- function(x){sd(x)/mean(x)}
cov.col <- apply(X = train.dat, MARGIN = 2, FUN = my.cov)
hist(cov.col)
boxplot(cov.col)
high.cov <- cov.col > quantile(cov.col, 0.5)
sum(high.cov)
new.train.dat <- train.dat[, high.cov]
write.csv(new.train.dat, "small_BrainAGE_train.csv")
