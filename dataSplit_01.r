#It makes sense to split the data from 50/50 to 80/20 (training/test)

set.seed(123) #until the seed is changed, this seed will be used for any pseudo-randomization command used hereafter

#Create the split to separate data into training and test
ksplit=sample.split(ev_adoption_01$Index, SplitRatio = 0.80)

training_set = subset(ev_adoption_01, ksplit==TRUE)
test_set = subset(ev_adoption_01, ksplit==FALSE)
