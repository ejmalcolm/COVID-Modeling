import lhsmdu

lhsmdu.setRandomSeed(None)
samples = lhsmdu.sample(1,100)
# initially, it is generated between 0 and 1
# we need to change it to a distribution between -100 and 100

adj = []
samples = samples.tolist()
print(samples[0])