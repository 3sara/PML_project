# pml project

## baum welch for iohmm

what i used

article: https://proceedings.neurips.cc/paper/1994/file/8065d07da4a77621450aa84fee5656d9-Paper.pdf

to find out how to adapt the forward and backward algorithm in a iohmm and how to write the likelihood

site: https://adeveloperdiary.com/data-science/machine-learning/derivation-and-implementation-of-baum-welch-algorithm-for-hidden-markov-model/

to find out how to use the alpha and beta to compute gamma and xi for computing the likelihood


POSSIBLE EXTENSIONS:

online learning: don't re-compute all the coeffcients when new data come in

improve training, use matrices operations instead of for
don't use torch for optimisation, try to use the derivatives written in the article

predicition for more than one obs

