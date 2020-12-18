# DifferentWorldModels

## Abstract

Car-racing v0 is an exciting and difficult environment that was recently solved using a model proposed in World Models[1]. In this demo, we explore alternate versions of the model that change the type of information utilized in the agent’s internal model of the world, such as using a beta-VAE and a randomly initialized MDN-RNN and compare the results to the standard implementation as proposed in the original world models paper [2, 3]. We show that the beta-VAE can achieve equal, if not better results, with a peak reward of 593.

VIDEO GOES HERE (probably): Record a 2-3 minute long video presenting your work. One option - take all your figures/example images/charts that you made for your website and put them in a slide deck, then record the video over zoom or some other recording platform (screen record using Quicktime on Mac OS works well). The video doesn't have to be particularly well produced or anything.

## Introduction

The goal of this project was to investigate the effectiveness of alternative implementations of the WorldModels model, as well as the replicability of their results. The original model is extremely exciting, as it’s based on the idea that humans have an internal model of the world around them and make decisions based on visual representations and memory of these representations. In this project, we explore the effects of changing the type of information stored in that internal model by changing the Beta value in the VAE (variational autoencoder).

We also thought it was interesting that a group randomly initialized an MDN-RNN and found it to perform just as well in learning an external environment [2]. Of course, this means that the agent will not be able to create hallucinations that visually make sense. We explored these results by also using a randomly initialized MDN-RNN.


## Related Work

1. https://worldmodels.github.io/
  - This is the original model. it uses a VAE, an MDN-RNN, and a linear controller to choose actions. it achieves a state of the art performance with 950 reward, solving the environment. 
2. https://ctallec.github.io/world-models/
  - This group reimplemented the original model, as well as an alternative version with a randomly initialized MDN-RNN. Their reimplementation of the model achieved 860 reward. Their reimplementation with the randomly initialized weights in the MDN-RNN achieved a score of 870.
3. https://openreview.net/references/pdf?id=Sy2fzU9gl
  - This group trains an alternative version of the standard VAE (https://arxiv.org/abs/1312.6114). When training a VAE, the loss function is made up of two terms: reconstruction loss and KL divergence. This group manipulated the loss function by multiplying the KL divergence by a value beta > 1. In doing so, they are able to change the properties of the latent vector z. The greater the beta value utilized, the more “disentangled” the z vector becomes from the original input image. After performing hyperparameter optimization with the beta value, they are able to train a VAE that achieves better reconstructions than the original (beta = 1).

## Approach

How did you decide to solve the problem? What network architecture did you use? What data? Lots of details here about all the things you did. This section describes almost your whole project.

Figures are good here. Maybe you present your network architecture or show some example data points?

## Results

How did you evaluate your approach? How well did you do? What are you comparing to? Maybe you want ablation studies or comparisons of different methods.

You may want some qualitative results and quantitative results. Example images/text/whatever are good. Charts are also good. Maybe loss curves or AUC charts. Whatever makes sense for your evaluation.

## Discussion

You can talk about your results and the stuff you've learned here if you want. Or discuss other things. Really whatever you want, it's your project.

