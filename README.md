# rl-hypothesis-1
Can sparse state spaces be gainfully populated with GANs when represented as embeddings?

- Essential ideas
  - RL is stuck in simulated worlds until it can avoid entire-game simulation. 
  - GANs can learn to simulate, but suck at exact recreation. I don't hold a grudge, my brain can't do it either.
  - So, we'll fit a q-learning network and chop it in-half to produce an embedding.
  - Fit a GAN to sample from the embedding.
  - Pad the (embedding-level) training dataset with GAN samples then refit the q-learning network (at cut and above). 
    - When samples are few, the statespace should be very sparsely sampled until padded with the GAN. 
    - The q-embedding should be easier to simulate, having only retained game-strategic info and no visual processing info. 
  - When finished, the final q-net is the original but updated post-cut. 

- Hypothesis
  - When game samples are constrained, the GAN will provide lift in total game wins. 

## Resources 

Fast and easy experimentation is prioritized. Content will be copied from public repos to avoid manual programming. 
- base q-net software [here](https://github.com/rlcode/reinforcement-learning/blob/master/3-atari/1-breakout/breakout_dqn.py) 
- execution environment [here](https://github.com/jaimeps/docker-rl-gym#docker-hub) 
  - NVidia TOS denies distribution of `cudnn` lib, so environment is CPU-only. This isn't a big setback since game simulation eats most of the time and is largely CPU-driven.
- gan [here](https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py) 
  - yeah, that's right--standard GANs. I'm that confident.  

I've tested the q-net and gan software. It's good stuff. The q-net really needs some heavy parallelization, ideally a parameter server. However, I don't have the dev time and will eat the cycles instead. Fortunately, it seems to be using available CPU cores (up to 14) and does seem to need a lot of RAM--so, I'll use a beefy node.  

## Build 

- Build environment: GCP console. 
- Configure with `config.sh` 
  - Service account must have read/write to cloud storage via `gsutil`  
- Execute build with `bash build.sh` 

## Execution 

## Results 

