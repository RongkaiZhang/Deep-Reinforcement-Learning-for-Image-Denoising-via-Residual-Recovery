# Deep-Reinforcement-Learning-for-Image-Denoising-via-Residual-Recovery (R3L).
1. This is a simple implementation of DRL (PPO is used) for image denoising via residual recovery.
2. Detailed illustration can be found in our paper [R3L: Connecting Deep Reinforcement Learning To Recurrent Neural Networks For Image Denoising Via Residual Recovery](https://arxiv.org/abs/2107.05318) (accepted by ICIP 2021).
3. Although this project is for a specific task, this framework is designed ASAP (as simple as possible) to be applied for different tasks trained in "Batch Environment" (Batch * Channel * Height * Width) by slightly modifing the corresponding network and envrionment.
# Introduction to this implementation:
1. Current implementations of PPO usually focus on environments with states in shape of (Height * Width) raising a gap for implementations in CV where (Channel * Height * Width) is needed.
2. This implementation aims for a easy-to-modify PPO framework for CV tasks.
3. The PPO used here is [PPO-clip](https://spinningup.openai.com/en/latest/algorithms/ppo.html).
# How to apply to other tasks:
1. Customize the environment by setting task specific reset(), step() in environment.py.
2. Customize the data file paths in PPO_batch.py.
3. Customize data argumentation in Load_batch.py.
