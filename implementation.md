The goal of this task is to modify the JEPA World Model architecture such that it can be generalise on SMAClite. The repo that is slightly modified and meant to store this project is at https://github.com/Kia-Lok/smac-jepa. This repo is private so if you can't find way to view it, it is fine and I want to just describe the task that will allow you to rebuild it.

Take inspiration from LeWorldModel at https://github.com/lucas-maes/le-wm. I would want my JEPA WM to be structured like LeWM. However, LeWM is hardcoded to handle pixels, which I don't need. The ViT as encoder and autoregressive model as predictor as way too overkill for SMAClite. SMAClite is a simulator that stores the global state as a vector and actions taken as vectors as well. The code details are here: https://github.com/uoe-agents/smaclite . Since vectors are used instead of images, an self-attention head from pytorch can simply be trained on the output from SMAClite. Predictor consequently shouldn't be big as well. Ideally, it doesn't need GPU to train (Especially at the exploring stage). Also note that LeWM was done on continuous tasks so frames are used. SMACLite is discrete joint action so frames can be set to 1 to account for discrete.

So I want you to do the following:
1) Look at SMACLite and generate a script that can take a env/scenario and generate data in the way that can be inserted to train and evaluate our SMAC-JEPA world model (It should be compatible with simulator env not necessarily WM env so I should throw this script into SMACLite pulled repo and run to collect)
2) Taking inspiration from LeWM, construct the JEPA world model. Dataloder, Sigreg should be kept while code blocks under module. jepa, utils and train should keep the relevant ones for our task and add in new ones to modify the world model
3) I want to do this project incrementally. At the first stage, I just want to make sure the code works. So the goal is for me to collect data from the common scenarios and see if the training loss can converge (Yes please build in a training loss tracker). 
4) After training, to evaluate is to find a way to check if given a state, can the model predict the next action (Or show best course of action in the next stage)

Reading links:
https://arxiv.org/pdf/2603.19312
https://arxiv.org/pdf/2306.02572
https://arxiv.org/pdf/2305.05566
Do also read up on how JEPA works before embarking on the code

