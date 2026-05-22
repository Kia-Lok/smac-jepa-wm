## Losses
**Total loss**: pred loss + sigreg weight * sigreg loss + decoder weight * decoder loss (Training Objective is minimise total loss)
**Pred loss**: How close the predicted latent representation is to the actual latent representation
**Sigreg loss**: Representation regularisation loss (Prevent model collapse by trying to make the created embeddings fall in a isotrophic gaussian distribution)
**Decoder loss**: Ability to reconstruct latent into useful entity-level information

Obseration:
- Losses usually converge by the first epoch
- Sigreg loss still showing steady but slow decrement over the rest of the epochs while pred and decoder losses start having weird trends

## Evaluation Metrics
Available data is split into 80/20 for train/test and split is done randomly. Evaluation is done by comparing what a model predicts when fed an obs and action and compared against actual

**Next state embedding mse**: How close predicted embedding is to actual embedding
**Decoded_mae** (Mean Absolute Error) 
**Decoded_mse** (Mean Square Error)
**Decoded r2**: Measures how much variance in the real target is explained by prediction. (Decoded predictions explain 90.1% of the variation in true next-state entity features)
**Presence_acc** (Due to switch in entity level encoding): Whether model correctly predicts whether an entity slot is present or missing
**Tol_acc_{num}**: How manty percent of predictions are within **num** of the actual.
