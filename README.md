
projekt.py output:

TRAFFIC SIGN CLASSIFICATION
Data ready. Train: 31367, Val: 7842

Training
Epoch 1/10 | Train Loss: 1.9417 | Val Loss: 0.6067 | Val Acc: 82.22%
Epoch 2/10 | Train Loss: 0.6730 | Val Loss: 0.2424 | Val Acc: 93.94%
Epoch 3/10 | Train Loss: 0.4270 | Val Loss: 0.1504 | Val Acc: 96.90%
Epoch 4/10 | Train Loss: 0.3159 | Val Loss: 0.1076 | Val Acc: 97.55%
Epoch 5/10 | Train Loss: 0.2491 | Val Loss: 0.0788 | Val Acc: 98.43%
Epoch 6/10 | Train Loss: 0.2086 | Val Loss: 0.0698 | Val Acc: 98.33%
Epoch 7/10 | Train Loss: 0.1779 | Val Loss: 0.0637 | Val Acc: 98.36%
Epoch 8/10 | Train Loss: 0.1573 | Val Loss: 0.0514 | Val Acc: 98.90%
Epoch 9/10 | Train Loss: 0.1425 | Val Loss: 0.0596 | Val Acc: 98.53%
Epoch 10/10 | Train Loss: 0.1317 | Val Loss: 0.0467 | Val Acc: 98.89%

Confusion matrix analysis
Overall accuracy: 98.89%

TOP 10 most confused pairs:
 True  Pred  Count            True_Name           Pred_Name
   12     1      9        Priority road      Speed limit 30
    1    12      6       Speed limit 30       Priority road
   12    38      5        Priority road          Keep right
   38    23      5           Keep right       Slippery road
   23    38      4        Slippery road          Keep right
   19    10      3 Dangerous curve left No passing (trucks)
   41    40      3       End no passing          Roundabout
   38    41      2           Keep right      End no passing
   36    26      2 Go straight or right     Traffic signals
   38    40      2           Keep right          Roundabout

Robustness test (Real-World Conditions)
Baseline                  -> Acc: 98.89% (Drop: +0.00%)
Light Blur                -> Acc: 98.69% (Drop: -0.20%)
Heavy Blur                -> Acc: 97.50% (Drop: -1.39%)
Gaussian Noise (σ=0.1)    -> Acc: 90.27% (Drop: -8.62%)
Gaussian Noise (σ=0.2)    -> Acc: 76.79% (Drop: -22.10%)
Brightness -30%           -> Acc: 98.55% (Drop: -0.34%)
Brightness -50%           -> Acc: 97.48% (Drop: -1.42%)
Low Contrast              -> Acc: 98.74% (Drop: -0.15%)

Grad-CAM vizualization

Adversarial robustness (FGSM Attack)
ε = 0.00 -> Accuracy: 98.89%
ε = 0.01 -> Accuracy: 86.23%
ε = 0.03 -> Accuracy: 58.15%
ε = 0.05 -> Accuracy: 41.48%
ε = 0.10 -> Accuracy: 23.99%
ε = 0.15 -> Accuracy: 18.21%
ε = 0.20 -> Accuracy: 15.34%
