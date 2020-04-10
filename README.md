# Code-for-Drivable-Area
This is a graduation project, thus, maybe not good enough.  
For now, net_beta is recommended in models.  
├─checkpoints  
├─data  
│  ├─TuSimple  
│  │  ├─test  
│  │  └─train  
├─models  
├─temp  
│  ├─img  
│  └─video  
└─utils  
  
# How to use?
Train:  
python main.py train  
Test:  
python main.py test --pretrained_net_root='checkpoints/{the .pth file of your trained model}'  
Inference:  
(for image)  
python main.py inference --pretrained_net_root="checkpoints/{the .pth file of your trained model}" --infer_mode=0 --infer="{the image root}"  
(for video)  
python main.py inference --pretrained_net_root="checkpoints/{the .pth file of your trained model}' --infer_mode=1 --infer="{the video root}"
