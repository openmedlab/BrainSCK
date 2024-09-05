# BrainSCK
Implementation code of MICCAI2024 paper 
BrainSCK: Brain Structure and Cognition Alignment via Knowledge Injection and Reactivation for Diagnosing Brain Disorder
****
## Requirements
* The following setup has been tested on Python 3.9, Ubuntu 20.04.
* Major dependences: pytorch 1.13.1 salesforce-lavis 1.0.2 transformers 4.28.1
****
## Usage
* Download and unzip the pretrained model weights from HCP datasets.
  link: https://pan.baidu.com/s/1enSzKtHRjUQuk7_31VpN7g  password: gtnh
* Prepare your data in proper way, run the script "run_train.sh" to tune the model for diagnosing brain disorders, then run the script "run_test.sh" for testing.  
