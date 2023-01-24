## Monotonic Multihead Attention
An off-the-shelf working version of Fairseq's monotonic multihead attention for SiMT

Depends on:
* Python 3.8
* PyTorch 1.5.0 cu10.2 `pip install torch==1.5.0 torchvision==0.6.0`


### To get this running
Clone this repo first
* `git clone https://github.com/protonish/monotonic_multihead_attention.git`
* `cd monotonic_multihead_attention`

Install fairseq inside this repo
* `git clone -b monotonic_multihead_attention https://github.com/facebookresearch/fairseq.git`
* `cd fairseq`
* `pip3 install -e .`

Then, `cd run_scripts` from this directory.

Set the path to this repo in the`ROOT` variable in each of the bash scripts.

Finally, run the following scripts:
* `bash prepare-iwslt14.sh`
* `bash train.sh`

