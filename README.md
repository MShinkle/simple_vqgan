# simple_vqgan
A simplified version of VQGAN-CLIP for art generation.

I've found existing VQGAN code to be feature-rich, but complicated for beginner use.  Also, they rarely have innate CPU support.  This is intended to be a lighter-weight but still capable variant.

Currently only tested with python 3.9 on linux (Ubuntu 20), but should work on Windows and MacOS and other version of Python 3.
## Anaconda Installation
It is recommended that you install via the command line using Anaconda.  E.g. first create a new anaconda environment:

`conda create -n simple_vqgan python=3.9`

and activate it:

`conda activate simple_vqgan`

Clone this repo, and then navigate into the cloned directory.  Then install with the setup.py.

`python setup.py install`

The package should now be installed!

### Possible installation issues

A common error on fresh linux installs is missing gcc--this can be installed via apt:

`sudo apt install gcc`

Then call `python setup.py install` again.

Also note that if you intend to use CUDA, you may need to install a version of pytorch specific to your CUDA version.

##### CUDA 10.2
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch

##### CUDA 11.3
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

CPU-only use *shouldn't* require special pytorch installation.

## Running
Using all the defaults, all you need to do is import the `generate` function and pass it a prompt string:

```
from simple_vqgan import generate

generate('Neural Dreams')
```

This will run and then save a generate image to [package directory]/outputs/Neural_Dreams.png

Further input args can be used to change image size, device (CPU vs CUDA), seeds, image transformations, etc.

Enjoy!
