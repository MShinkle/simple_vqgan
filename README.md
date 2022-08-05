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

![fractal_4k_HD_0](https://user-images.githubusercontent.com/49564869/183120324-13fff210-4f63-4e6d-8c03-91ef7aed7d11.jpg)

![Ocean_cityscape__1](https://user-images.githubusercontent.com/49564869/183120381-6ebc370d-7672-4a10-8b8b-7cdc22a35bf2.jpg)
![Beksiński_depths_organic_brutalism_0](https://user-images.githubusercontent.com/49564869/183120449-57e5dc12-d6cb-4850-a3c4-b9d14f1b6a27.jpg)
![God_is_dead_unreal_engine_0](https://user-images.githubusercontent.com/49564869/183120486-908592ff-6d64-4791-a992-f69607098305.jpg)
![Flavortown_trending_on_artstation_0](https://user-images.githubusercontent.com/49564869/183120533-3d577845-5754-478b-ab22-9ccc4015111c.jpg)
![andrew_jones_city_pencil_drawing](https://user-images.githubusercontent.com/49564869/183120699-5e080c52-32aa-4244-b22c-55a9e507a2ab.png)
![andrew_jones_city_pencil_drawing_buzsaki](https://user-images.githubusercontent.com/49564869/183120705-81a59240-a931-4d47-9eda-e395a5a779cb.png)
