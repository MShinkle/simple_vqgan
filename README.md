# simple_vqgan
A simplified version of VQGAN-CLIP for art generation.

I've found existing VQGAN code to be feature-rich, but complicated for beginner use.  Also, they rarely have innate CPU support.  This is intended to be a lighter-weight but still capable variant.

Developed primarily for linux (Ubuntu 20) with python 3.9, but has been tested some on Windows and MacOS (still w/ python 3.9.)
## Anaconda Installation
It is recommended that you install via the command line using Anaconda.  E.g. first create a new anaconda environment:

`conda create -n simple_vqgan python=3.9`

and activate it:

`conda activate simple_vqgan`

Clone this repo, and then navigate into the cloned directory.  Then install with the setup.py.

`python setup.py install`

The package should now be installed!

### If you want to use CUDA (gpu)

Also note that if you intend to use a GPU, you may need to install a version of pytorch specific to your CUDA version.

##### CUDA 10.2
`conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch`

##### CUDA 11.3
`conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch`

CPU-only use *shouldn't* require special pytorch installation.

## Running
Using all the defaults, all you need to do is import the `generate` function and pass it a prompt string:

```
from simple_vqgan import generate

generate('Neural Dreams')
```

This will run and then save a generated image.  By default, it will be named after the prompt (e.g. Neural_Dreams.png) to a directory named 'renders' in the current directory.  Both the file name and the save location can be specified with input arguments.

You can change other aspects of the generation with other inputs.  Of note, you can change the size of the generated image by specifying `height` and `width` values. Further input args can be used to change image size, device (CPU vs CUDA), iterations, seeds, image transformations, etc.

For example, to generate a 500x500-pixel image for 300 iterations using the GPU, you could run:

`generate('Neural Dreams', height=500, width=500, iterations=300, device='CUDA')`


## Examples

You can pass any text string to the model, and it will do its best to create a matching image.  For example:

*"ocean cityscape"*

![Ocean_cityscape__1](https://user-images.githubusercontent.com/49564869/183120381-6ebc370d-7672-4a10-8b8b-7cdc22a35bf2.jpg)



You can get interesting renders by specifying modifiers, such as styles...:

*"landscape pastels"*

![Landscape_pastels_0](https://user-images.githubusercontent.com/49564869/183120917-5ccc2fcb-bf7f-4f0c-9fcd-23158c601247.jpg)


*"fractal 4k HD"*

![fractal_4k_HD_0](https://user-images.githubusercontent.com/49564869/183120324-13fff210-4f63-4e6d-8c03-91ef7aed7d11.jpg)



Artists...:

*"Andrew Jones City intricate pencil drawing"*

![andrew_jones_city_pencil_drawing](https://user-images.githubusercontent.com/49564869/183120699-5e080c52-32aa-4244-b22c-55a9e507a2ab.png)


*"Beksi??ski depths organic brutalism"*

![Beksi??ski_depths_organic_brutalism_0](https://user-images.githubusercontent.com/49564869/183120449-57e5dc12-d6cb-4850-a3c4-b9d14f1b6a27.jpg)


*"Andrew Jones City in the style of Beksi??ski"*

![andrew_jones_city_Beksi??ski](https://user-images.githubusercontent.com/49564869/183120705-81a59240-a931-4d47-9eda-e395a5a779cb.png)



Engines, websites, etc...:

*"God is dead Unreal Engine"*

![God_is_dead_unreal_engine_0](https://user-images.githubusercontent.com/49564869/183120486-908592ff-6d64-4791-a992-f69607098305.jpg)


*"Flavortown trending on ArtStation"*

![Flavortown_trending_on_artstation_0](https://user-images.githubusercontent.com/49564869/183120533-3d577845-5754-478b-ab22-9ccc4015111c.jpg)



These are just the tip of the iceberg; have fun!
