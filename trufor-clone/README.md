# TruFor

[![TruFor](https://img.shields.io/badge/TruFor%20webpage-222222.svg?style=for-the-badge&logo=github)](https://grip-unina.github.io/TruFor)
[![arXiv](https://img.shields.io/badge/-arXiv-B31B1B.svg?style=for-the-badge)](https://doi.org/10.48550/arXiv.2212.10957)
[![GRIP](https://img.shields.io/badge/-GRIP-0888ef.svg?style=for-the-badge)](https://www.grip.unina.it)

Official PyTorch implementation of the paper "TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization"

<p align="center">
 <img src="./docs/teaser.png" alt="teaser" width="70%" />
</p>

## News

*   2023-06-28: Test code is now available
*   2023-02-27: Paper has been accepted at CVPR 2023
*   2022-12-21: Paper has been uploaded on arXiv


## Overview

**TruFor** is a forensic framework that can be applied to a large variety of image manipulation methods, from classic cheapfakes to more recent manipulations based on deep learning. We rely on the extraction of both high-level and low-level traces through a transformer-based fusion architecture that combines the RGB image and a learned noise-sensitive fingerprint. The latter learns to embed the artifacts related to the camera internal and external processing by training only on real data in a self-supervised manner. Forgeries are detected as deviations from the expected regular pattern that characterizes each pristine image. Looking for anomalies makes the approach able to robustly detect a variety of local manipulations, ensuring generalization. In addition to a pixel-level **localization map** and a whole-image **integrity score**, our approach outputs a **reliability map** that highlights areas where localization predictions may be error-prone. This is particularly important in forensic applications in order to reduce false alarms and allow for a large scale analysis. Extensive experiments on several datasets show that our method is able to reliably detect and localize both cheapfakes and deepfakes manipulations outperforming state-of-the-art works.


## Architecture

<center> <img src="./docs/architecture.png" alt="architecture" width="80%" /> </center>

We cast the forgery localization task as a supervised binary segmentation problem, combining high-level (**RGB**) and low-level (**Noiseprint++**) features using a cross-modal framework.


## Setup (docker)

Go in the test_docker folder and run the following command:
```
bash docker_build.sh
```

This command will also automatically download the [weights](https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip) and unzip them in the "test_docker/weights" folder. 
MD5 is 7bee48f3476c75616c3c5721ab256ff8.



## Usage

To run all images in "images/" directory, run:
```
bash docker_run.sh
```

You can change the following parameters in docker_run.sh:
- *-gpu*: default is gpu '0'. Put '-1' if you want to use cpu.
- *-in*:  default is "images/"
 is used and the input folder is "images/". It can be a single file (data/tampered1.png), a directory (data/) or a glob statement (data/*.png)
- If you want to save the Noiseprint++ aswell, you can add the flag ```--save_np``` at the end of the command.

The output is a .npz containing the following files:
- *'map'*: anomaly localization map
- *'conf'*: confidence map
- *'score'*: score in the range [0,1]
- *'np++'*: Noiseprint++ (if flag ```--save_np``` was specified)
- *'imgsize'*: size of the image

Note: if the output file already exists, it is not overwritten and it is skipped

Note: that the score values can slightly change when a different version of python, pytorch, cuda, cudnn, or other libraries changes.


## Visualization

To visualize the output for an image, run the following:
```
python visualize.py --image image_path --output output_path [--mask mask_path]
```
Providing the mask is optional.

Note that ```matplotlib``` and ```pillow``` packages are required to run this script.


## Metrics

In the file ```metrics.py``` you can find the functions we used to compute the metrics. <br/>
Localization metrics have to be computed only on fake images, and the ground truth **has to be 0 for pristine pixels and 1 for forged pixels**. <br/>
When computing F1 score, we take the maximum between the F1 using the localization map and the F1 using the inverse of the localization map.
We do not consider pixels close to the borders of the forged area in the ground truth, since in most cases they are not accurate. 


## CocoGlide dataset

You can download the CocoGlide dataset [here](https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip).


## License

Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA'). 

All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package) 


## Bibtex
 
 ```
 @InProceedings{Guillaro_2023_CVPR,
    author    = {Guillaro, Fabrizio and Cozzolino, Davide and Sud, Avneesh and Dufour, Nicholas and Verdoliva, Luisa},
    title     = {TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {20606-20615}
}
```


## Acknowledgments
 
We gratefully acknowledge the support of this research by the Defense Advanced Research Projects Agency (DARPA) under agreement number FA8750-20-2-1004. 
The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright notation thereon.
The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of DARPA or the U.S. Government.

In addition, this work has received funding by the European Union under the Horizon Europe vera.ai project, Grant Agreement number 101070093, and is supported by Google and by the PREMIER project, funded by the Italian Ministry of Education, University, and Research within the PRIN 2017 program.
Finally, we would like to thank Chris Bregler for useful discussions and support.
