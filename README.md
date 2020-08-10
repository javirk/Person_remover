# Person Remover

Versión en español disponible [aquí](README_es.md).

Would you like to travel to a touristic spot and yet appear alone in the photos? 

_Person remover_ is a project that combines Pix2Pix and YOLO arhitectures in order to remove people or other objects from
photos. For Pix2Pix, the code from [Tensorflow](https://www.tensorflow.org/beta/tutorials/generative/pix2pix) has been adapted,
whereas for YOLO, the code has been adapted from https://github.com/zzh8829/yolov3-tf2.

This project is capable of removing objects in images and video.

Python 3.7 and Tensorflow 2.0-beta have been used in this project.

#### Try it in [Google Colab](https://colab.research.google.com/drive/1JDpH8MAjaKoekQ_H9ZaxYJ9_axiDtDGm?usp=sharing).


## How does it work?

YOLO has been combined with Pix2Pix. A pre-trained YOLO network has been used for object detection (generating a bounding
box around them), and its output is fed to a Pix2Pix's generator that has learned how to fill holes in the center of images,
using the images without holes as a reference:
1. YOLO detects the objects
2. A subimage of every object is taken, adding the pixels around it
3. Out of every subimage, the center pixels are removed (replaced by ones) and the result is sent to the generator, whose
task is to fill it with the surrounding pixels.

In order to illustrate the training process of Pix2Pix, the following images can be observed. A hole has been drilled and 
the generator has learnt how to fill it.

![p2p_fill_1](https://github.com/javirk/Person_remover/blob/master/images_readme/fill_1.png)
![p2p_fill_2](https://github.com/javirk/Person_remover/blob/master/images_readme/fill_2.png)

These instructions will you train a model in your local machine. However, the training dataset that has been used for 
Pix2Pix are not [publicly available](http://graphics.cs.cmu.edu/projects/whatMakesParis/). This dataset consists of 14900,
256x256x3 images. The code handles the creation of a hole in the center of the images and learns how to fill it with the
surrounding data.

### Requisites

In order to use the program Python 3.7 and the libraries specified in  `requirements.txt` should be installed.

### Installation

Clone the repository
```
git clone https://github.com/javirk/Person_remover.git
```
Download and save the YOLO weights in the folder `./yolo`, convert them and move them to `./yolo/data`
```
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py
```
Download the weights for Pix2Pix from [Google Drive](https://drive.google.com/open?id=19VsarMcYRNPLTDr6b6ABJyY8JUeBueL8)
and put them in `./pix2pix/checkpoint/`.

To get results of images, run `person_remover.py`:
```
python person_remover.py -i /dir/of/input/images
``` 
In a video, in contrast:
```
python person_remover.py -v /dir/of/video
``` 
It is also possible to specify the type of object to remove (people, bags and handbags are chosen by default):
```
python person_remover.py -i /dir/to/input/images -ob 1 2 3
``` 
Which will remove the objects specified as 1, 2 and 3 (starting from 0) that appear in the file `yolo/data/coco.names`.
In this case bikes, cars and motorbikes.

### Training

YOLO network is taken pretrained. For Pix2Pix networks, the training has spanned 23 epochs in a dataset of 14900 training
and 100 test images using the default parameters. It is worth noticing that the training process is extremely sensitive,
so the best results might not come in the first run.

Training with the default parameters is performed as follows:
```
python image_inpainting.py -train /dir/of/training/images -test /dir/of/test/images -mode /train
```

## Image removal

![p2p_fill_3](https://github.com/javirk/Person_remover/blob/master/images_readme/Imagen6.png)
![p2p_fill_4](https://github.com/javirk/Person_remover/blob/master/images_readme/Imagen7.png)
![p2p_fill_5](https://github.com/javirk/Person_remover/blob/master/images_readme/Imagen1.png)
![p2p_fill_6](https://github.com/javirk/Person_remover/blob/master/images_readme/Imagen2.png)
![p2p_fill_7](https://github.com/javirk/Person_remover/blob/master/images_readme/Imagen3.png)
![p2p_fill_8](https://github.com/javirk/Person_remover/blob/master/images_readme/Imagen4.png)
![p2p_fill_9](https://github.com/javirk/Person_remover/blob/master/images_readme/Imagen5.png)
![p2p_fill_10](https://github.com/javirk/Person_remover/blob/master/images_readme/Imagen8.png)


## Video removal

[A walking tour of Paris video](https://www.youtube.com/watch?v=_dRjY9gMcxE) has been used.

![p2p_fill_11](https://github.com/javirk/Person_remover/blob/master/images_readme/GIF.gif)

## Next steps

Results can be improved replacing the object detector network (YOLO) by a semantic segmentator. In this way, the generator
will have to fill just the part relative to the person, not the whole bounding box. Due to a matter of time and processing
capacity, this improvement could not be developed yet.

Modification of Pix2Pix by a more advanced architecture, such as Pix2PixHD.

## Author

* **Javier Gamazo** - *Initial work* - [Github](https://github.com/javirk). [LinkedIn](https://www.linkedin.com/in/javier-gamazo-tejero/)

## License

This project is under MIT license. See [LICENSE.md](LICENSE.md) for more details.

## Acknowledgments

* [zzh8829](https://github.com/zzh8829/yolov3-tf2) for YOLO's code
* [Tensorflow](https://www.tensorflow.org/) for Pix2Pix' code
