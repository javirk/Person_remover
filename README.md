# Person Remover

_Person remover_ es un proyecto que combina la arquitectura Pix2Pix con Yolo para eliminar a las personas y objetos de 
las fotos. Para Pix2Pix se ha adaptado el código de [Tensorflow](https://www.tensorflow.org/beta/tutorials/generative/pix2pix), y
para Yolo, de https://github.com/zzh8829/yolov3-tf2.

Este proyecto es capaz de eliminar objetos tanto en imágenes como en vídeo.

Se ha utilizado Python 3.7 y Tensorflow 2.0-beta


## ¿Cómo funciona?

Se han combinado YOLO con Pix2Pix para eliminar personas de las fotos. Para ello, se ha tomado una red YOLO preentrenada
que se encarga de detectar los objetos de las imágenes (generando una _bounding box_ a su alrededor) y se ha utiilizado después 
una Pix2Pix que ha aprendido a rellenar huecos en el centro de las imágenes, tomando como referencia las imágenes sin agujero:
1. YOLO detecta los objetos
2. Se toma una subimagen con cada uno de los objetos, añadiendo píxeles a su alrededor
3. De cada subimagen se elimina a la persona, que se encuentra en el centro, y es enviada al generador de Pix2Pix para 
que rellene a partir de los píxeles que quedan.

Con el objeto de ilustrar el proceso de entrenamiento de Pix2Pix, se pueden observar las siguientes imágenes, en las que
se ha generado un agujero y el generador ha aprendido a rellenarlo.
![p2p_fill_1](https://raw.githubusercontent.com/javirk/Person_remover/master/images_readme/salida_23_0.png)
![p2p_fill_2](https://raw.githubusercontent.com/javirk/Person_remover/master/images_readme/salida_0_0.png)

Estas instrucciones te ayudarán a entrenar un modelo en tu máquina local. Sin embargo, los datos de entrenamiento que se han utilizado
para Pix2Pix [no son públicos](http://graphics.cs.cmu.edu/projects/whatMakesParis/). Este conjunto consta de 14900 imágenes
256x256x3. El código se encarga de crear un agujero en el centro de las imágenes y aprender a rellenarlo con los datos
que hay alrededor.

Para YOLO se ha utilizado un modelo preentrenado.

### Prerequisites

Para utilizar el programa necesitarás Python 3.7 y los paquetes especificados en el archivo `requirements.txt`.

### Instalación

Clonar el repositorio
```
git clone https://github.com/javirk/Person_remover.git
```
Entrar en la carpeta `./yolo`, descargar los pesos de YOLO, convertirlos y moverlos a la carpeta `./yolo/data`
```
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py
```
Descargar los pesos de Pix2Pix de [Google Drive](https://drive.google.com/open?id=19VsarMcYRNPLTDr6b6ABJyY8JUeBueL8) y
colocarlos en `./pix2pix/checkpoint/`.

Para sacar resultados de imágenes tan solo hay que ejecutar el archivo `person_remover.py`:
```
python person_remover.py -i /ruta/a/imagenes/input
``` 

En un vídeo, por el contrario:
```
python person_remover.py -v /ruta/a/video
``` 

También es posible especificar el tipo de objeto a eliminar (por defecto serán las personas, mochilas y bolsos). Para hacer
esto:
```
python person_remover.py -i /ruta/a/imagenes/input -ob 1 2 3
``` 

Lo que eliminará los objetos especificados como 1, 2 y 3 (empezando desde 0) que aparecen en el archivo `yolo/data/coco.names`.
Es decir, bicicletas, coches y motos.

### Entrenamiento

La red YOLO se tomó preentrenada. Las redes que formaban parte del Pix2Pix han sido entrenadas durante 23 épocas en un
conjunto de 14900 imágenes de entrenamiento y 100 de test con los parámetros por defecto de Pix2Pix. Nótese que el entrenamiento
es tremendamente sensible, por lo que pueden no obtenerse los mejores resultados a la primera prueba.

Para ejecutar el entrenamiento con los parámetros por defecto, tras haber descargado los archivos:
```
python image_inpainting.py -train /ruta/a/imagenes/entrenamiento -test /ruta/a/imagenes/test -mode /test
```


## Eliminación en imágenes



### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
