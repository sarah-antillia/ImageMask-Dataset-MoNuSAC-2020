<h2>ImageMask-Dataset-MoNuSAC-2020</h3>
This is 512x512 JPEG PreAugmented MoNuSAC 2020 ImageMaskDatatset for Imae Segmentation.<br>

The original dataset used here has been take from the following web-site.<br>

<a href="https://monusac-2020.grand-challenge.org/Data/">Challenges MoNuSAC 2020 Data</a>
<br>
<br>
<b>Download PreAugmented-MoNuSAC-Imag</b><br>
You can down load our dataset created here from the google drive 
<a href="https://drive.google.com/file/d/1sMmUW6Gqm9hBNdmIHJ60qhgj14n_TPZg/view?usp=sharing">PreAugmented-MoNuSAC-ImageMask-Dataset-V1.zip</a>.<br>

<h3>1. Dataset Citatioin</h3>
The orginal dataset use here has been taken
<a href="https://monusac-2020.grand-challenge.org/Data/"
<b>Challenges/MoNuSAC 2020/Data</b></a><br>
<br>
<b>Data</b><br>
H&E staining of human tissue sections is a routine and most common protocol used by pathologists 
to enhance the contrast of tissue sections for tumor assessment (grading, staging, etc.) at multiple microscopic resolutions. Hence, we will provide the annotated dataset of H&E stained digitized tissue images of several patients acquired at multiple hospitals using one of the most common 40x scanner magnification. The annotations will be done with the help of expert pathologists. 
<br>
<br>
<b>License</b><br>
The challenge data is released under the creative commons license (CC BY-NC-SA 4.0).
<br>

<h3>2. Download Training and Testing dataset</h3>
If you would like to create your own dataset, please download Trainign ad Test dataset from the following link.<br>
<a href="https://drive.google.com/file/d/1lxMZaAPSpEHLSxGA9KKMt_r-4S8dwLhq/view">MoNuSAC_images_and_annotations.zip</a><br><br>
<a href="https://drive.google.com/file/d/1G54vsOdxWY1hG7dzmkeK3r0xz9s-heyQ/view?usp=sharing">MoNuSAC Testing Data and Annotations.zip</a><br><br>


Training and Testing dataset contains many svs and tif image files, and xml anntotation files as shown below.<br> 

<pre>
./MoNuSAC_images_and_annotation
├─TCGA-5P-A9K0-01Z-00-DX1
│  ├─TCGA-5P-A9K0-01Z-00-DX1_1.svs
│  ├─TCGA-5P-A9K0-01Z-00-DX1_1.tif
│  ├─TCGA-5P-A9K0-01Z-00-DX1_1.xml
│  ├─TCGA-5P-A9K0-01Z-00-DX1_2.svs
│  ├─TCGA-5P-A9K0-01Z-00-DX1_2.tif
│  └─TCGA-5P-A9K0-01Z-00-DX1_2.xml
├─TCGA-55-1594-01Z-00-DX1

...

</pre>

<pre>
./MoNuSAC Testing Data and Annotations
├─TCGA-2Z-A9JG-01Z-00-DX1
│  ├─TCGA-2Z-A9JG-01Z-00-DX1_1.svs
│  ├─TCGA-2Z-A9JG-01Z-00-DX1_1.tif
│  ├─TCGA-2Z-A9JG-01Z-00-DX1_1.xml
│  ├─TCGA-2Z-A9JG-01Z-00-DX1_2.svs
│  ├─TCGA-2Z-A9JG-01Z-00-DX1_2.tif
│  └─TCGA-2Z-A9JG-01Z-00-DX1_2.xml
├─TCGA-2Z-A9JN-01Z-00-DX1

...

</pre>


<h3>3. Generate ImageMask master</h3>
 
Please run the following command for Python script <a href="./ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a><br>
<pre>
>python ImageMaskDatasetGenerator,py True
</pre>.
This command generate <b>PreAugmented-MoNuSAC-master</b> from tif and xml files in MoNuSAC_images_and_annotation dataset,
and <b>MoNuSAC-mini-test</b>
from MoNuSAC Testing Data and Annotations.<br>

This script generates 512x512 resized JPEG image and colorized mask files by using some offline augmentation methods in the script 
from the training dataset, and non-resized image and colorized mask files from the testing dataset.
<pre>
./PreAugmented-MoNuSAC-master
├─images
└─masks
</pre>
<pre>
./MoNuSAC-mini-test
├─images
└─masks
</pre>

We used the following BGR colors for the 4 categories annotations to create colorized masks.  

<pre>
colormap = {'Macrophage':(255, 0, 0), 
            'Epithelial':(0, 255, 0), 
            'Neutrophil':(0, 0, 255), 
            'Lymphocyte':(0, 255, 255)}

</pre>
<b>Attribute name in annotation<br>
<img src = "./asset/json_annotation_category.png" width = "640" height="auto"><br>
<br>
<b>Vertex in annotation<br>
<img src = "./asset/json_annotation_vertex.png" width = "640" height="auto"><br>
<br>
<br>
<h3>4. Split master </h3>
Please run the following command for Python script <a href="./split_master.py">split_mastr</a>.<br>

<pre>
>python split_master.py
</pre>
<hr>
This command generates PreAugmented-MoNuSAC-ImageMask-Dataset-V1 dataset.<br>
<pre>
./ PreAugmented-MoNuSAC-ImageMask-Dataset-V1
├─test
│  ├─images
│  └─masks
├─train
│  ├─images
│  └─masks
└─valid
    ├─images
    └─masks
</pre>
<hr>
<b>Train images sample</b><br>
<img src="./asset/train_images_sample.png" width="1024" height="auto"><br>
<br>
<b>Train masks sample</b><br>
<img src="./asset/train_masks_sample.png" width="1024" height="auto"><br>

<hr>

<b>Dataset Statistics</b><br>
<img src="./PreAugmented-MoNuSAC-ImageMask-Dataset-V1_Statistics.png" width="540" height="auto"><br>


<hr>

<b>MoNuSAC-mini-test images sample</b><br>
<img src="./asset/mini-test_images_sample.png" width="1024" height="auto"><br>
<br>
<b>MoNuSAC-mini-test masks sample</b><br>
<img src="./asset/mini-test_masks_sample.png" width="1024" height="auto"><br>

<hr>


