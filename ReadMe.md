# AIFFEL PoC_Lab_Incheon_3rd_rice

- **해커톤 주제**: AI를 활용한 벼 품종 분류
- **프로젝트명**: Rice classification using CNN

<br>

## 프로젝트 요약
벼 품종 개량을 위한 연구에서 벼 품종의 분류는 수작업으로 이루어지고 있다. 수작업으로 이루어지므로써 발생되는 시간적 손실을 해결하고자 본 글에서는 CNN을 이용한 벼 품종 분류 방법을 제안한다. 분류를 위한 학습 데이터는 전처리를 통해 미리 준비했다. 전처리가 이뤄진 이미지를 대상으로 특징 추출을 위해 세 계층의 합성곱 계층을 사용하였고, 분류를 위해 두 계층의 완전 연결 신경망을 사용하였다. 본 글에서 설계한 모델이 벼 품종 분류에 효과가 있음을 보이기 위해 다양한 성능평가 지표를 사용하였으며, 결과 정확도는 99.5%를 상회한다.
## 


![]()
<br>

## 구성원 및 역할

* 이평섭
[![Tech Blog Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github&link=https://github.com/fmfmsd)](https://github.com/fmfmsd)
  
  
  - 데이터 크롭 및 전처리
  - 모델 테스트 및 파인튜닝
 
  
* 강서준
  [![Tech Blog Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github&link=https://github.com/dyno-seojoon)](https://github.com/dyno-seojoon)
 
  - 데이터 크롭
  - 모델 테스트
  - 보고서 및 ppt 작성
  
<br>

---
![MS](https://github.com/DatatonProject/PoC_Lab_Incheon_3rd_Rice/blob/main/Data_Sample/pic/milestone.png)
---

<br>

## 사용한 기술 스택

- Goole Colab, Jupyter Notebook
- Google Cloud Platform
- Keras.API

<br>

## 프로젝트 세부 동작

### Input 이미지가 되는 벼를 CNN model을 이용하여 학습한다. 합성곱 계층은 Conv2D-Batch Normalization-MaxPooling2D을 순차적으로 연결하여 구성하였다.


<br>

#### 1. splitfolders로 train, val, test를 각 8:1:1 비율로 나눈다.
```python
splitfolders.ratio(base_ds, output='imgs', seed=123, ratio=(.8,.1,.1), group_prefix=None)

# (Copying files: 20003 files [06:23, 52.18 files/s])
```

<br>

```python
japonica_norm = [fn for fn in os.listdir(f'{base_ds}/japonica_norm') if fn.endswith('.png')]
japonica_extra = [fn for fn in os.listdir(f'{base_ds}/japonica_extra') if fn.endswith('.png')]
indica_norm = [fn for fn in os.listdir(f'{base_ds}/indica_norm') if fn.endswith('.png')]
indica_extra = [fn for fn in os.listdir(f'{base_ds}/indica_extra') if fn.endswith('.png')]
rice = [japonica_norm, japonica_extra, indica_norm, indica_extra]
rice_classes = []
for i in os.listdir('imgs/train'):
    rice_classes+=[i]
rice_classes.sort()
```

<br>

#### 2. 이미지 전처리 및 라벨링

```python
datagen = ImageDataGenerator(rescale=1./255)

#rescale = 1./255

# 원본 이미지 파일은 0-255의 RGB 계수로 구성되는데, 
# 입력값은 모델을 효과적으로 학습시키기에 너무 높다. (통상적인 learning rate를 사용할 경우). 
# 따라서 전처리 과정에 앞서 가장 먼저 적용하고 1/255로 스케일링하여 0-1 범위로 변환. 

```

```python
train_ds = datagen.flow_from_directory(
    'imgs/train',
    target_size = (img_height, img_width),
    batch_size = batch_size,
    subset = "training",
    class_mode='categorical')

val_ds = datagen.flow_from_directory(
    'imgs/val',
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode='categorical',
    shuffle=False)

test_ds = datagen.flow_from_directory(
    'imgs/test',
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode='categorical',
    shuffle=False)
    
```

<br>

![rice](https://github.com/DatatonProject/PoC_Lab_Incheon_3rd_Rice/blob/main/Data_Sample/pic/000.%20%EC%8C%80.png?raw=true)

<br>

#### 3. 학습

바닐라CNN
```python

model_vanilla = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=input_shape),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(axis = 3),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same'),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(axis = 3),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same'),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(axis = 3),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same'),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(4, activation='softmax')
])
```

```python
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 248, 248, 32)      896       
                                                                 
 batch_normalization (BatchN  (None, 248, 248, 32)     128       
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 248, 248, 32)      9248      
                                                                 
 batch_normalization_1 (Batc  (None, 248, 248, 32)     128       
 hNormalization)                                                 
                                                                 
 max_pooling2d (MaxPooling2D  (None, 124, 124, 32)     0         
 )                                                               
                                                                 
 dropout (Dropout)           (None, 124, 124, 32)      0         
                                                                 
 conv2d_2 (Conv2D)           (None, 124, 124, 64)      18496     
                                                                 
 batch_normalization_2 (Batc  (None, 124, 124, 64)     256       
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 124, 124, 64)      36928     
                                                                 
 batch_normalization_3 (Batc  (None, 124, 124, 64)     256       
 hNormalization)                                                 
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 62, 62, 64)       0         
 2D)                                                             
                                                                 
 dropout_1 (Dropout)         (None, 62, 62, 64)        0         
                                                                 
 conv2d_4 (Conv2D)           (None, 62, 62, 128)       73856     
                                                                 
 batch_normalization_4 (Batc  (None, 62, 62, 128)      512       
 hNormalization)                                                 
                                                                 
 conv2d_5 (Conv2D)           (None, 62, 62, 128)       147584    
                                                                 
 batch_normalization_5 (Batc  (None, 62, 62, 128)      512       
 hNormalization)                                                 
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 31, 31, 128)      0         
 2D)                                                             
                                                                 
 dropout_2 (Dropout)         (None, 31, 31, 128)       0         
                                                                 
 flatten (Flatten)           (None, 123008)            0         
                                                                 
 dense_2 (Dense)             (None, 512)               62980608  
                                                                 
 batch_normalization_6 (Batc  (None, 512)              2048      
 hNormalization)                                                 
                                                                 
 dropout_3 (Dropout)         (None, 512)               0         
                                                                 
 dense_3 (Dense)             (None, 128)               65664     
                                                                 
 dropout_4 (Dropout)         (None, 128)               0         
                                                                 
 dense_4 (Dense)             (None, 4)                 516       
                                                                 
=================================================================
Total params: 63,337,636
Trainable params: 63,335,716
Non-trainable params: 1,920
_________________________________________________________________
```

<br>

```python

vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
vgg16.trainable = False
inputs = tf.keras.Input(input_shape)
x = vgg16(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(4, activation='softmax')(x)
model_vgg16 = tf.keras.Model(inputs, x)

```

```python

model_vgg16.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_vgg16.summary()
Model: "model_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_4 (InputLayer)        [(None, 250, 250, 3)]     0         
                                                                 
 vgg16 (Functional)          (None, 7, 7, 512)         14714688  
                                                                 
 global_average_pooling2d_1   (None, 512)              0         
 (GlobalAveragePooling2D)                                        
                                                                 
 dense_5 (Dense)             (None, 1024)              525312    
                                                                 
 dense_6 (Dense)             (None, 4)                 4100      
                                                                 
=================================================================
Total params: 15,244,100
Trainable params: 529,412
Non-trainable params: 14,714,688
_________________________________________________________________
```

```python

#finetune
vgg16.trainable = True
model_vgg16.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

```

<br>

# Vanilla CNN Confusion Matrix

![ccm](https://github.com/DatatonProject/PoC_Lab_Incheon_3rd_Rice/blob/main/Data_Sample/pic/4.%20CNN%20Confusion.png)

<br>

# Vgg16 Confusion Matrix

![vcm](https://github.com/DatatonProject/PoC_Lab_Incheon_3rd_Rice/blob/main/Data_Sample/pic/8.%20Vgg16%20Confusion.png)

<br>

#### 4. 최종 출력


## GradCAM
```python

list_images_sample = ["/content/imgs/train/indica_extra/indica (5000).png",
                      "/content/imgs/test/indica_norm/indica (1002).png",
                      "/content/imgs/test/japonica_extra/japonica (10078).png",
                      "/content/imgs/test/japonica_norm/japonica (1048).png"]

model_builder = keras.applications.xception.Xception
img_size = (299, 299)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions
imag = []

last_conv_layer_name = "block14_sepconv2_act"
# 이미지를 np array로

def get_img_array(img_path, size):
    img = keras.preprocessing.image.load_img(img_path, target_size = size) 
    array = keras.preprocessing.image.img_to_array(img) 
    array = np.expand_dims(array, axis = 0)
    return array

# Top Heatmap sampling

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index = None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Heat map sorting

covid_noncovid_heatmap = []

for i in list_images_sample:
    img_array = preprocess_input(get_img_array(i, size = img_size))
    model = model_builder(weights = "imagenet")
    model.layers[-1].activation = None
    preds = model.predict(img_array)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    covid_noncovid_heatmap.append(heatmap)

# GradCAM 출력

def save_and_display_gradcam(img_path, heatmap, cam_path = "cam.jpg", alpha = 0.4):
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("hsv")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)

    imag.append(cv2.imread(img_path))
    imag.append(cv2.imread("./cam.jpg"))


for i in range(len(list_images_sample)):
    save_and_display_gradcam(list_images_sample[i], covid_noncovid_heatmap[i])


def plot_multiple_img(img_matrix_list, title_list, ncols, main_title = ""):

    fig, myaxes = plt.subplots(figsize = (15, 8), nrows = 2, ncols = ncols, squeeze = False)
    fig.suptitle(main_title, fontsize = 18)
    fig.subplots_adjust(wspace = 0.3)
    fig.subplots_adjust(hspace = 0.3)

    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize = 15)

    plt.show()



titles_list = ["indica_extra",'indica_extra Grad', 'japonica_norm','japonica_norm Grad', 'japonica_extra','japonica_extra Grad','indica_norm','indica_norm Grad']

plot_multiple_img(imag, titles_list, ncols = 4, main_title = "RICE TYPE Image Analysis")

```

<br>

![Grad cam](https://github.com/DatatonProject/PoC_Lab_Incheon_3rd_Rice/blob/main/Data_Sample/pic/11.%20cam.png)

<br>


# Pred & True

```python

plt.figure(figsize=(10, 10))
x, label= train_ds.next()
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x[i])
    result = np.where(label[i]==1)
    predict = model_vgg16(tf.expand_dims(x[i], 0))
    score = tf.nn.softmax(predict[0])
    score_label = rice_classes[np.argmax(score)]
    plt.title(f'Truth: {rice_classes[result[0][0]]}\nPrediction:{score_label}')
    plt.axis(False)

```

<br>

![pred&true](https://github.com/DatatonProject/PoC_Lab_Incheon_3rd_Rice/blob/main/Data_Sample/pic/11.%20pred%20%26%20true.png)

<br>

## Reference
* [CNN](https://github.com/sjchoi86/dl_tutorials_10weeks/blob/master/papers/ImageNet%20Classification%20with%20Deep%20Convolutional%20Neural%20Networks.pdf)
* [VGGnet](https://arxiv.org/pdf/1409.1556.pdf)
* [Fine-Tuning VGG Neural Network For Fine-grained State Recognition of Food Images](https://arxiv.org/ftp/arxiv/papers/1809/1809.09529.pdf)
* [Grad-CAM](https://arxiv.org/pdf/1610.02391.pdf)
* [Keras_API](https://keras.io/api/)
