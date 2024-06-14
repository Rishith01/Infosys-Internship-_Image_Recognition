import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Reshape, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

def ssd_vgg16(num_classes):
    input_shape = (300, 300, 3)
    base_model = VGG16(input_shape=input_shape, include_top=False)
    
    # Extract feature maps from base model
    feature_maps = [
        base_model.get_layer('block4_conv3').output,
        base_model.get_layer('block5_conv3').output
    ]
    
    # Define extra layers for SSD
    x = base_model.output
    x = Conv2D(1024, (3, 3), padding='same', activation='relu', name='conv6')(x)
    x = Conv2D(1024, (1, 1), padding='same', activation='relu', name='conv7')(x)
    
    # Add additional feature maps
    feature_maps.append(x)
    for i in range(8, 12):
        x = Conv2D(256, (1, 1), padding='same', activation='relu', name=f'conv{i}')(x)
        x = Conv2D(512, (3, 3), padding='same', strides=2, activation='relu', name=f'conv{i+1}')(x)
        feature_maps.append(x)
    
    # Define the SSD head
    def ssd_head(x, num_priors, name):
        num_boxes = num_priors * (4 + num_classes)
        x = Conv2D(num_boxes, (3, 3), padding='same', name=name)(x)
        x = Reshape((-1, 4 + num_classes), name=f'{name}_reshape')(x)
        return x
    
    # Predict class scores and bounding boxes
    num_priors = [4, 6, 6, 6, 4, 4]  # Example prior box configuration
    predictions = [ssd_head(fm, num_priors[i], f'pred_{i}') for i, fm in enumerate(feature_maps)]
    
    # Concatenate all predictions
    predictions = Concatenate(axis=1, name='predictions')(predictions)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
