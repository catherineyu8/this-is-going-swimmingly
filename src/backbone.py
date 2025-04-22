import tensorflow as tf
from tensorflow import keras
import math
from typing import Dict, List, Optional
from position_encoding import build_position_encoding

class NestedTensor:
    # mimics same pytorch method
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

class FrozenBatchNorm2d(keras.layers.Layer):
    def __init__(self, n, **kwargs):
        super(FrozenBatchNorm2d, self).__init__(**kwargs)
        self.weight = self.add_weight(name='weight', shape=(n,), initializer='ones', trainable=False) # all frozen
        self.bias = self.add_weight(name='bias', shape=(n,), initializer='zeros', trainable=False) 
        self.running_mean = self.add_weight(name='running_mean', shape=(n,), initializer='zeros', trainable=False)
        self.running_var = self.add_weight(name='running_var', shape=(n,), initializer='ones', trainable=False)

    def call(self, x):
        w = tf.reshape(self.weight, [1, 1, 1, -1]) #tf switches channels from 1 to 3
        b = tf.reshape(self.bias, [1, 1, 1, -1])
        rv = tf.reshape(self.running_var, [1, 1, 1, -1])
        rm = tf.reshape(self.running_mean, [1, 1, 1, -1])
        eps = 1e-5
        scale = w * tf.math.rsqrt(rv + eps)
        bias = b - rm * scale
        return x * scale + bias

class IntermediateLayerGetter(keras.Model):
    def __init__(
        self, 
        model: keras.Model, 
        return_layers: Dict[str, str],
        # resnet_mapping: Optional[Dict[str, str]] = None # i think would be easier to do this way
    ):
        super().__init__()
        self.model = model
        self.return_layers = return_layers
        
        # if using resnet_mapping insert if else
        self.layer_names = {
            "0": "conv2_block3_out",  #keras layer ids???
            "1": "conv3_block4_out",  #think this should work, have to figure out how to trace back
            "2": "conv4_block6_out", 
            "3": "conv5_block3_out"   
        }
        # self.layer_mapping = resnet_mapping(model)
        
        # extract features we want from return_layers
        self.output_layers = []
        for output_key, layer_name in self.layer_names.items():
            if output_key in self.return_layers.values():
                try:
                    layer = model.get_layer(layer_name)
                    self.output_layers.append((output_key, layer))
                except ValueError:
                    print(f"Warning: Layer {layer_name} not found in model")
        
        if self.output_layers:
            outputs = [layer.output for _, layer in self.output_layers]
            self.feature_extractor = keras.Model(inputs=model.input, outputs=outputs)
        else:
            print("Warning: No matching layers found")
            self.feature_extractor = None
    
    def call(self, x):
        if self.feature_extractor is None:
            return {}
        features = self.feature_extractor(x)
        features = features if isinstance(features, list) else [features] # make list
        # translate back to original names
        reverse_mapping = {v: k for k, v in self.return_layers.items()}
        
        outputs = {}
        for i, (output_key, _) in enumerate(self.output_layers):
            # find original layer name
            original_name = reverse_mapping[output_key]
            outputs[original_name] = features[i]
        return outputs

class BackboneBase(keras.Model):
    def __init__(self, backbone, train_backbone, num_channels, return_interm_layers, **kwargs):
        super().__init__(**kwargs)
        
        if not train_backbone:
            backbone.trainable = False
        else:
            for layer in backbone.layers:
                if not any(layer_name in layer.name for layer_name in ['conv2_block', 'conv3_block', 'conv4_block', 'conv5_block']):
                    layer.trainable = False
        
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
    
    def call(self, tensor_list):
        # Handle inputs as either dict or NestedTensor
        tensors = tensor_list.tensors
        mask = tensor_list.mask
        
        xs = self.body(tensors)
        out = {}
        
        for name, x in xs.items():
            # Resize mask to match feature map size
            m = mask
            m_expanded = tf.expand_dims(tf.cast(m, tf.float32), axis=0)
            resized_mask = tf.image.resize(m_expanded, size=tf.shape(x)[1:3])
            resized_mask = tf.cast(resized_mask > 0.5, tf.bool)[0]
            
            out[name] = NestedTensor(tensors=x, mask=resized_mask)
        
        return out

class Backbone(BackboneBase):
    def __init__(self, name, train_backbone, return_interm_layers, dilation, **kwargs):
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet', 
            include_top=False,
            input_shape=(None, None, 3)
        )
        # ResNet50 has 2048 channels in final layer
        num_channels = 2048
        
        super().__init__(base_model, train_backbone, num_channels, return_interm_layers, **kwargs)

class Joiner(keras.Model):
    def __init__(self, backbone, position_embedding, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.num_channels = backbone.num_channels
    
    def call(self, tensor_list):
        xs = self.backbone(tensor_list)
        out = []
        pos = []
        
        for name, x in xs.items():
            out.append(x)
            position = self.position_embedding(x)
            position = tf.cast(position, x.tensors.dtype)
            pos.append(position)
        
        return out, pos

def build_backbone(args):    # in general, backbone called in model to extract features. that is what imlayergetter does
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

#TEST BLOCK
if __name__ == "__main__":
    model = Backbone(name="resnet50", train_backbone=False, return_interm_layers=True, dilation=False)
    dummy_input = tf.random.uniform((1, 224, 224, 3))
    dummy_mask = tf.ones((224, 224), dtype=tf.bool)
    nested_input = NestedTensor(dummy_input, dummy_mask)
    
    outputs = model(tensor_list=nested_input)
    
    for name, out in outputs.items():
        print(f"{name}: tensor shape = {out.tensors.shape}, mask shape = {out.mask.shape}")