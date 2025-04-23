import math
import tensorflow as tf
from tensorflow import keras

class PositionEmbeddingSine(keras.layers.Layer): #switched to keras in params, otherwise class is like the same
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, **kwargs):
        super().__init__(**kwargs) # pass to base
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def call(self, tensor, mask): #switch to call so works w tensorflow
        mask = tf.cast(mask, tf.bool)
        
        not_mask = tf.cast(~mask, tf.float32)
        
        y_embed = tf.cumsum(not_mask, axis=1)
        x_embed = tf.cumsum(not_mask, axis=2)
        
        if self.normalize: #option given in original code
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        

        dim_t = tf.range(self.num_pos_feats, dtype=tf.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        # positional embeddings for x and y
        pos_x = tf.expand_dims(x_embed, axis=-1) / dim_t
        pos_y = tf.expand_dims(y_embed, axis=-1) / dim_t
        
        # Stack x and y encodings
        pos_x_sin = tf.sin(pos_x[:, :, :, 0::2])
        pos_x_cos = tf.cos(pos_x[:, :, :, 1::2])
        pos_x = tf.stack([pos_x_sin, pos_x_cos], axis=4)
        pos_x = tf.reshape(pos_x, shape=tf.concat([tf.shape(pos_x)[:3], [self.num_pos_feats]], axis=0))
        
        pos_y_sin = tf.sin(pos_y[:, :, :, 0::2])
        pos_y_cos = tf.cos(pos_y[:, :, :, 1::2])
        pos_y = tf.stack([pos_y_sin, pos_y_cos], axis=4)
        pos_y = tf.reshape(pos_y, shape=tf.concat([tf.shape(pos_y)[:3], [self.num_pos_feats]], axis=0))
        
        pos = tf.concat([pos_y, pos_x], axis=3)  # [batch_size, H, W, 2*num_pos_feats]
        
        return pos
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_pos_feats': self.num_pos_feats,
            'temperature': self.temperature,
            'normalize': self.normalize,
            'scale': self.scale,
        })
        return config

class PositionEmbeddingLearned(keras.layers.Layer):
    def __init__(self, num_pos_feats=256, **kwargs):
        super().__init__(**kwargs)
        self.num_pos_feats = num_pos_feats
        self.row_embed = keras.layers.Embedding(50, num_pos_feats)
        self.col_embed = keras.layers.Embedding(50, num_pos_feats)

    def call(self, tensor, mask=None):
        batch_size = tf.shape(tensor)[0]
        h, w = tf.shape(tensor)[1], tf.shape(tensor)[2]

        i = tf.range(w, dtype=tf.int32)
        j = tf.range(h, dtype=tf.int32)

        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        x_emb = tf.expand_dims(x_emb, axis=0)  # [1, w, num_pos_feats]
        x_emb = tf.repeat(x_emb, h, axis=0)     # [h, w, num_pos_feats]

        y_emb = tf.expand_dims(y_emb, axis=1)  # [h, 1, num_pos_feats]
        y_emb = tf.repeat(y_emb, w, axis=1)     # [h, w, num_pos_feats]

        pos = tf.concat([y_emb, x_emb], axis=-1)  # [h, w, 2*num_pos_feats]
        pos = tf.expand_dims(pos, axis=0)         # [1, h, w, 2*num_pos_feats]
        pos = tf.repeat(pos, batch_size, axis=0)  # [batch_size, h, w, 2*num_pos_feats]

        return pos

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_pos_feats': self.num_pos_feats,
        })
        return config

    
def build_position_encoding(args): # I think this should be same
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")
    return position_embedding

#TEST BLOCK
if __name__ == "__main__":
    batch_size = 2
    height = 10
    width = 12
    channels = 256

    dummy_tensor = tf.random.normal((batch_size, height, width, channels))
    dummy_mask = tf.zeros((batch_size, height, width), dtype=tf.int32)

    print("Testing PositionEmbeddingSine...")
    sine_embed = PositionEmbeddingSine(num_pos_feats=64, normalize=True)
    pos_sine = sine_embed(dummy_tensor, dummy_mask)
    print(f"Output shape (sine): {pos_sine.shape}")  # Expect [2, 10, 12, 128]
    print(f"Output sample (sine) at [0,0,0]: {pos_sine[0,0,0].numpy()[:8]}")

    print("\nTesting PositionEmbeddingLearned...")
    learned_embed = PositionEmbeddingLearned(num_pos_feats=64)
    pos_learned = learned_embed(dummy_tensor)
    print(f"Output shape (learned): {pos_learned.shape}")  # Expect [2, 10, 12, 128]
    print(f"Output sample (learned) at [0,0,0]: {pos_learned[0,0,0].numpy()[:8]}")

