import tensorflow as tf
import tensorflow.keras.layers as layers
from transformers import TFCLIPModel, BertConfig, BertTokenizer, TFBertModel
# from transformers.models.bert.modeling_bert import BertLayer


class RackleMuffin(tf.keras.Model):

    def __init__(self):
        super(RackleMuffin, self).__init__()

        self.embedding_size = 768
        self.num_classes = 2
        self.text_hidden_size = 512
        self.image_hidden_size = 768 # self.config.hidden_size
        self.num_attn_heads = 8
        self.adam_epsilon = 1e-8
        self.learning_rate = 5e-4
        self.clip_learning_rate = 1e-6
        self.max_len_clip_text = 77
        self.dropout_rate = 0.1
        self.dim_feedforward = 2048

        # CLIP model both image and text processing
        self.clip_model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.trainable = False
        # text and image linear layers to reshape CLIP output to hidden size
        self.text_linear = tf.keras.Sequential([
            layers.Dense(self.image_hidden_size, input_shape=(self.text_hidden_size,)),
            layers.Dropout(self.dropout_rate),
            layers.Activation(tf.keras.activations.gelu)  # or tf.keras.layers.GELU() in newer TF versions
        ])
        self.image_linear = tf.keras.Sequential([
            layers.Dense(self.image_hidden_size, input_shape=(self.image_hidden_size,)),
            layers.Dropout(self.dropout_rate),
            layers.Activation(tf.keras.activations.gelu)  # or tf.keras.layers.GELU() in newer TF versions
        ])
        
        # RESNET for image processing
        # TODO: the paper joins resnet with position encoding
        # TODO: check params for resnet/use their own backbone. include_top=False removes final classificaiton layer
        self.resnet_backbone = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.resnet_linear = layers.Dense(self.image_hidden_size)

        # BERT model for text processing
        # Load tokenizer (same as in PyTorch)
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # self.bert_config = BertConfig.from_pretrained("bert-base-uncased")
        self.bert_model = TFBertModel.from_pretrained("bert-base-uncased")
        self.bert_model.trainable = False
        self.bert_linear = tf.keras.Sequential([
            layers.Dense(self.embedding_size),
            layers.ReLU(),
            layers.LayerNormalization()
        ])
        
        # SFIM (SHALLOW FEATURE INTERACTION MODULE)
        self.sfim_imgtext_cross = CrossAtten(self.embedding_size, self.num_attn_heads, self.dim_feedforward, self.dropout_rate)

        # RCLM (RELATIONAL CONTEXT LEARNING MODULE)
        # text cross atten
        self.text_linear2 = tf.keras.Sequential([
            layers.Dense(self.embedding_size, input_shape=(self.embedding_size*2,)),
            layers.ReLU(),
            layers.Dense(self.embedding_size),
            layers.LayerNormalization()
        ])
        self.text_self_atten = SelfAtten(self.embedding_size, self.num_attn_heads, self.dim_feedforward, self.dropout_rate)
        self.text_cros_atten = CrossAtten(self.embedding_size, self.num_attn_heads, self.dim_feedforward, self.dropout_rate)

        # img cross atten
        self.image_linear2 = tf.keras.Sequential([
            layers.Dense(self.embedding_size, input_shape=(self.embedding_size*2,)),
            layers.ReLU(),
            layers.Dense(self.embedding_size),
            layers.LayerNormalization()
        ])
        self.image_self_atten = SelfAtten(self.embedding_size, self.num_attn_heads, self.dim_feedforward, self.dropout_rate)
        self.image_cros_atten = CrossAtten(self.embedding_size, self.num_attn_heads, self.dim_feedforward, self.dropout_rate)

        # co-atten btwn img and text
        self.co_atten = CoAtten(self.embedding_size, self.dropout_rate)

        # MuFFM (MULTIPLEX FEATURE FUSION MODULE)
        self.muffm_mlp_sigmoid = tf.keras.Sequential([
            layers.Dense(self.embedding_size),
            layers.ReLU(),
            layers.Dense(self.embedding_size, activation='sigmoid')
        ])
        self.muffm_mlp = tf.keras.Sequential([
            layers.Dense(self.embedding_size),
            layers.ReLU(),
            layers.Dense(self.embedding_size),
            layers.LayerNormalization()
        ])

        # PREDICTION
        self.classifier = layers.Dense(self.num_classes)
    
        # LOSS
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, inputs, text_data):
        # NOTE: batch_size = 32
        """
        CLIP Model Expectation
            inputs = {
                "input_ids": tf.Tensor,         # shape: (batch_size, sequence_length)
                "attention_mask": tf.Tensor,    # shape: (batch_size, sequence_length)
                "pixel_values": tf.Tensor       # shape: (batch_size, 3, 224, 224)
            }
        """
        clip_output = self.clip_model(**inputs, output_attentions=True)

        # extract text and image features from CLIP and reshape to the hidden size
        # text_features = clip_output['text_model_output']['last_hidden_state']  # 128，77，512
        # image_features = clip_output['vision_model_output']['last_hidden_state']  # 128，50，768
        clip_text_feature = clip_output['text_model_output']['pooler_output'] # 32, 512
        clip_image_feature = clip_output['vision_model_output']['pooler_output'] # 32, 768
        clip_text_feature = self.text_linear(clip_text_feature)  # 32, 768
        clip_image_feature = self.image_linear(clip_image_feature)  # 32, 768

        # RESNET
        transformed_images = tf.transpose(inputs["pixel_values"], perm=[0, 2, 3, 1]) # convert dim format to what tensorflow expects: (B, H, W, C)
        transformed_images = tf.keras.applications.resnet50.preprocess_input(transformed_images) # TODO: if we use their backbone def, maybe don't do this, but this is what resnet expects/needs
        resnet_img_features = self.resnet_backbone(transformed_images) # (32, 2048)
        resnet_img_features = self.resnet_linear(resnet_img_features) # (32, 768)

        # BERT
        text_data = text_data.tolist() # convert numpy array to python list for bert
        bert_encoded_input = self.bert_tokenizer(text_data, padding=True, truncation=True, return_tensors="tf")
        # TODO: move bert input to device?
        bert_output = self.bert_model(**bert_encoded_input, training=False)
        # last_hidden_states = bert_output.last_hidden_state  # 32,56,768
        bert_pooler_output = bert_output.pooler_output  # (32, 768)
        bert_txt_features = self.bert_linear(bert_pooler_output) # (32, 768)

        # SFIM (SHALLOW FEATURE INTERACTION MODULE)
        # expand dims to be able to do cross-atten
        resnet_img_features = tf.expand_dims(resnet_img_features, axis=1)  # (32, 1, 768)
        bert_txt_features = tf.expand_dims(bert_txt_features, axis=1)    # (32, 1, 768)

        sfim_img_txt = self.sfim_imgtext_cross(target=resnet_img_features, memory=bert_txt_features) # (32, 1, 768)
        sfim_img_txt = tf.squeeze(sfim_img_txt, axis=1)  # (32, 768)

        sfim_txt_im = self.sfim_imgtext_cross(target=bert_txt_features, memory=resnet_img_features) # (32, 1, 768)
        sfim_txt_im = tf.squeeze(sfim_txt_im, axis=1)  # (32, 768)

        # RCLM (RELATIONAL CONTEXT LEARNING MODULE)
        # for text:
        # concat and squeeze
        txt_cat_sqz = self.text_linear2(tf.concat([clip_text_feature, sfim_txt_im], axis=-1)) # (32, 768)
        txt_cat_sqz = tf.expand_dims(txt_cat_sqz, axis=1) # (32, 1, 768)
        # stack and self atten
        text_self = self.text_self_atten(tf.stack([clip_text_feature, sfim_txt_im], axis=1)) # (32,2,768)
        # cross attention on the 2 above ouptuts
        text_cross = self.text_cros_atten(target=txt_cat_sqz, memory=text_self) # 32,1,768
        text_cross = tf.squeeze(text_cross, axis=1) # (32, 768)

        # for images:
        # concat and squeeze
        img_cat_sqz = self.text_linear2(tf.concat([clip_image_feature, sfim_img_txt], axis=-1)) # (32, 768)
        img_cat_sqz = tf.expand_dims(img_cat_sqz, axis=1) # (32, 1, 768)
        # stack and self atten
        img_self = self.img_self_atten(tf.stack([clip_image_feature, sfim_img_txt], axis=1)) # (32,2,768)
        # cross attention on the 2 above ouptuts
        img_cross = self.img_cros_atten(target=img_cat_sqz, memory=img_self) # 32,1,768
        img_cross = tf.squeeze(img_cross, axis=1) # (32, 768)
        
        # co atten x2 between image and text outputs
        txt_co_atten = self.co_atten(text_cross, img_cross, img_cross) # 32, 768
        img_co_atten = self.co_atten(img_cross, text_cross, text_cross) # 32, 768
        
        # linear combo of outputs
        rclm_output = 0.6*img_co_atten + 0.4*txt_co_atten  # (32,768)

        # CLIP-VIEW FEATURE FUSION
        # cross atten btwn og CLIP img/txt outputs
        clip_txt_cross_atten = self.co_atten(clip_text_feature, clip_image_feature, clip_image_feature) # 32,768
        clip_img_cross_atten = self.co_atten(clip_image_feature, clip_text_feature, clip_text_feature) # 32,768
        # linear combo
        clip_fuse_features = 0.7*clip_txt_cross_atten + 0.3*clip_img_cross_atten # (32,768)

        # MuFFM (MULTIPLEX FEATURE FUSION MODULE)
        muffm_sigmoid = self.muffm_mlp_sigmoid(tf.concat([clip_fuse_features, rclm_output], axis=-1)) # 32,768
        muffm = self.muffm_mlp(tf.concat([clip_fuse_features, rclm_output], axis=-1)) # 32,768

        output = 0.5*clip_fuse_features + 0.5*(muffm_sigmoid+muffm) # (32,768)

        # PREDICTION
        logits = self.classifier(output) # (32,2)

        # softmax for probabilities -- Got rid of this for correct loss function!
        #probs = tf.nn.softmax(logits, axis=-1)

        # want to return logits for CE Loss
        return logits

# cross attention custom class (called TransformerCrossLayer in paper)
class CrossAtten(layers.Layer):
    # num_heads=8, d_model = embedding_size=768, dropout=.1
    # dim_feedforward=2048
    def __init__(self, embedding_size, num_heads, dim_feedforward, dropout):
        super().__init__()
        self.cross_atten = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_size // num_heads, dropout=dropout)
        
        self.linear1 = layers.Dense(dim_feedforward, activation='relu')
        self.linear2 = layers.Dense(embedding_size)

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
    

    def call(self, target, memory):
        # TODO: paper adds positional embedding to query and key?
        atten_output = self.cross_atten(query=target, key=memory, value=memory)

        # apply residual connection between original target and atten_output
        target = target + self.dropout1(atten_output)
        target = self.norm1(target)

        # feed forward
        ff_output = self.linear1(target)
        ff_output = self.dropout2(ff_output)
        ff_output = self.linear2(ff_output)

        # apply residual connection between target from before and ff output
        target = target + self.dropout3(ff_output)
        target = self.norm2(target)

        return target
    
# self attention custom class (called TransformerEncoderLayer in paper)
class SelfAtten(layers.Layer):
    # num_heads=8, d_model = embedding_size=768, dropout=.1
    # dim_feedforward=2048
    def __init__(self, embedding_size, num_heads, dim_feedforward, dropout):
        super().__init__()
        self.self_atten = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_size, dropout=dropout)
        
        self.linear1 = layers.Dense(dim_feedforward, activation='relu')
        self.linear2 = layers.Dense(embedding_size)

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
    

    def call(self, source):
        # TODO: paper adds positional embedding to query and key (source)
        atten_output = self.self_atten(query=source, key=source, value=source)

        # apply residual connection between original source and atten_output
        source = source + self.dropout1(atten_output)
        source = self.norm1(source)

        # feed forward
        ff_output = self.linear1(source)
        ff_output = self.dropout2(ff_output)
        ff_output = self.linear2(ff_output)

        # apply residual connection between target from before and ff output
        source = source + self.dropout3(ff_output)
        source = self.norm2(source)

        return source
    
# co-atten (= cross atten) custom class, called CrossAttention in paper
class CoAtten(layers.Layer):
    def __init__(self, feature_dim, dropout):
        super(CoAtten, self).__init__()
        self.q_linear = layers.Dense(feature_dim)
        self.k_linear = layers.Dense(feature_dim)
        self.v_linear = layers.Dense(feature_dim)
        self.dropout = layers.Dropout(dropout)

    def call(self, q, k, v):
        # TODO: they check that q/k/v shapes are 768 and apply linear layer to get 768 if not
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        # compute attention scores
        atten_scores = tf.matmul(q, k, transpose_b=True)
        atten_scores = atten_scores / tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))

        # turn scores into weights
        atten_weights = tf.nn.softmax(atten_scores, axis=-1)

        # compute atention values
        atten_vals = tf.matmul(atten_weights, v)
        atten_vals = self.dropout(atten_vals)

        return atten_vals