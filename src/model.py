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

        # CLIP model both image and text processing
        self.clip_model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
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
        self.bert_linear = tf.keras.Sequential([
            layers.Dense(self.embedding_size),
            layers.ReLU(),
            layers.LayerNormalization(axis=-1) # TODO: check axis=-1
        ])
        
        # SFIM (SHALLOW FEATURE INTERACTION MODULE)
        # self.sfim_imgtxt_cross = layers.MultiHeadAttention(num_heads=8, key_dim=64)  # d_model = 64 * 8 = 512
        # TODO: implement paper's TransformerCrossLayer starting with nn.MultiHeadAttention
        
        return
    
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
        text_feature = clip_output['text_model_output']['pooler_output'] # 32, 512
        image_feature = clip_output['vision_model_output']['pooler_output'] # 32, 768
        text_feature = self.text_linear(text_feature)  # 32, 768
        image_feature = self.image_linear(image_feature)  # 32, 768

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