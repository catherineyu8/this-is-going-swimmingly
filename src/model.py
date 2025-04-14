import tensorflow as tf
import tensorflow.keras.layers as layers
from transformers import TFCLIPModel, BertConfig, TFBertModel, BertTokenizer
# from transformers.models.bert.modeling_bert import BertLayer


class RackleMuffin(tf.keras.Model):

    def __init__(self):
        super(RackleMuffin, self).__init__()

        self.num_classes = 2
        self.text_hidden_size = 512
        self.image_hidden_size = 768 # self.config.hidden_size
        self.num_attn_heads = 8
        self.adam_epsilon = 1e-8
        self.learning_rate = 5e-4
        self.clip_learning_rate = 1e-6
        self.max_len_clip_text = 77
        self.dropout_rate = 0.1

        # define clip model and bert model for image and text processing
        self.clip_model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.bert_config = BertConfig.from_pretrained("bert-base-uncased")
        self.bert_model = TFBertModel.from_pretrained("bert-base-uncased")
        # self.resnet_backbone = # join resnet with position encoding

        # text and image linear layers used for initial feature extraction
        self.text_linear = tf.keras.Sequential([
            tf.keras.layers.Dense(self.image_hidden_size, input_shape=(self.text_hidden_size,)),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Activation(tf.keras.activations.gelu)  # or tf.keras.layers.GELU() in newer TF versions
        ])
        self.image_linear = tf.keras.Sequential([
            tf.keras.layers.Dense(self.image_hidden_size, input_shape=(self.image_hidden_size,)),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Activation(tf.keras.activations.gelu)  # or tf.keras.layers.GELU() in newer TF versions
        ])
        
        return
    
    def call(self, inputs):
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
        text_feature = clip_output['text_model_output']['pooler_output'] # 64，512
        image_feature = clip_output['vision_model_output']['pooler_output'] # 64，768
        text_feature = self.text_linear(text_feature)  # 64，768
        image_feature = self.image_linear(image_feature)  # 64,768