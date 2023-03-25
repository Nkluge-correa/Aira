import tensorflow as tf
from tensorflow import keras
from keras import layers


class TransformerEncoder(layers.Layer):
    """
    The TransformerEncoder class is a custom Keras layer that implements a 
    single transformer encoder block. The transformer encoder block consists 
    of a multi-head self-attention layer followed by a feedforward neural 
    network with a residual connection and layer normalization applied at 
    the input and output of each sub-layer. 

    The class takes in the following arguments:

        embed_dim: an integer specifying the dimensionality of the embedding space.
        dense_dim: an integer specifying the number of units in the feedforward neural network.
        num_heads: an integer specifying the number of attention heads to use.

    The call method is the main computation performed by the layer. It takes 
    in an input tensor and an optional mask tensor indicating which inputs to 
    consider in the attention calculation. It returns the output tensor of the 
    transformer encoder block.

    The get_config method returns a dictionary of configuration information for 
    the layer, including the embed_dim, num_heads, and dense_dim parameters.
    """

    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config


class PositionalEmbedding(layers.Layer):
    """
    The PositionalEmbedding layer class is used to create an embedding layer that 
    combines both token embeddings and positional embeddings for input sequences. 

    The class takes in the following arguments:

    sequence_length: An integer representing the maximum length of the input sequence.
    input_dim: An integer representing the size of the input vocabulary.
    output_dim: An integer representing the size of the embedding vectors.

    The call(self, inputs) method that takes input tensor as an argument and 
    returns the embedded tensor after adding the token embeddings and positional 
    embeddings. It also computes the positions for the input sequence.

    The compute_mask(self, inputs, mask=None) method that returns a mask tensor 
    computed based on the input tensor.

    The get_config(self): Method that returns a dictionary containing the configuration 
    of the layer.
    """

    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config
