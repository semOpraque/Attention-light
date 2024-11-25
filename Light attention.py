import numpy as np

class AttentionLeve:
    def __init__(self, temperature=1.0, embed_dim=0, batch_size=0):
        # I started with the initialization of the key parameters. 
        # The temperature is important because it will help control the smoothing of the attention.
        self.temperature = temperature  # The temperature will adjust the attention scale; the higher the temperature, the smoother the attention.

        # I created the embedding matrix with random values. This matrix represents the features of the data 
        # (like words in translation or in a language model).
        # Each row corresponds to an item in the batch, and each column represents a dimension of the embedding.
        self.embed_dim = np.random.rand(batch_size, embed_dim)  # Initializes random embeddings with the defined dimensions

        # I stored the batch size here so I could use it in other calculations like Q and K.
        self.batch_size = batch_size

        # I added specific weights for the queries (Q), keys (K), and for the attention itself.
        # These weights will be multiplied by the embeddings to control how each dimension influences the final attention.
        self.XWq = np.random.rand(embed_dim)  # Weight for the query (Q)
        self.XWk = np.random.rand(embed_dim)  # Weight for the key (K)
        self.weight = np.random.rand(embed_dim)  # General weight applied to the attention

        # Here, I initialized the Q and K variables as None. They will be calculated later.
        self.Q = None
        self.K = None
        self.attention = None

        # Finally, I called the function to calculate the importance of the embeddings.
        # The importance will help us adjust how each embedding contributes to the final attention.
        self.importance = self.calculate_importance()

    def calculate_importance(self):
        """
        I created this function to calculate the importance of each value within the embeddings. 
        I decided to calculate importance based on the difference between consecutive embeddings.
        This helps highlight how "important" an embedding is relative to the previous one.
        """
        # I created an importance matrix with the same size as the embedding matrix.
        importance = np.zeros_like(self.embed_dim)  # Start with a matrix of zeros

        # The importance of the first embedding is simply the absolute value of its components.
        # This means that the first embedding is treated in a straightforward manner.
        importance[0, :] = np.abs(self.embed_dim[0, :])  # Initial importance is the absolute value of the first embedding
        
        # For the other embeddings, I calculated the absolute difference between the current and previous values.
        # This allows me to capture the change between consecutive embeddings, reflecting their "relative importance".
        for i in range(1, self.batch_size):
            importance[i, :] = np.abs(self.embed_dim[i, :] - self.embed_dim[i-1, :])

        return importance

    def Q_K(self):
        """
        I created this function to calculate the Q and K matrices. 
        They are weighted combinations of the embeddings, with the importance of each value within each embedding being considered.
        This step is crucial for the attention operation because queries and keys are used to compute the relevance between embeddings.
        """
        # I initialized the Q and K matrices with the same shape as the embeddings, because they will have the same number of rows and columns.
        self.Q = np.zeros_like(self.embed_dim)
        self.K = np.zeros_like(self.embed_dim)

        # For each item in the batch, I calculated Q and K based on the product of the embedding, weight, and importance.
        # This multiplication results in a transformation of the embeddings, taking into account their weight and importance.
        for i in range(self.batch_size):
            for j in range(self.embed_dim.shape[1]):  # For each dimension of the embedding
                # For each value of Q and K, I multiply the embedding, its corresponding weight, and importance
                self.Q[i, j] = self.embed_dim[i, j] * self.XWq[j] * self.importance[i, j]
                self.K[i, j] = self.embed_dim[i, j] * self.XWk[j] * self.importance[i, j]
        
        return self.Q, self.K

    def cal(self):
        """
        The `cal` function is where I calculate the final attention. The attention depends on Q, K, weight, and importance.
        I chose this formula because it combines the values of Q and K, applies the weight, 
        and adjusts with importance and temperature, which allows us to control the smoothing of the attention.
        """
        # The final formula for attention is:
        # ATTENTION = ((Q * K) * WEIGHT) * (IMPORTANCE / TEMPERATURE)
        # I do this to adjust the intensity of the attention as needed.
        self.attention = ((self.Q * self.K) * self.weight) * (self.importance / self.temperature)
        return self.attention

# I create a model with 10 embedding dimensions and a batch size of 6
model = AttentionLeve(temperature=1.0, embed_dim=10, batch_size=6)

# Calculate Q and K with the Q_K function
model.Q_K()

# Display the embeddings before applying the weights
print("Embedding:")
print(model.embed_dim)

# Display the calculated Q and K values
print("Q:")
print(model.Q)
print("K:")
print(model.K)

# Display the importance of values within the embeddings (based on the difference between consecutive embeddings)
print("Importance of Values in the Embeddings (based on the difference with the previous embedding):")
print(model.importance)

# Calculate and display the attention (attention)
model.cal()
print("Calculated Attention:")
print(model.attention)
