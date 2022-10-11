import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """**PositionalEmbedding layer**"""

    def __init__(self, sequence_length, vocab_size, embed_dim, name=None, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.name = name

        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings=sequence_length, embedding_dim=embed_dim
        )

    def forward(self, inputs):
        embedded_tokens = self.token_embeddings(inputs)
        positions = torch.arange(start=0, end=inputs.shape[1], step=1)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions


class PatchEmbedding(nn.Module):
    """Positional Embedding for images"""

    def __init__(self, c_in, projection_dim, num_patches, name=None, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.name = name
        self.position_embedding = nn.Embedding(
            num_embeddings=num_patches, embedding_dim=projection_dim
        )
        self.projection = nn.Linear(c_in, projection_dim)

    def forward(self, patch):
        positions = torch.arange(start=0, end=self.num_patches, step=1)
        # (num_patches, projection_dim)
        embedded_positions = self.position_embedding(positions)
        embeded_patch = self.projection(patch)
        # (batch_size, num_patches, projection_dim)
        embedded_projection = embeded_patch
        return embedded_projection + embedded_positions


if __name__ == "__main__":
    print("Testing PatchEmbedding")
    a = torch.randn(1, 6, 4)
    patch_embedding = PatchEmbedding(
        c_in=a.shape[-1], num_patches=a.shape[-2], projection_dim=12
    )
    results = patch_embedding(a)
    print(results.shape)

    print("Testing PositionalEmbedding")
    a = torch.randint(low=0, high=100, size=(1, 6))
    pos_embedding = PositionalEmbedding(
        sequence_length=a.shape[-1], vocab_size=100, embed_dim=8
    )
    results = pos_embedding(a)
    print(results.shape)
