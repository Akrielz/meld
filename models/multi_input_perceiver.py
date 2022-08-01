from typing import List, Optional, Callable

import torch
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from perceiver_pytorch.perceiver_pytorch import PreNorm, Attention, FeedForward, fourier_encode
from pydantic_yaml import YamlModel
from torch import nn


class LatentMerger(nn.Module):
    """
    LatentMerger is a layer designed to merge multiple Latent Spaces that have
    the same size, into a single latent.

    In the bigger picture, this is needed for the PerceiverParallelInput class
    in order to merge together all the specific Latent Spaces that each input
    generates into a single latent.

    In more technical details:
    Input:
        latent_spaces: List[Tensor]:
            - Tensor shape: [batch_size, num_latents, latent_dim];
            - List shape: [num_inputs, batch_size, num_latents, latent_dim];

    Output:
        merged_latent: [Tensor]:
            - shape: [batch_size, num_latents, latent_dim]
    """

    net: nn.Module

    def __init__(self, latent_dim: int, num_inputs: int, dropout: float = 0.0, duplicate_latents: bool = True):
        """
        Initialize a LatentMerger making use of the following arguments:

        latent_dim:
            - describes the input of each latent

        num_inputs:
            - describes the number of inputs

        dropout:
            - describes the dropout chance inside the Dropout Layer

        duplicate_latents:
            - if enabled, the output dim will be the latent_dim
            - if disabled, the output dim will be the latent_dim * num_inputs
        """

        super(LatentMerger, self).__init__()

        out_dim = latent_dim if duplicate_latents else latent_dim * num_inputs

        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim * num_inputs),
            nn.Linear(latent_dim * num_inputs, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, latent_spaces: List[torch.Tensor]) -> torch.Tensor:
        """
        Concatenate the latent_spaces together, normalize them using a
        Normalization Layer and merge them using a Linear layer, followed with
        an eventual Dropout layer.
        """

        # [i, b, n, m] -> [b, n, m*i]
        latent_spaces = torch.cat(latent_spaces, dim=-1)

        # [b, n, m*i] -> [b, n, o]
        # where o is either m*i, either m
        return self.net(latent_spaces)


class MultiInputPerceiverParallel(nn.Module):
    """
    --- Description ---

    MultiInputPerceiverParallel is a Perceiver derived which allows multiple inputs
    instead of a single input.

    MultiInputPerceiverParallel creates an "Understanding Input Layer" for each input
    in order to gather the specifics about the input inside its own Latent
    Space. This is happening by using combining the initial merged latent, with
    each specific input.

    All the specific latents are then merged together in a single latent, in
    order to merge the information from the inputs.

    --- PerceiverParallelInput Architecture ---

    [ ## Layers ##
        [ ## Understanding Input layer ##
            CrossAttention(),
            FeedForward(),

            [ ## Latent Transformer ##
                SelfAttention()
                FeedForward()
            ] x self_per_cross_attn
        ] x inputs

        LatentMerger()
        FeedForward()

        [ ## Latent Transformer ##
            SelfAttention()
            FeedForward()
        ] x self_per_cross_attn
    ] x depth

    --- Notations ---

    "x something":
        - repeated "something" times
    "## name ##":
        - the name of the layer for easier reference

    --- Other ---

    More information can be found at:
    https://ordaos.atlassian.net/wiki/spaces/APO/pages/1274347521/Multi-Input+Perceiver
    """

    class Config(YamlModel):
        """
        Config class for MultiInputPerceiverParallel.

        # Input
        input_channels: List[int]
            - Number of channels for each token of the input.

        input_axis: List[int]
            - Number of axes for input data (2 for images, 3 for video)

        # Fourier Encoding
        fourier_encode_data: bool = True
            - Whether to auto-fourier encode the data, using the input_axis given
            - Defaults to True, but can be turned off if you are fourier encoding the data yourself

        num_freq_bands: int = 6
            - Number of freq bands, with original value (2 * K + 1)

        max_freq: int = 10
            - Maximum frequency, hyperparameter depending on how fine the data is.

        # Architecture
        depth: int = 3
            - Depth of network.

        # Latents
        num_latents: int = 256
            - Number of latents, or induced set points, or centroids.
            - Different papers giving it different names.

        latent_dim: int = 512
            - Latent dimension.

        duplicate_latents: bool = True
            - If enabled, the same latent is used for all inputs
            - If enabled, the latent merger will return a single latent
            - If disabled, all the inputs are using a unique latent
            - If disabled, the latent merger will return all the latents

        # Cross Attention
        cross_heads: int = 2
            - Number of heads for Cross Attention.

        cross_dim_head: int = 64
            - Number of dimensions per Cross Attention head.

        # Self Attention
        self_per_cross_attn: int = 1
            - Number of Self Attention per Latent Transformer
            - This increases the number of Feed Forwards per Self Attention

        latent_heads: int = 8
            - Number of heads for latent Self Attention.

        latent_dim_head: int = 64,
            - Number of dimensions per latent Self Attention head.

        # Dropouts
        attn_dropout: float = 0.0
            - Attention dropout

        ff_dropout: float = 0.0
            - Feed Forward dropout

        merger_dropout: float = 0.0
            - Latent Merger dropout

        # Output
        num_classes: int = 1000
            - Number of classes / logits

        final_classifier_head: bool = True
            - Mean pool and project embeddings to number of classes (num_classes) at the end.
            - Can be turned off to return the latents
        """

        # Information about input
        input_channels: List[int] = []
        input_axis: List[int] = []

        # Information about latent space
        latent_dim: int = 512
        num_latents: int = 256
        duplicate_latents: bool = True

        # Information about the Architecture's depth
        depth: int = 3

        # Information about Cross Attentions
        cross_heads: int = 2
        cross_dim_head: int = 64

        # Information about Self Attentions
        self_per_cross_attn: int = 1
        latent_heads: int = 8
        latent_dim_head: int = 64

        # Information about Dropouts
        attn_dropout: float = 0.0
        ff_dropout: float = 0.0
        merger_dropout: float = 0.0

        # Information about Fourier Encoding
        max_freq: int = 10
        num_freq_bands: int = 6
        fourier_encode_data: bool = True

        # Information about Output
        num_classes: int = 1000
        final_classifier_head: bool = True

    def __init__(
            self,
            config: Config
    ):
        """
        Process the given information to build the model layers accordingly.
        """

        nn.Module.__init__(self)

        # Check the given config information
        assert len(self.config.input_channels) != 0, "The input_channels list was not provided"
        assert len(self.config.input_axis) != 0, "The input_channels list was not provided"

        assert len(self.config.input_channels) == len(self.config.input_axis), \
            "The lengths of input_channels and input_axis differ"

        # Save relevant data
        self.num_inputs = len(self.config.input_channels)

        # Calculate the fourier_channels dimensions required for each input
        fourier_channels = [
            axis * ((self.config.num_freq_bands * 2) + 1) if self.config.fourier_encode_data else 0
            for axis in self.config.input_axis
        ]

        # Calculate the input dimensions, considering the possibility of having
        # fourier channels attached
        self.input_dims = [
            fourier_channel + input_channel
            for fourier_channel, input_channel in zip(fourier_channels, self.config.input_channels)
        ]

        # Initialize the latents
        self.merged_latent_dim = self.config.latent_dim if self.config.duplicate_latents else \
            self.config.latent_dim * self.num_inputs

        self.latents = nn.Parameter(torch.randn(self.config.num_latents, self.merged_latent_dim))
        # Build the layers
        self.__build()

    def __build(self):
        """
        Build the inner layers, according to the architecture presented.
        """
        get_merged_feed_forward, get_understanding_input_layers, get_latent_merger, get_merged_latent_transformer = \
            self.__define_building_components()

        # Create the inner layers used inside the architecture
        self.layers = nn.ModuleList([])
        for i in range(self.config.depth):
            self.layers.append(
                nn.ModuleList([
                    get_understanding_input_layers(),
                    get_latent_merger(),
                    get_merged_feed_forward(),
                    get_merged_latent_transformer(),
                ])
            )

        # Append a form of "TaskHead" to get the logits, if needed
        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(self.merged_latent_dim),
            nn.Linear(self.merged_latent_dim, self.config.num_classes)
        ) if self.config.final_classifier_head else nn.Identity()

    def __define_building_components(self) -> List[Callable[[], nn.Module]]:
        """
        Create functions to automatically create the necessary building blocks
        from the architecture, without the need to pass many parameters each
        time.
        """

        def get_cross_attention_generic(input_dim: int) -> nn.Module:
            """
            Create Cross Attention layer, combining an input of specific
            input_dim with the latent.
            Apply a Pre Normalization on both the query and the context.
            """
            return PreNorm(
                dim=self.config.latent_dim,
                fn=Attention(
                    query_dim=self.config.latent_dim,
                    context_dim=input_dim,
                    heads=self.config.cross_heads,
                    dim_head=self.config.cross_dim_head,
                    dropout=self.config.attn_dropout
                ),
                context_dim=input_dim
            )

        def get_single_self_attention() -> nn.Module:
            """
            Create Self Attention layer for a single latent space.
            Apply a Pre Normalization on latent.
            """
            return PreNorm(
                dim=self.config.latent_dim,
                fn=Attention(
                    query_dim=self.config.latent_dim,
                    heads=self.config.latent_heads,
                    dim_head=self.config.latent_dim_head,
                    dropout=self.config.attn_dropout
                )
            )

        def get_merged_self_attention() -> nn.Module:
            """
            Create Self Attention layer for a single latent space.
            Apply a Pre Normalization on latent.
            """
            return PreNorm(
                dim=self.merged_latent_dim,
                fn=Attention(
                    query_dim=self.merged_latent_dim,
                    heads=self.config.latent_heads,
                    dim_head=self.config.latent_dim_head,
                    dropout=self.config.attn_dropout
                )
            )

        def get_single_feed_forward() -> nn.Module:
            """
            Create a Feed-Forward layer for a single latent space.
            Apply a Pre Normalization on latent.
            """
            return PreNorm(
                dim=self.config.latent_dim,
                fn=FeedForward(
                    dim=self.config.latent_dim,
                    dropout=self.config.ff_dropout
                )
            )

        def get_merged_feed_forward() -> nn.Module:
            """
            Create a Feed-Forward layer for the merged latent space.
            Apply a Pre Normalization on latent.
            """
            return PreNorm(
                dim=self.merged_latent_dim,
                fn=FeedForward(
                    dim=self.merged_latent_dim,
                    dropout=self.config.ff_dropout
                )
            )

        def get_latent_merger() -> nn.Module:
            """
            Create a Latent Merger layer to be able to merge the latent space
            from all the inputs.
            """
            return LatentMerger(
                latent_dim=self.config.latent_dim,
                num_inputs=self.num_inputs,
                dropout=self.config.merger_dropout,
                duplicate_latents=self.config.duplicate_latents
            )

        def get_single_latent_transformer() -> nn.Module:
            """
            Create a Latent Transformer to refine the content of a given latent.
            """
            self_attention_layers = nn.ModuleList([])
            for self_attention_index in range(self.config.self_per_cross_attn):
                self_attention_layers.append(
                    nn.ModuleList([
                        get_single_self_attention(),
                        get_single_feed_forward()
                    ])
                )
            return self_attention_layers

        def get_merged_latent_transformer() -> nn.Module:
            """
            Create a Latent Transformer to refine the content of a given latent.
            """
            self_attention_layers = nn.ModuleList([])
            for self_attention_index in range(self.config.self_per_cross_attn):
                self_attention_layers.append(
                    nn.ModuleList([
                        get_merged_self_attention(),
                        get_merged_feed_forward()
                    ])
                )
            return self_attention_layers

        def get_understanding_input_layers() -> nn.Module:
            """
            Create the UnderStanding Input Layers for each available input.
            """
            understanding_input_layers = nn.ModuleList([])
            for input_dim in self.input_dims:
                understanding_input_layers.append(
                    nn.ModuleList([
                        get_cross_attention_generic(input_dim=input_dim),
                        get_single_feed_forward(),
                        get_single_latent_transformer(),
                    ])
                )
            return understanding_input_layers

        # Return only the building components needed for the __build__().
        # The rest of components are already included inside these big
        # components.
        return [get_merged_feed_forward, get_understanding_input_layers, get_latent_merger, get_merged_latent_transformer]

    def forward(
            self,
            data_list: List[torch.Tensor],
            mask_list: Optional[List[torch.Tensor]] = None,
            return_embeddings: bool = False
    ) -> torch.Tensor:
        assert len(data_list) == self.num_inputs, "A different number of inputs was given"

        if mask_list is None:
            mask_list = [None] * len(data_list)

        assert len(data_list) == len(mask_list), "A different number of inputs and masks were given"

        batch_size = data_list[0].shape[0]
        device = data_list[0].device

        # Gather axis info, while checking if the inputs are sent properly
        axis_list = []
        for i, (data, config_axis, input_channels) in \
                enumerate(zip(data_list, self.config.input_axis, self.config.input_channels)):
            batch_size_cur, *axis, dim = data.shape
            assert dim == input_channels, \
                f'The dimension of the data_list[{i}] differ from the one specified in __init__()'
            assert len(axis) == config_axis, \
                f'The axis of the given data_list[{i}] differ from the one specified in __init__()'
            assert batch_size_cur == batch_size, \
                'Inputs have different batch_sizes'

            axis_list.append(axis)

        # Apply Fourier Encoding for each input data, and append it inside the original data
        for i, axis in enumerate(axis_list):
            if self.config.fourier_encode_data:
                dtype = data_list[i].dtype
                axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
                pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
                enc_pos = fourier_encode(pos, self.config.max_freq, self.config.num_freq_bands)
                enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
                enc_pos = repeat(enc_pos, '... -> b ...', b=batch_size)

                data_list[i] = torch.cat((data_list[i], enc_pos), dim=-1)

            data_list[i] = rearrange(data_list[i], 'b ... d -> b (...) d')

        # Create the latents for the give batch size
        latents_merged = repeat(self.latents, '... -> b ...', b=batch_size)

        # Apply the layers
        for understanding_input_layer, latent_merger, feed_froward, latent_transformer in self.layers:

            # Split the merged latent into inputs
            if self.config.duplicate_latents:
                latents_merged = repeat(latents_merged, '... -> i ...', i=self.num_inputs)
            else:
                latents_merged = rearrange(latents_merged, '... (d i) -> i ... d', i=self.num_inputs)

            # Prepare latent list for each input
            latents_list = []

            # Apply the Understanding Input layer
            for i, (cross_attention, cross_feed_forward, latent_input_transformer) in enumerate(understanding_input_layer):

                # Apply Cross Attention for each input
                latents_input = \
                    cross_attention(x=latents_merged[i], context=data_list[i], mask=mask_list[i]) + latents_merged[i]
                latents_input = cross_feed_forward(latents_input) + latents_input

                # Apply Latent Transformer for each latent from each input
                for self_attention, self_feed_forward in latent_input_transformer:
                    latents_input = self_attention(latents_input) + latents_input
                    latents_input = self_feed_forward(latents_input) + latents_input

                # Append the latents of each input to a single list
                latents_list.append(latents_input)

            # Merge all the latents back to a single latent
            latents_merged = latent_merger(latents_list)
            latents_merged = feed_froward(latents_merged)

            # Apply a Latent Transformer for the merged latents
            for self_attention, self_feed_forward in latent_transformer:
                latents_merged = self_attention(latents_merged) + latents_merged
                latents_merged = self_feed_forward(latents_merged) + latents_merged

        # Return the latents if required
        if return_embeddings:
            return latents_merged

        # Return the classes / logits
        return self.to_logits(latents_merged)
