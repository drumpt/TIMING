import math
import abc

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchModel(nn.Module, abc.ABC):
    """
    Class extends torch.nn.Module. Mainly for user to specify the forward with a ``return_all``
    option. The model is supposed to accept inputs of shape (num_samples, num_features, num_times).
    If return_all is True, the output should be of shape (num_samples, num_states, num_times).
    Otherwise, the output should be of shape (num_samples, num_states)
    """

    def __init__(self, feature_size, num_states, hidden_size, device):
        """
        Constructor

        Args:
            feature_size:
               The number of features the model is accepting.
            num_states:
               The number of output nodes.
            hidden_size:
               The hidden size of the model
            device:
               The torch device the model is on.
        """
        super().__init__()
        self.feature_size = feature_size
        self.num_states = num_states
        self.hidden_size = hidden_size
        if self.num_states > 1:
            activation = torch.nn.Softmax(dim=1)
        else:
            activation = torch.nn.Sigmoid()
        self.activation = activation
        self.device = device

    @abc.abstractmethod
    def forward(self, input, mask=None, timesteps=None, return_all=True):
        """
        Specify the forward function for this torch.nn.Module. The forward function should not
        include the activation function at the end. i.e. the output should be in logit space.

        Args:
            input:
                Shape = (num_samples, num_features, num_times)
            return_all:
                True if we want to get the output of the model only at the last timestep.

        Returns:
            A tensor of shape (num_samples, num_states, num_times) if return_all is True. Otherwise,
            a tensor of shape (num_samples, num_states) is returned.
        """

    def predict(self, input, mask=None, timesteps=None, return_all=True):
        """
        Apply the activation after the forward function.

            input:
                Shape = (num_samples, num_features, num_times)
            return_all:
                True if we want to get the output of the model only at the last timestep.

        Returns:
            A tensor of shape (num_samples, num_states, num_times) if return_all is True. Otherwise,
            a tensor of shape (num_samples, num_states) is returned.
        """
        return self.activation(self.forward(input, mask, timesteps, return_all))


class ConvClassifier(TorchModel):
    def __init__(self, feature_size, num_states, hidden_size, kernel_size, device):
        super().__init__(feature_size, num_states, hidden_size, device)

        if kernel_size % 2 != 0:
            raise Exception("Odd kernel size")
        padding = kernel_size // 2

        # Input to torch Conv
        self.regressor = nn.Sequential(
            torch.nn.Conv1d(
                in_channels=feature_size,
                out_channels=self.hidden_size,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="replicate",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="replicate",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="replicate",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.num_states,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="replicate",
            ),
        )

    def forward(self, input, mask=None, timesteps=None, return_all=True):
        """
        Specify the forward function for this torch.nn.Module. The forward function should not
        include the activation function at the end. i.e. the output should be in logit space.

        Args:
            input:
                Shape = (num_samples, num_features, num_times)
            return_all:
                True if we want to get the output of the model only at the last timestep.

        Returns:
            A tensor of shape (num_samples, num_states, num_times) if return_all is True. Otherwise,
            a tensor of shape (num_samples, num_states) is returned.
        """
        num_samples, _, num_times = input.shape
        self.regressor.to(self.device)
        input = input.to(self.device)
        regressor = self.regressor(input)
        if return_all:
            return regressor[:, :, -num_times:]  # (num_samples, nstate, num_times)
        else:
            return regressor[:, :, -1]  # (num_samples, nstate)


class StateClassifier(TorchModel):
    def __init__(
        self,
        feature_size,
        num_states,
        hidden_size,
        device,
        rnn="GRU",
        num_layers=None,
        dropout=0.5,
    ):
        super().__init__(feature_size, num_states, hidden_size, device)
        self.rnn_type = "GRU" if rnn is None else rnn
        self.num_layers = 1 if num_layers is None else num_layers

        # Input to torch LSTM should be of size (num_times, num_samples, num_features)
        self.regres_in_size = self.hidden_size

        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                feature_size, self.hidden_size, num_layers=self.num_layers
            ).to(self.device)
        else:
            self.rnn = nn.LSTM(
                feature_size, self.hidden_size, num_layers=self.num_layers
            ).to(self.device)

        self.regressor = nn.Sequential(
            nn.BatchNorm1d(num_features=self.regres_in_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.regres_in_size, self.num_states),
        )

    def forward(
        self, input, mask=None, timesteps=None, return_all=False, past_state=None
    ):
        """
        Specify the forward function for this torch.nn.Module. The forward function should not
        include the activation function at the end. i.e. the output should be in logit space.

        Args:
            input:
                Shape = (num_samples, num_features, num_times)
            return_all:
                True if we want to get the output of the model only at the last timestep.
            past_state:
                The past state for the rnn. Shape = (num_layers, num_samples, hidden_size).
                Default to be None which will set this to zero.

        Returns:
            A tensor of shape (num_samples, num_states, num_times) if return_all is True. Otherwise,
            a tensor of shape (num_samples, num_states) is returned.
        """

        num_samples, num_features, num_times = input.shape
        input = input.permute(2, 0, 1).to(self.device)
        # input.shape (num_times, num_samples, num_features)
        self.rnn.to(self.device)
        self.regressor.to(self.device)
        if past_state is None:
            #  Size of hidden states: (num_layers, num_samples, hidden_size)
            past_state = torch.zeros(
                [self.num_layers, input.shape[1], self.hidden_size]
            ).to(self.device)
        if self.rnn_type == "GRU":
            all_encodings, encoding = self.rnn(input, past_state)
            # all_encodings.shape = (num_times, num_samples, num_hidden)
            # encoding.shape = (num_layers, num_samples, num_hidden)
        else:
            all_encodings, (encoding, state) = self.rnn(input, (past_state, past_state))

        if return_all:
            reshaped_encodings = all_encodings.view(
                all_encodings.shape[1] * all_encodings.shape[0], -1
            )
            # shape = (num_times * num_samples, num_hidden)
            return torch.t(
                self.regressor(reshaped_encodings).view(all_encodings.shape[0], -1)
            ).reshape(num_samples, self.num_states, num_times)
            # (num_samples, n_state, num_times)
        else:
            encoding = encoding[-1, :, :]  # encoding.shape (num_samples, num_hidden)
            return self.regressor(encoding)  # (num_samples, num_states)


class multiTimeAttention(nn.Module):
    def __init__(self, feature_size, nhidden=16, embed_time=16, num_heads=1):
        super().__init__()

        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = feature_size
        self.nhidden = nhidden

        # Query and Key transformations with LayerNorm
        self.linears = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_time, embed_time), nn.LayerNorm(embed_time)
                ),
                nn.Sequential(
                    nn.Linear(embed_time, embed_time), nn.LayerNorm(embed_time)
                ),
                nn.Sequential(
                    nn.Linear(feature_size * num_heads, nhidden), nn.LayerNorm(nhidden)
                ),
            ]
        )

        # Add LayerNorm for attention outputs
        self.attention_norm = nn.LayerNorm(feature_size * num_heads)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)

        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = dropout(p_attn)

        attended_values = torch.sum(p_attn * value.unsqueeze(-3), -2)
        return attended_values, p_attn

    def forward(self, query, key, value, mask=None, dropout=None):
        batch, seq_len, dim = value.size()
        if mask is not None:
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)

        # Transform and normalize query and key
        query, key = [
            l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
            for l, x in zip(self.linears[:2], (query, key))
        ]

        # Apply attention
        x, _ = self.attention(query, key, value, mask, dropout)

        # Reshape and normalize attention output
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        x = self.attention_norm(x)

        # Final linear transformation and normalization
        return self.linears[-1](x)


class mTAND(TorchModel):
    def __init__(
        self,
        feature_size,
        num_timesteps,
        num_states,
        device,
        nhidden=128,  # Increased hidden size
        embed_time=128,  # Increased embedding size
        num_heads=1,  # Increased number of heads
        freq=10,
    ):
        super().__init__(feature_size, num_states, nhidden, device)

        self.feature_size = feature_size
        self.num_timesteps = num_timesteps
        self.output_dim = num_states
        self.device = device

        self.freq = freq
        self.embed_time = embed_time
        self.nhidden = nhidden

        # Enhanced attention with layer norm and dropout
        self.query = torch.linspace(0, 1, self.num_timesteps).to(device)
        self.att = multiTimeAttention(
            2 * self.feature_size, nhidden, embed_time, num_heads
        )
        self.enc = nn.GRU(nhidden, nhidden)
        self.classifier = nn.Sequential(
            nn.Linear(nhidden, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, self.output_dim),
        )
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, input, mask, timesteps=None, return_all=False):
        input = input.permute(0, 2, 1)  # B x F x T -> B x T x F
        mask = 1 - mask.permute(0, 2, 1)  # B x F x T -> B x T x F
        if timesteps is None:
            timesteps = (
                torch.linspace(0, 1, input.shape[1])
                .unsqueeze(0)
                .repeat(input.size(0), 1)
                .to(input.device)
            )
        timesteps = timesteps.unsqueeze(-1)

        input = input * (mask > 0).float()  # Zeroize masked values

        x = torch.cat((input, mask), 2)
        mask = x[:, :, self.feature_size :]
        mask = torch.cat((mask, mask), 2)

        key = self.learn_time_embedding(timesteps)
        query = self.learn_time_embedding(self.query.unsqueeze(0))

        out = self.att(query, key, x, mask)
        out = out.permute(1, 0, 2)
        if return_all:
            out, _ = self.enc(out)
            out = out.permute(1, 0, 2)
        else:
            _, out = self.enc(out)
            out = out.squeeze(0)
        out = self.classifier(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, max_time=20000, n_dim=10):
        super().__init__()
        self.max_time = max_time
        self.n_dim = n_dim
        self._num_timescales = self.n_dim // 2

        # Initialize timescales
        timescales = self.max_time ** torch.linspace(0, 1, self._num_timescales)
        self.register_buffer("timescales", timescales)

    def forward(self, times):
        scaled_time = times / self.timescales[None, None, :]
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return signal


class SetAttentionLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        n_layers=2,
        width=128,
        latent_width=128,
        dot_prod_dim=64,
        n_heads=4,
        attn_dropout=0.3,
    ):
        super().__init__()
        self.width = width
        self.dot_prod_dim = dot_prod_dim
        self.attn_dropout = attn_dropout
        self.n_heads = n_heads

        # Build psi network
        psi_layers = []
        curr_width = input_dim
        for _ in range(n_layers):
            psi_layers.extend(
                [nn.Linear(curr_width, width), nn.ReLU(), nn.LayerNorm(width)]
            )
            curr_width = width
        psi_layers.extend(
            [nn.Linear(width, latent_width), nn.ReLU(), nn.LayerNorm(latent_width)]
        )
        self.psi = nn.Sequential(*psi_layers)

        # rho network
        self.rho = nn.Sequential(
            nn.Linear(latent_width, latent_width), nn.ReLU(), nn.LayerNorm(latent_width)
        )

        # Initialize attention weights
        self.W_k = nn.Parameter(
            torch.empty(latent_width + input_dim, dot_prod_dim * n_heads)
        )
        nn.init.kaiming_uniform_(self.W_k, mode="fan_in", nonlinearity="relu")
        self.W_q = nn.Parameter(torch.zeros(n_heads, dot_prod_dim))

    def forward(self, inputs, mask):
        batch_size, seq_len, _ = inputs.size()

        # Keep original inputs for later
        raw_inputs = inputs

        # Encode inputs through psi network
        encoded = self.psi(inputs)  # [B, T*F, latent_width]

        # Masked mean aggregation (with safe operations)
        mask_f = mask.float().unsqueeze(-1)  # Convert to float for safe multiplication
        masked_encoded = encoded * mask_f
        mask_sum = mask_f.sum(dim=1) + 1e-8  # Add epsilon for numerical stability
        agg = masked_encoded.sum(dim=1) / mask_sum

        # Transform aggregated features
        agg = self.rho(agg)  # [B, latent_width]

        # Expand back to match sequence length
        agg_scattered = agg.unsqueeze(1).expand(-1, seq_len, -1)

        # Combine with raw inputs
        combined = torch.cat([raw_inputs, agg_scattered], dim=-1)

        # Compute keys
        keys = torch.matmul(combined, self.W_k)
        keys = keys.view(batch_size, seq_len, self.n_heads, self.dot_prod_dim)
        keys = keys.permute(0, 2, 1, 3)  # [B, h, T*F, d]

        # Prepare queries
        queries = self.W_q.unsqueeze(0).unsqueeze(-1)  # [1, h, d, 1]

        # Compute attention scores with numerical stability
        scale = math.sqrt(self.dot_prod_dim)
        scores = torch.matmul(keys, queries) / scale  # [B, h, T*F, 1]
        scores = scores.squeeze(-1)  # [B, h, T*F]

        # Apply mask
        mask_bool = mask.bool().unsqueeze(1)
        scores = scores.masked_fill(~mask_bool, -1e9)  # Use finite value instead of inf

        # Apply dropout during training
        if self.training and self.attn_dropout > 0:
            dropout_mask = torch.bernoulli(
                torch.full_like(scores, 1 - self.attn_dropout)
            ).bool()
            scores = scores.masked_fill(~dropout_mask, -1e9)

        # Compute attention weights with safe softmax
        max_scores = torch.max(scores, dim=2, keepdim=True)[0]  # [B, h, 1]
        exp_scores = torch.exp(scores - max_scores)  # [B, h, T*F]
        exp_scores = exp_scores * mask.float().unsqueeze(1)  # [B, h, T*F]
        attn_weights = exp_scores / (
            exp_scores.sum(dim=2, keepdim=True) + 1e-8
        )  # [B, h, T*F]
        # No need to return as list
        return attn_weights, encoded


class SeFT(TorchModel):
    def __init__(self, feature_size, num_timesteps, num_states, device, nhid=128):
        super().__init__(feature_size, num_states, nhid, device)

        self.feature_size = feature_size
        self.num_timesteps = num_timesteps
        self.output_dim = num_states
        self.device = device
        self.nhid = nhid

        # Fixed parameters
        self.pos_dim = 10
        self.num_heads = 4
        self.dot_prod_dim = 64

        # Time encoding
        self.pos_encoder = PositionalEncoding(n_dim=self.pos_dim)

        # Register modality indices buffer
        modality_indices = torch.arange(feature_size)
        self.register_buffer("modality_indices", modality_indices)

        # Calculate input dimension
        self.input_dim = self.pos_dim + 1 + self.feature_size  # time + value + modality

        # Set attention layer
        self.attention = SetAttentionLayer(
            input_dim=self.input_dim,
            n_layers=2,
            width=nhid,
            latent_width=nhid,
            dot_prod_dim=self.dot_prod_dim,
            n_heads=self.num_heads,
        )

        self.output_net = nn.Sequential(
            nn.Linear(nhid * self.num_heads, nhid),
            nn.ReLU(),
            nn.LayerNorm(nhid),
            nn.Linear(nhid, self.output_dim),
        )

    def forward(self, input, mask, timesteps=None, return_all=False):
        input = input.permute(0, 2, 1)  # B x T x F
        mask = 1 - mask.permute(0, 2, 1)  # B x T x F

        batch_size, seq_len, _ = input.shape
        device = input.device

        # Create time values and reshape input/mask
        values = input  # [B, T, F]
        masks = mask  # [B, T, F]

        if timesteps is None:
            times = torch.linspace(0, 1, seq_len, device=device)  # [T]
            times = times.unsqueeze(0).repeat(batch_size, 1)  # [B, T]
        else:
            times = timesteps

        # Reshape values and masks
        values_reshaped = values.reshape(batch_size, -1)  # [B, T*F]
        masks_reshaped = masks.reshape(batch_size, -1)  # [B, T*F]

        # Now repeat times for each feature
        times_repeated = times.unsqueeze(-1).repeat(
            1, 1, self.feature_size
        )  # [B, T, F]
        times_reshaped = times_repeated.reshape(batch_size, -1)  # [B, T*F]

        # Apply masking
        values_masked = values_reshaped * (masks_reshaped > 0).float()
        times_masked = times_reshaped * (masks_reshaped > 0).float()

        # Get time encoding using masked times
        times_for_encoding = times_masked.reshape(
            batch_size, seq_len, self.feature_size
        )  # [B, T, F]
        transformed_times = self.pos_encoder(
            times_for_encoding.unsqueeze(-1)
        )  # [B, T, F, pos_dim]

        # One-hot encode modalities
        modality_encoding = F.one_hot(
            torch.arange(self.feature_size, device=device),
            num_classes=self.feature_size,
        )  # [F, F]
        modality_encoding = (
            modality_encoding.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, seq_len, -1, -1)
        )  # [B, T, F, F]

        # Prepare values for concatenation
        values_for_concat = values_masked.reshape(
            batch_size, seq_len, self.feature_size, 1
        )  # [B, T, F, 1]

        # Combine features
        combined_values = torch.cat(
            [
                transformed_times,  # [B, T, F, pos_dim]
                values_for_concat,  # [B, T, F, 1]
                modality_encoding,  # [B, T, F, F]
            ],
            dim=-1,
        )

        # Reshape to set format
        combined_values = combined_values.reshape(
            batch_size, seq_len * self.feature_size, -1
        )

        attention_mask = masks_reshaped > 0

        # Apply attention
        attention_weights, encoded = self.attention(combined_values, attention_mask)
        encoded_expanded = encoded.unsqueeze(1)  # [B, 1, T*F, hidden]

        if return_all:
            # Create time step masks for all steps at once [T, T*F]
            time_masks = torch.zeros(
                seq_len, seq_len * self.feature_size, device=device
            )
            feature_blocks = torch.arange(seq_len, device=device) * self.feature_size
            time_masks.scatter_(
                1,
                feature_blocks.unsqueeze(1).expand(-1, self.feature_size)
                + torch.arange(self.feature_size, device=device),
                1.0,
            )

            # Expand masks for batch and heads
            time_masks = time_masks.unsqueeze(0).unsqueeze(1)  # [1, 1, T, T*F]
            time_masks = time_masks.expand(
                batch_size, self.num_heads, -1, -1
            )  # [B, h, T, T*F]

            # Apply masked attention for all time steps at once
            attention_expanded = attention_weights.unsqueeze(2)  # [B, h, 1, T*F]
            masked_attention = attention_expanded * time_masks  # [B, h, T, T*F]

            # Process all time steps in parallel
            encoded_expanded = encoded.unsqueeze(1).unsqueeze(
                2
            )  # [B, 1, 1, T*F, hidden]
            time_attended = (
                masked_attention.unsqueeze(-1) * encoded_expanded
            )  # [B, h, T, T*F, hidden]
            time_features = time_attended.sum(dim=3)  # [B, h, T, hidden]

            # Prepare features for output network
            time_combined = time_features.permute(0, 2, 1, 3)  # [B, T, h, hidden]
            time_combined = time_combined.reshape(
                batch_size * seq_len, -1
            )  # [B*T, h*hidden]

            # Get predictions for all time steps
            output = self.output_net(time_combined)  # [B*T, output_dim]
            output = output.reshape(batch_size, seq_len, -1)  # [B, T, output_dim]
        else:
            # Single prediction using all time steps
            attended = encoded_expanded * attention_weights.unsqueeze(
                -1
            )  # [B, h, T*F, hidden]
            head_features = attended.sum(dim=2)  # [B, h, hidden]
            combined_features = head_features.reshape(batch_size, -1)  # [B, h*hidden]
            output = self.output_net(combined_features)

        return output
