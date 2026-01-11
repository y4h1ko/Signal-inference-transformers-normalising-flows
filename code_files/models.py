from .imports_and_libraries import *
from .positional_encodings import *


#e-only transformer, head (w)
class TransformerModel1(nn.Module):
    """
    Encoder-only Transformer for time-series to single-frequency regression.

    The input is a 1D time-series (discretized sine signal). The model embeds
    each time step, adds sinusoidal positional encodings, processes the sequence
    with Transformer encoder layers, pools over time (mean pooling), and predicts
    a single scalar frequency.

    Parameters
    ----------
    seq_len : int, optional
        Sequence length (number of time steps). Defaults to `cfg.discr_of_time`.
    d_model : int, optional
        Transformer embedding dimension. Defaults to `cfg.dmodel`.
    nhead : int, optional
        Number of attention heads. Defaults to `cfg.nhead`.
    num_layers : int, optional
        Number of Transformer encoder layers. Defaults to `cfg.num_layers`.
    dim_f : int, optional
        Feed-forward network dimension in encoder layers. Defaults to `cfg.dim_f`.
    dropout : float, optional
        Dropout probability in encoder layers. Defaults to `cfg.dropout`.
    """

    def __init__(self, seq_len: int=cfg.discr_of_time, d_model: int=cfg.dmodel, nhead: int=cfg.nhead, num_layers: int=cfg.num_layers, 
                 dim_f: int=cfg.dim_f, dropout: float=cfg.dropout):
        
        super().__init__()

        self.input_embedding = nn.Linear(1, d_model)
        self.position_encoding = PositionalEncoding(d_model, seq_len)       #creates order for time
        
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_f, dropout=dropout, batch_first=True)        #creates layers/blocks
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers)

        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))     #normalization and prediction for w

    def forward(self, src) -> torch.Tensor:
        """
        Forward pass producing a point estimate of frequency.

        Parameters
        ----------
        src : torch.Tensor
            Input tensor of shape (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Predicted frequency tensor of shape (batch_size, 1).
        """

        src = src.unsqueeze(-1)

        src = self.input_embedding(src)
        src = self.position_encoding(src)
        z = self.transformer_encoder(src)

        pool = z.mean(dim=1)
        output = self.head(pool)
        return output



class HeadWithFlow1(nn.Module):
    """
    Conditional normalizing-flow head for a single frequency parameter (omega).

    The head models p(omega | context) using a Masked Autoregressive Flow (MAF).
    A small context network maps the Transformer pooled representation to a context
    vector used by the flow transforms.

    Parameters
    ----------
    context_dim : int
        Dimensionality of the context vector (Transformer pooled embedding size).
    hidden_features : int, optional
        Hidden size used both in the context network and in MAF transforms.
        Defaults to `cfg.flow_hidden_features`.
    num_layers : int, optional
        Number of MAF transforms in the flow. Defaults to `cfg.flow_num_layers`.
    """

    def __init__(self, context_dim: int, hidden_features: int=cfg.flow_hidden_features, num_layers: int=cfg.flow_num_layers):
        super().__init__()

        self.context_dim  = context_dim
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.context_net = nn.Linear(context_dim, hidden_features)

        #MAF - creating matrixes which will transform distribution
        transform_list = []
        for _ in range(num_layers):
            transform_list.append(transforms.MaskedAffineAutoregressiveTransform(features=1, hidden_features=hidden_features, context_features=hidden_features))

        #chaining transformation sequence
        transform = transforms.CompositeTransform(transform_list)

        #wraps flow as object and keeps the information
        base_dist = distributions.StandardNormal(shape=[1])
        self.flow = flows.Flow(transform=transform, distribution=base_dist)

    def encode_context_head(self, context) -> torch.Tensor:
        """
        Map context features to flow conditioning features.

        Parameters
        ----------
        context : torch.Tensor
            Context tensor of shape (batch_size, context_dim).

        Returns
        -------
        torch.Tensor
            Encoded context of shape (batch_size, hidden_features).
        """
        return self.context_net(context)

    def log_prob(self, omega, context) -> torch.Tensor:
        """
        Compute log p(omega | context).

        Parameters
        ----------
        omega : torch.Tensor
            Target omega values of shape (batch_size, 1).
        context : torch.Tensor
            Context tensor of shape (batch_size, context_dim).

        Returns
        -------
        torch.Tensor
            Log-probabilities of shape (batch_size,).
        """

        ctx = self.encode_context_head(context)
        log_p = self.flow.log_prob(inputs=omega, context=ctx)
        return log_p

    def sample(self, context, num_samples: int) -> torch.Tensor:
        """
        Draw samples from p(omega | context).

        Parameters
        ----------
        context : torch.Tensor
            Context tensor of shape (batch_size, context_dim).
        num_samples : int
            Number of samples to draw per context item.

        Returns
        -------
        torch.Tensor
            Samples of shape (batch_size, num_samples, 1).
        """
        #function gives out mean, uncertainity and shape of distribution
        ctx = self.encode_context_head(context)  
    
        samples_bs1 = self.flow.sample(num_samples=num_samples, context=ctx)
        samples = samples_bs1.permute(1, 0, 2) 
        return samples



class TransformerModel2(nn.Module):
    """
    Transformer encoder with a conditional normalizing-flow head (single omega).

    Methods:
    - forward(x): returns a point estimate computed as the mean of flow samples.
    - log_prob(x, y): returns log p(y | x) for NLL training.
    - sample(x, S): returns S samples for uncertainty estimation.

    Parameters
    ----------
    seq_len, d_model, nhead, num_layers, dim_f, dropout : see TransformerModel1.
    flow_hidden_features : int, optional
        Hidden size for the flow head. Defaults to `cfg.flow_hidden_features`.
    flow_num_layers : int, optional
        Number of flow transforms. Defaults to `cfg.flow_num_layers`.
    """

    def __init__(self, seq_len: int=cfg.discr_of_time, d_model: int=cfg.dmodel, nhead: int=cfg.nhead, num_layers: int=cfg.num_layers,
                 dim_f: int=cfg.dim_f, dropout: float=cfg.dropout, flow_hidden_features: int=cfg.flow_hidden_features, flow_num_layers: int=cfg.flow_num_layers):

        super().__init__()

        self.input_embedding   = nn.Linear(1, d_model)
        self.position_encoding = PositionalEncoding(d_model, seq_len)

        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_f, dropout=dropout, batch_first=True)        #creates layers/blocks
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers)

        self.pre_head_norm = nn.LayerNorm(d_model)

        #flowhead
        self.flow_head = HeadWithFlow1( context_dim=d_model, hidden_features=flow_hidden_features, num_layers=flow_num_layers)


    def forward(self, src) -> torch.Tensor:
        """
        Produce a point estimate by sampling the flow (no gradients).

        Parameters
        ----------
        src : torch.Tensor
            Input tensor of shape (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Predicted omega mean of shape (batch_size, 1).
        """

        src = src.unsqueeze(-1)
        src = self.input_embedding(src)
        src = self.position_encoding(src)
        z = self.transformer_encoder(src)
        pool = z.mean(dim=1)
        head_norm = self.pre_head_norm(pool)
        
        with torch.no_grad():
            samples = self.flow_head.sample(head_norm, num_samples=20)
        mu = samples.mean(dim=0)
        return mu

    def log_prob(self, src, target) -> torch.Tensor:
        """
        Compute log p(target | src) for NLL training.

        Parameters
        ----------
        src : torch.Tensor
            Input tensor of shape (batch_size, seq_len).
        target : torch.Tensor
            Target omega of shape (batch_size, 1).

        Returns
        -------
        torch.Tensor
            Log-probabilities of shape (batch_size,).
        """

        src = src.unsqueeze(-1)
        src = self.input_embedding(src)
        src = self.position_encoding(src)
        z = self.transformer_encoder(src)
        pool = z.mean(dim=1)
        head_norm = self.pre_head_norm(pool)
        
        log_p = self.flow_head.log_prob(target, context=head_norm)
        return log_p

    def sample(self, src, num_samples: int = 100) -> torch.Tensor:
        """
        Sample from the predictive distribution p(omega | src).

        Parameters
        ----------
        src : torch.Tensor
            Input tensor of shape (batch_size, seq_len).
        num_samples : int, optional
            Number of samples per input. Defaults to 100.

        Returns
        -------
        torch.Tensor
            Samples of shape (batch_size, num_samples, 1).
        """
        
        src = src.unsqueeze(-1)
        src = self.input_embedding(src)
        src = self.position_encoding(src)
        z = self.transformer_encoder(src)
        pool = z.mean(dim=1)
        head_norm = self.pre_head_norm(pool) 
        
        samples = self.flow_head.sample(head_norm, num_samples)
        return samples



class HeadWithFlow2w(nn.Module):
    """
    Conditional normalizing-flow head for two frequency parameters (w1, w2).

    Models p([w1, w2] | context) using a MAF flow in 2D. Optional random
    permutations between flow layers can be used to improve mixing.

    Parameters
    ----------
    context_dim : int
        Dimensionality of the Transformer context vector.
    hidden_features : int, optional
        Hidden size used in the context network and MAF transforms.
        Defaults to `cfg.flow_hidden_features`.
    num_layers : int, optional
        Number of MAF transforms. Defaults to `cfg.flow_num_layers`.
    out_dim : int, optional
        Dimensionality of the output (2 for two frequencies). Defaults to 2.
    use_permutations : bool, optional
        If True, insert random permutations between MAF blocks (except last).
        Defaults to True.
    """

    def __init__(self, context_dim: int, hidden_features: int=cfg.flow_hidden_features, num_layers: int=cfg.flow_num_layers, 
                 out_dim: int=2, use_permutations: bool=True):
        super().__init__()

        self.context_dim  = context_dim
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.context_net = nn.Linear(context_dim, hidden_features)

        #MAF - transform distribution with matrixes
        transform_list = []
        for i in range(num_layers):
            maf = transforms.MaskedAffineAutoregressiveTransform(features=out_dim, hidden_features=hidden_features, context_features=hidden_features)
            transform_list.append(maf)
            if use_permutations and i < num_layers - 1:
                perm = transforms.RandomPermutation(features=out_dim)
                transform_list.append(perm)

        #chaining transformation sequence
        transform = transforms.CompositeTransform(transform_list)

        #wraps flow as object and keeps the information
        base_dist = distributions.StandardNormal(shape=[out_dim])
        self.flow = flows.Flow(transform=transform, distribution=base_dist)

    def encode_context_head(self, context)-> torch.Tensor:
        """
        Map Transformer context to flow conditioning features.

        Parameters
        ----------
        context : torch.Tensor
            Context tensor of shape (batch_size, context_dim).

        Returns
        -------
        torch.Tensor
            Encoded context of shape (batch_size, hidden_features).
        """
        return self.context_net(context)

    def log_prob(self, omega, context) -> torch.Tensor:
        """
        Compute log p(omega | context) for 2D omega targets.

        Parameters
        ----------
        omega : torch.Tensor
            Target tensor of shape (batch_size, out_dim).
        context : torch.Tensor
            Context tensor of shape (batch_size, context_dim).

        Returns
        -------
        torch.Tensor
            Log-probabilities of shape (batch_size,).
        """
        ctx = self.encode_context_head(context)
        log_p = self.flow.log_prob(inputs=omega, context=ctx)
        return log_p

    def sample(self, context, num_samples: int) -> torch.Tensor:
        """
        Compute log p(omega | context) for 2D omega targets.

        Parameters
        ----------
        omega : torch.Tensor
            Target tensor of shape (batch_size, out_dim).
        context : torch.Tensor
            Context tensor of shape (batch_size, context_dim).

        Returns
        -------
        torch.Tensor
            Log-probabilities of shape (batch_size).
        """
        ctx = self.encode_context_head(context)  
    
        samples = self.flow.sample(num_samples=num_samples, context=ctx)
        if samples.dim() == 3:
            S0, S1, D = samples.shape

            if S0 == num_samples:
                samples = samples.permute(1, 0, 2)
            elif S1 == num_samples:
                samples = samples

        return samples



class TransformerModel3(nn.Module):
    """
    Transformer encoder with a conditional normalizing-flow head for two frequencies.

    Targets are treated as an unordered set {w1, w2}. During training, the log-likelihood
    is symmetrized by averaging p(w1, w2) and p(w2, w1) in log-space.

    Methods:
    - forward(x): returns a point estimate computed from flow samples (sorted per sample).
    - log_prob(x, y): returns symmetric log p(y | x).
    - sample(x, S): returns S samples for uncertainty.

    Parameters
    ----------
    seq_len, d_model, nhead, num_layers, dim_f, dropout : Transformer hyperparameters.
    flow_hidden_features : int, optional
        Hidden size for the flow head. Defaults to `cfg.flow_hidden_features`.
    flow_num_layers : int, optional
        Number of flow transforms. Defaults to `cfg.flow_num_layers`.
    out_dim : int, optional
        Output dimension (2 for two frequencies). Defaults to 2.
    """

    def __init__(self, seq_len: int=cfg.discr_of_time, d_model: int=cfg.dmodel, nhead: int=cfg.nhead, num_layers: int=cfg.num_layers,
                 dim_f: int=cfg.dim_f, dropout: float=cfg.dropout, flow_hidden_features: int=cfg.flow_hidden_features, flow_num_layers: int=cfg.flow_num_layers,
                 out_dim: int=2):

        super().__init__()

        self.input_embedding = nn.Linear(1, d_model)
        self.position_encoding = PositionalEncoding(d_model, seq_len)

        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_f, dropout=dropout, batch_first=True)        #creates layers/blocks
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers)

        self.pre_head_norm = nn.LayerNorm(d_model)

        #flowhead
        self.flow_head = HeadWithFlow2w(context_dim=d_model, hidden_features=flow_hidden_features, num_layers=flow_num_layers, out_dim=out_dim)

    def forward(self, src, num_samples: int = 50) -> torch.Tensor:
        """
        Produce a point estimate for {w1, w2} using Monte Carlo sampling.

        Samples are drawn from the flow, sorted along the last dimension so that
        the output is comparable across permutations, then averaged.

        Parameters
        ----------
        src : torch.Tensor
            Input tensor of shape (batch_size, seq_len).
        num_samples : int, optional
            Number of samples drawn per input. Defaults to 50.

        Returns
        -------
        torch.Tensor
            Predicted pair mean of shape (batch_size, 2).
        """

        src = src.unsqueeze(-1)
        src = self.input_embedding(src)
        src = self.position_encoding(src)
        z = self.transformer_encoder(src)
        pool = z.mean(dim=1)
        head_norm = self.pre_head_norm(pool)

        samples = self.flow_head.sample(head_norm, num_samples)
        samples_sorted, _ = torch.sort(samples, dim=-1)
        mu = samples_sorted.mean(dim=1)
        return mu

    def log_prob(self, src, target) -> torch.Tensor:
        """
        Compute symmetric log-likelihood for unordered targets {w1, w2}.

        Computes:
            log(0.5 * (p(w1, w2) + p(w2, w1)))

        Parameters
        ----------
        src : torch.Tensor
            Input tensor of shape (batch_size, seq_len).
        target : torch.Tensor
            Target tensor of shape (batch_size, 2).

        Returns
        -------
        torch.Tensor
            Symmetrized log-probabilities of shape (batch_size,).
        """

        src = src.unsqueeze(-1)
        src = self.input_embedding(src)
        src = self.position_encoding(src)
        z = self.transformer_encoder(src)

        pool = z.mean(dim=1)
        head_norm = self.pre_head_norm(pool)

        #(w1, w2) and (w2, w1) and their mixture
        log_p1 = self.flow_head.log_prob(target, context=head_norm)
        log_p2 = self.flow_head.log_prob(target.flip(dims=[1]), context=head_norm)

        #p_sym = 0.5 * (p(w1,w2) + p(w2,w1))
        stacked = torch.stack([log_p1, log_p2], dim=1)
        log_p_sym = torch.logsumexp(stacked, dim=1) - math.log(2.0)

        return log_p_sym
    
    def sample(self, src, num_samples: int=100) -> torch.Tensor:
        """
        Sample from the predictive distribution p([w1, w2] | src).

        Parameters
        ----------
        src : torch.Tensor
            Input tensor of shape (batch_size, seq_len).
        num_samples : int, optional
            Number of samples per input. Defaults to 100.

        Returns
        -------
        torch.Tensor
            Samples of shape (batch_size, num_samples, 2).
        """
        
        src = src.unsqueeze(-1)
        src = self.input_embedding(src)
        src = self.position_encoding(src)
        z = self.transformer_encoder(src)
        pool = z.mean(dim=1)
        head_norm = self.pre_head_norm(pool) 
        
        samples = self.flow_head.sample(head_norm, num_samples)
        return samples
