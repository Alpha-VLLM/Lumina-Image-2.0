import torch

'''
(t-r) is very small, so we need to set a small epsilon to avoid log(0)
'''
eps_dict = {
    torch.float16: 1e-3,
    torch.float32: 1e-4,
    torch.bfloat16: 0.0079,
}


class InductiveMomentMatching:
    def __init__(
            self,
            noise_schedule="fm",
            sigma_delta=0.5,
            T=0.994,
            time_scale=1000.0):
        self.noise_schedule = noise_schedule
        self.sigma_data = sigma_delta
        self.T = T
        self.time_scale = time_scale
    

    def mapping_function(self, s, t, k=12, log_eta_min=None, log_eta_max=None, min_gap=1e-4):
        """
        Calculate the mapping function r(s,t) for Inductive Moment Matching with batch inputs,
        using logarithmic operations for numerical stability.
        
        Args:
            s: Tensor of lower time boundaries (batch)
            t: Tensor of upper time boundaries (batch)
            k: Power for decrement size calculation (default=12)
            log_eta_min: Log of minimum value of eta (default=log(0+eps))
            log_eta_max: Log of maximum value of eta (default=log(160))
            min_gap: Optional minimum gap between t and r (for numerical stability)
        
        Returns:
            r: Tensor of intermediate time values between s and t
        """
        # Ensure inputs are tensors with the same device and dtype
        device = t.device if torch.is_tensor(t) else (s.device if torch.is_tensor(s) else 'cpu')
        dtype = t.dtype if torch.is_tensor(t) else (s.dtype if torch.is_tensor(s) else torch.float32)
        
        if not torch.is_tensor(s):
            s = torch.tensor(s, device=device, dtype=dtype)
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device, dtype=dtype)
        
        # Set default log_eta values if not provided
        eps = eps_dict[dtype]  # Small epsilon to avoid log(0)
        if log_eta_min is None:
            log_eta_min = torch.log(torch.tensor(eps, device=device, dtype=dtype))
        if log_eta_max is None:
            log_eta_max = torch.log(torch.tensor(160.0, device=device, dtype=dtype))
        
        # Calculate log decrement in eta space
        # log(epsilon) = log((eta_max - eta_min) / 2^k) = log(eta_max - eta_min) - k*log(2)
        log_2 = torch.log(torch.tensor(2.0, device=device, dtype=dtype))
        log_decrement = torch.log(torch.exp(log_eta_max) - torch.exp(log_eta_min)) - k * log_2
        
        # Define eta function (noise-to-signal ratio) for OT-FM schedule
        # For OT-FM: eta(t) = t/(1-t)
        # We'll do this calculation directly, no need for log here
        eta_t = t / (1.0 - t)
        log_eta_t = torch.log(eta_t)
        
        # Calculate decremented eta value in log space
        # log(eta_r) = log(max(0, eta_t - epsilon))
        # Since we can't directly subtract in log space, we'll convert back briefly
        decrement = torch.exp(log_decrement)
        eta_r = torch.clamp(eta_t - decrement, min=eps)
        
        # Define inverse eta function
        # For OT-FM: inverse_eta(eta_value) = eta_value/(1+eta_value)
        r = eta_r / (1.0 + eta_r)
        
        # Apply minimum gap if specified (for numerical stability in BFloat16)
        if min_gap is not None:
            min_gap = torch.tensor(min_gap, device=device, dtype=dtype)
            r = torch.minimum(t - min_gap, r)
        
        # Ensure r is not less than s
        r = torch.maximum(s, r)
        
        return r


    def generate_time_parameters(self, batch_size, M, T=0.994, device='cuda', dtype=torch.float32):
        """
        Generate time parameters s and t for Inductive Moment Matching with M-particle groups.
        
        Args:
            batch_size: Total batch size B
            M: Number of particles per group (B/M groups in total)
            eps: Minimum time value (default=0.0)
            T: Maximum time value (default=0.994 for OT-FM schedule)
            device: Device to place tensors on
            dtype: Data type for tensors
            
        Returns:
            s: Tensor of shape [batch_size] with lower time boundaries 
            t: Tensor of shape [batch_size] with upper time boundaries
            group_indices: Tensor of shape [batch_size] with group indices
        """
        # Check if batch_size is divisible by M
        assert batch_size % M == 0, f"Batch size {batch_size} must be divisible by M={M}"
        
        eps = eps_dict[dtype]

        num_groups = batch_size // M
        
        # Generate time parameters for each group
        group_t = torch.zeros(num_groups, dtype=dtype, device=device)
        group_s = torch.zeros(num_groups, dtype=dtype, device=device)
        
        # Sample t from uniform distribution U(eps, T)
        group_t = torch.rand(num_groups, dtype=dtype, device=device) * (T - eps) + eps
        
        # Sample s from uniform distribution U(eps, t) for each t
        group_s = torch.rand(num_groups, dtype=dtype, device=device) * (group_t - eps) + eps
        
        # Create expanded versions for the full batch
        # Each group of M consecutive samples shares the same s and t
        s = torch.zeros(batch_size, dtype=dtype, device=device)
        t = torch.zeros(batch_size, dtype=dtype, device=device)
        group_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for group_idx in range(num_groups):
            start_idx = group_idx * M
            end_idx = start_idx + M
            
            s[start_idx:end_idx] = group_s[group_idx]
            t[start_idx:end_idx] = group_t[group_idx]
            group_indices[start_idx:end_idx] = group_idx
        
        return s, t, group_indices


    def get_alpha_sigma(self, t): 
        if self.noise_schedule == 'fm':
            alpha_t = (1 - t)
            sigma_t = t
        elif self.noise_schedule == 'vp_cosine': 
            alpha_t = torch.cos(t * torch.pi * 0.5)
            sigma_t = torch.sin(t * torch.pi * 0.5)
            
        return alpha_t, sigma_t 
    

    def add_noise(self, y, t,   noise=None):
        if noise is None:
            noise = torch.randn_like(y) * self.sigma_data

        alpha_t, sigma_t = self.get_alpha_sigma(t)
         
        return alpha_t * y + sigma_t * noise, noise 


    def ddim(self, yt, y, t, s, noise=None):
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        alpha_s, sigma_s = self.get_alpha_sigma(s)
        
        if noise is None: 
            ys = (alpha_s -   alpha_t * sigma_s / sigma_t) * y + sigma_s / sigma_t * yt
        else:
            ys = alpha_s * y + sigma_s * noise
        return ys
    

    def simple_edm_sample_function(self, yt, y, t, s ):
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        alpha_s, sigma_s = self.get_alpha_sigma(s)
         
        c_skip = (alpha_t * alpha_s + sigma_t * sigma_s) / (alpha_t**2 + sigma_t**2)

        c_out = - (alpha_s * sigma_t - alpha_t * sigma_s) * (alpha_t**2 + sigma_t**2).rsqrt() * self.sigma_data
        
        return c_skip * yt + c_out * y
    

    def euler_fm_sample_function(self, yt, y, t, s ):
        assert self.noise_schedule == 'fm'  

        return  yt - (t - s) * self.sigma_data *  y
    

    def forward_model(self, model, x_t, t, s, model_kwargs):
        x_s = model(
            x_t = x_t, 
            t = t, 
            s = s,
            **model_kwargs
        )

        


    def training_losses(self, model, x1, model_kwargs=None):
        s, t, group_indices = self.generate_time_parameters(
            x1.shape[0], 
            M=1, 
            T=self.T, 
            device=x1.device, 
            dtype=x1.dtype
        )

        if model_kwargs is None:
            model_kwargs = {}

        r = self.mapping_function(s, t)