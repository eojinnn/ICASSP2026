import math
import time
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2_norm_over_time(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Column-wise (feature-wise) ℓ2 normalization over time dimension.
    x: (B, H, T, D)
    returns: same shape
    """
    # Norm over time T for each feature column
    norm = torch.sqrt(torch.clamp((x ** 2).sum(dim=2, keepdim=True), min=eps))
    return x / norm


def _roll_time(x: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Time roll along dimension 2 (T).
    x: (B, H, T, D)
    """
    return torch.roll(x, shifts=shift, dims=2)


def _diag_abs_sum(mat: torch.Tensor) -> torch.Tensor:
    """
    Sum of absolute values of diagonal entries.
    mat: (B, H, D, D)
    return: (B, H)
    """
    # take diagonal along last two dims
    diag = torch.diagonal(mat, dim1=-2, dim2=-1)  # (B, H, D)
    return diag.abs().sum(dim=-1)


def _offdiag_abs_sum(mat: torch.Tensor) -> torch.Tensor:
    """
    Sum of absolute values of off-diagonal entries.
    mat: (B, H, D, D)
    return: (B, H)
    """
    abs_sum = mat.abs().sum(dim=(-2, -1))                  # (B, H)
    diag_abs = _diag_abs_sum(mat)                          # (B, H)
    return abs_sum - diag_abs


class CorrelatedAttentionBlock(nn.Module):
    """
    Correlated Attention Block (CAB) per 논문 3.2.2.

    입력/출력: (B, T, D)
    내부: 멀티헤드로 쪼개 (B, H, T, Dh) => 채널-채널 상관 행렬 (Dh x Dh)로 가중치 계산.
    - 정규화: 시간축 기준 ℓ2 정규화 (열 정규화)
    - 시차 후보: 1..T-1 중 TopK (k = c * ceil(log(T))) 선택 (λ로 자기/교차 상관 비중 조절)
    - 집계: 즉시(0) + 시차(l_i) 항을 β로 혼합, 소프트맥스는 특징 축(Dh)에서 수행
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        c_topk: int = 2,
        max_lag_candidates: Optional[int] = None,
        use_evenly_sampled_lags: bool = True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0  # We use temperature τ as parameter instead.

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable parameters: lambda (λ), beta (β), temperature (τ)
        # Initialize per paper: τ=1, λ=β=0.5
        self.log_tau = nn.Parameter(torch.zeros(1))              # tau = exp(log_tau) => init ~1
        self.lambda_auto = nn.Parameter(torch.tensor(0.5))       # λ
        self.beta_lag = nn.Parameter(torch.tensor(0.5))          # β

        # TopK configuration
        self.c_topk = int(c_topk)
        self.max_lag_candidates = max_lag_candidates
        self.use_evenly_sampled_lags = use_evenly_sampled_lags
        
        # 사용자 정의 추가가
        self.log_tau_lag = nn.Parameter(torch.zeros(1))
        self._time_idx_cache = {}  # (T, L) -> gather index 캐시


    def _choose_lag_candidates(self, T: int) -> List[int]:
        """
        T가 1보다 작으면 lag가 없으므로 빈리스트를 반환하고, 
        최대 후보 수가 지정되어있지 않거나, 전체 데이터보다 크게 지정되어있으면 모든 lag를 반환하고,
        균등분포 지연을 사용하면, lag들을 일정한 간격으로 샘플해서 사용하고, 
        random을 고르면 랜덤하게 몇 샘플 골라서 오름차순 하여, lag 값들을 반환.
        Generate candidate lags in [1, T-1]. lag 값들은 [1, T-1] 범위 내에서 선택됨.
        Optionally subsample to max_lag_candidates (even spacing).
        """
        if T <= 1:  # 길이가 1 이하면 lag 없으므로 빈 리스트 반환
            return []
        all_lags = list(range(1, T))  # 1..T-1 # lag 0(자신)은 제외
        if self.max_lag_candidates is None or self.max_lag_candidates >= (T - 1):
            return all_lags
        # Evenly sample candidates
        if self.use_evenly_sampled_lags:
            step = max(1, (T - 1) // self.max_lag_candidates)
            cand = list(range(1, T, step))[: self.max_lag_candidates]
            if cand and cand[-1] != (T - 1):
                cand[-1] = T - 1
            return cand
        else:
            # Random subset
            import random
            return sorted(random.sample(all_lags, self.max_lag_candidates))

    def _build_time_gather_idx(self, T: int, lag_list: torch.Tensor, device):
        """
        (1,1,L,T,1) 형태의 gather 인덱스를 캐시/반환.
        """
        key = (T, int(lag_list.numel()))
        idx = self._time_idx_cache.get(key, None)
        if idx is None or idx.device != device:
            base_t = torch.arange(T, device=device).view(1, 1, 1, T)   # (1,1,1,T)
            idx = (base_t - lag_list.view(1, 1, -1, 1)) % T            # (1,1,L,T)
            idx = idx.view(1, 1, -1, T, 1)                             # (1,1,L,T,1)
            self._time_idx_cache[key] = idx
        return idx

    def _topk_lags(
        self,
        q_hat: torch.Tensor,   # (B,H,T,Dh)
        k_hat: torch.Tensor,   # (B,H,T,Dh)
        lag_candidates: list,
        k_top: int
    ):
        """
        루프 없이 모든 lag를 벡터화해 '정확 점수(대각+비대각)'로 Top-K 선택.
        return: top_lags(B,H,k), top_scores(B,H,k)
        """
        B, H, T, Dh = q_hat.shape
        device = q_hat.device
        L = len(lag_candidates)
        if L == 0:
            z_l = torch.zeros(B, H, 0, dtype=torch.long, device=device)
            z_s = torch.zeros(B, H, 0, device=device)
            return z_l, z_s

        lag_list = torch.as_tensor(lag_candidates, device=device, dtype=torch.long)  # (L,)
        idx = self._build_time_gather_idx(T, lag_list, device)                       # (1,1,L,T,1)

        # 모든 lag를 한 번에 roll: (B,H,L,T,Dh)
        KhL = k_hat.unsqueeze(2).expand(B, H, L, T, Dh)
        Kroll = torch.gather(KhL, dim=3, index=idx.expand(B, H, L, T, 1).expand_as(KhL))

        # cov(l) = (T,Dh)^T @ (T,Dh)  → (BHL, Dh, Dh)
        Kf = Kroll.reshape(B*H*L, T, Dh)
        Qf = q_hat.unsqueeze(2).expand(B, H, L, T, Dh).reshape(B*H*L, T, Dh)
        cov = torch.matmul(Kf.transpose(1, 2), Qf).view(B, H, L, Dh, Dh)  # (B,H,L,Dh,Dh)

        # 점수 = λ*|diag|합 + (1-λ)*|off|합  (교차상관 반영)
        diag_abs  = torch.diagonal(cov, dim1=-2, dim2=-1).abs().sum(dim=-1)  # (B,H,L)
        total_abs = cov.abs().sum(dim=(-2, -1))                               # (B,H,L)
        off_abs   = total_abs - diag_abs
        lam = torch.clamp(self.lambda_auto, 0.0, 1.0)
        score = lam * diag_abs + (1.0 - lam) * off_abs                        # (B,H,L)

        # Top-K
        k_top = min(k_top, L)
        top_scores, top_idx = torch.topk(score, k=k_top, dim=-1)              # (B,H,k)
        top_lags = lag_list[top_idx]                                          # (B,H,k)
        return top_lags, top_scores



    def forward(self, x: torch.Tensor, need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: (B, T, D) 
        input : ((btf), u, c)
        returns: (B, T, D), (optional attention diagnostics)
        """
        B, T, D = x.shape 
        H = self.num_heads
        Dh = self.head_dim

        # Linear projections
        Q = self.q_proj(x)  # (B,T,D)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split heads -> (B, H, T, Dh)
        def split_heads(t):
            return t.view(B, T, H, Dh).transpose(1, 2).contiguous()

        Qh = split_heads(Q)
        Kh = split_heads(K)
        Vh = split_heads(V)

        # Column-wise ℓ2 normalization over time (per feature column)
        Qh_hat = _l2_norm_over_time(Qh)   # (B,H,T,Dh)
        Kh_hat = _l2_norm_over_time(Kh)

        # Build lag candidates and TopK size
        lag_candidates = self._choose_lag_candidates(T)
        k_top = max(1, int(self.c_topk * math.ceil(math.log(max(T, 2), 2))))  # use log2

        # Instantaneous term (lag=0)
        tau = torch.exp(self.log_tau).clamp_min(1e-4)  # temperature > 0
        # ROLL(K,0)^T Q_hat => (B,H,Dh,Dh) ## 지연이 0일때 channel attention
        cov0 = torch.einsum('bhtd,bhtc->bhdc', Kh_hat, Qh_hat)
        att0 = F.softmax(cov0 / tau, dim=-1)          # softmax over feature-dim
        inst = torch.einsum('bhtd,bhdc->bhtc', Vh, att0)  # (B,H,T,Dh)

        # Lagged terms
        # ---- 벡터화된 lag 집계 ----
        if len(lag_candidates) > 0:
            top_lags, top_scores = self._topk_lags(Qh_hat, Kh_hat, lag_candidates, k_top)  # (B,H,k)
            B, H, T, Dh = Qh_hat.shape
            Ksel = top_lags.shape[-1]
            device = Qh_hat.device

            # per-(b,h,k) shift 인덱스 만들기: idx[b,h,k,t] = (t - lag[b,h,k]) % T
            base_t = torch.arange(T, device=device).view(1,1,1,T)            # (1,1,1,T)
            idx = (base_t - top_lags.unsqueeze(-1)) % T                      # (B,H,k,T)
            # gather용 shape 맞추기
            idx = idx.unsqueeze(-1).expand(B, H, Ksel, T, Dh)                # (B,H,k,T,Dh)

            # (B,H,T,Dh) -> (B,H,k,T,Dh) 로 확장 후 time-dim(=3) gather
            Kh_exp = Kh_hat.unsqueeze(2).expand(B, H, Ksel, T, Dh)           # (B,H,k,T,Dh)
            Vh_exp = Vh.unsqueeze(2).expand(B, H, Ksel, T, Dh)               # (B,H,k,T,Dh)

            rolled_K = torch.gather(Kh_exp, dim=3, index=idx)                # (B,H,k,T,Dh)
            rolled_V = torch.gather(Vh_exp, dim=3, index=idx)                # (B,H,k,T,Dh)

            # 채널-채널 상관: (B,H,k,Dh,Dh) = sum_t rolled_K(b,h,k,t,d) * Qh_hat(b,h,t,c)
            cov_l = torch.einsum('bhktd,bhtc->bhkdc', rolled_K, Qh_hat)      # (B,H,k,Dh,Dh)
            att_l = F.softmax(cov_l / tau, dim=-1)                           # (B,H,k,Dh,Dh)

            # 기여도: (B,H,k,T,Dh)
            contrib_k = torch.einsum('bhktd,bhkdc->bhktc', rolled_V, att_l)  # (B,H,k,T,Dh)
            # k개 lag 평균 → (B,H,T,Dh)
            # lag_sum = contrib_k.mean(dim=2)
            tau_lag = torch.exp(getattr(self, 'log_tau_lag', torch.zeros(1, device=contrib_k.device))).clamp_min(1e-4) # 단순히 topk lag를 뽑는것이 아니라 점수기반 가중치합을 취함.
            w = F.softmax(top_scores / tau_lag, dim=-1).unsqueeze(-1).unsqueeze(-1)  # (B,H,k,1,1)
            lag_sum = (contrib_k * w).sum(dim=2)
        else:
            lag_sum = torch.zeros_like(inst)
            print('check lag_sum is zero?')

        # Combine instantaneous and lagged with β
        beta = torch.clamp(self.beta_lag, 0.0, 1.0)
        out_h = (1.0 - beta) * inst + beta * lag_sum  # (B,H,T,Dh)
        out_h = self.dropout(out_h)

        # Merge heads
        out = out_h.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        if need_weights:
            # Optionally return the instantaneous attention map averaged over heads for inspection
            # Shape: (B, T, Dh, Dh) is heavy; provide head-avg over DhxDh (optional)
            # Here we return None to keep memory light. Customize if you want diagnostics.
            return out, None
        return out, None


class TemporalSelfAttention(nn.Module):
    """
    표준 MultiheadAttention(시간 축) 래퍼.
    입력/출력: (B, T, D)
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.mha(x, x, x, need_weights=False)
        return y


class MixtureOfHeadAttention(nn.Module):
    """
    논문 3.2.3의 mixture-of-head attention 아이디어를 모듈화:
    - m개 헤드는 "시간 어텐션" (표준 MHA)
    - 나머지 헤드는 "상관 어텐션(CAB)"
    간단히 “두 블록의 출력을 평균(or 가중) 합성”하는 형태로 제공합니다.
    필요시 더 정교하게 head-splitting을 실제로 구현할 수 있습니다.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        m_temporal_heads: int = None,
        dropout: float = 0.0,
        c_topk: int = 2,
        max_lag_candidates: Optional[int] = None
    ):
        super().__init__()
        assert num_heads >= 1
        if m_temporal_heads is None:
            m_temporal_heads = num_heads // 2  # default split
        self.m_temporal_heads = m_temporal_heads
        self.num_heads = num_heads

        # 시간 어텐션 블록 (전체 head를 사용하지만 가중 결합 시 비중으로 제어)
        self.temporal = TemporalSelfAttention(embed_dim, num_heads=m_temporal_heads, dropout=dropout)
        # 상관 어텐션 블록 (전체 embed을 쓰되 내부에서 헤드 분할)
        self.cab = CorrelatedAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads - m_temporal_heads if num_heads - m_temporal_heads > 0 else 1,
            dropout=dropout,
            c_topk=c_topk,
            max_lag_candidates=max_lag_candidates
        )

        # 두 가지 출력을 합성할 때의 학습 가능한 가중치
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 0=temporal only, 1=CAB only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_temporal = self.temporal(x)                    # (B,T,D)
        y_cab, _ = self.cab(x)                          # (B,T,D)
        a = torch.clamp(self.alpha, 0.0, 1.0)
        y = (1.0 - a) * y_temporal + a * y_cab
        return y
