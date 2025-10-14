import numpy as np
import pandas as pd
from basemodel import LinearModel
from basetrainer import Trainer
import itertools # for enumerating combinations

# Global variables to store means and stds for normalization
# (global variables are dirty !!!
# but there's nothing i could do about it as only submission.py can be modified)
means = None
stds = None

import numpy as np
import pandas as pd

def extract_features_chatgpt(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)

    # Raw logs of power-of-two parameters (helps linear models)
    for col in ['MWG','NWG','KWG','MDIMC','NDIMC','MDIMA','NDIMB','KWI','VWM','VWN']:
        X[f'log2_{col}'] = np.log2(df[col])

    # Workgroup/thread geometry
    X['wg_threads'] = df['MDIMC'] * df['NDIMC']               # threads per workgroup
    X['warp_count'] = X['wg_threads'] / 32.0                  # warps per workgroup
    X['occup_ge512'] = (X['wg_threads'] >= 512).astype(int)   # coarse occupancy indicator
    X['occup_eq1024'] = (X['wg_threads'] == 1024).astype(int) # max threads per WG

    # Macro tile metrics
    X['macro_tile_area'] = df['MWG'] * df['NWG']
    X['macro_per_thread'] = X['macro_tile_area'] / X['wg_threads']  # outputs per thread (approx)

    # Per-thread work estimate along M/N (vector widths amplify per-thread output)
    m_per_thread = df['MWG'] / df['MDIMC']
    n_per_thread = df['NWG'] / df['NDIMC']
    X['per_thread_m'] = m_per_thread
    X['per_thread_n'] = n_per_thread
    X['per_thread_out'] = m_per_thread * n_per_thread * df['VWM'] * df['VWN']

    # Arithmetic intensity proxy (reuse improves when caching enabled)
    # Base: (MWG*NWG)/(MWG+NWG) ~ larger tiles improve compute/byte ratio
    ai_base = (df['MWG'] * df['NWG']) / (df['MWG'] + df['NWG'])
    X['ai_base'] = ai_base
    X['ai_with_SA'] = ai_base * df['SA']
    X['ai_with_SB'] = ai_base * df['SB']
    X['ai_with_both'] = ai_base * (df['SA'] & df['SB'])

    # Vectorization effects
    X['vec_m'] = df['VWM']
    X['vec_n'] = df['VWN']
    X['vec_total'] = df['VWM'] * df['VWN']
    X['vec_ge4_m'] = (df['VWM'] >= 4).astype(int)   # 128-bit loads/stores
    X['vec_ge4_n'] = (df['VWN'] >= 4).astype(int)

    # Alignment/coalescing indicators
    X['align_m_threads'] = ((df['MWG'] % df['MDIMC']) == 0).astype(int)
    X['align_n_threads'] = ((df['NWG'] % df['NDIMC']) == 0).astype(int)
    X['align_m_vec'] = ((df['MWG'] % df['VWM']) == 0).astype(int)
    X['align_n_vec'] = ((df['NWG'] % df['VWN']) == 0).astype(int)

    # Stride penalties (hurt coalescing)
    X['stride_M'] = df['STRM']
    X['stride_N'] = df['STRN']
    X['stride_any'] = np.maximum(df['STRM'], df['STRN'])

    # K-dimension tiling & unrolling
    X['KWG'] = df['KWG']  # keep raw, but we already have log2_KWG
    X['KWI'] = df['KWI']  # keep raw, but we already have log2_KWI
    X['unroll_times_tileK'] = df['KWI'] * df['KWG']       # deeper unroll of a larger K-tile
    X['unroll_over_tileK'] = df['KWI'] / df['KWG']        # coarse register pressure proxy

    # Shared/local memory tile shape (bank and footprint hints)
    X['shared_A_footprint'] = df['MDIMA'] * df['KWG']
    X['shared_B_footprint'] = df['NDIMB'] * df['KWG']
    X['shared_A_is32'] = (df['MDIMA'] == 32).astype(int)  # align with 32-bank SMEM
    X['shared_B_is32'] = (df['NDIMB'] == 32).astype(int)

    # Macro tile balance (square tiles often better)
    X['mn_ratio'] = df['MWG'] / df['NWG']
    X['mn_diff'] = df['MWG'] - df['NWG']

    # Cache flags (keep raw)
    X['SA'] = df['SA']
    X['SB'] = df['SB']

    # Interaction: caching with vectorization (vector loads + cache typically good)
    X['cache_vec_m'] = df['SA'] * X['vec_ge4_m']
    X['cache_vec_n'] = df['SB'] * X['vec_ge4_n']

    # Optional: mild nonlinearity via squared logs (use regularization to handle collinearity)
    # for col in ['log2_MWG','log2_NWG','log2_KWG','log2_MDIMC','log2_NDIMC','log2_KWI','log2_VWM','log2_VWN']:
    #     X[f'{col}_sq'] = X[col] ** 2

    return X

def extract_features_claude_4_5(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract domain-informed features for GPU kernel performance prediction
    """
    features = pd.DataFrame()
    
    # ===== 1. BASIC FEATURES (14 original) =====
    features['MWG'] = df['MWG']
    features['NWG'] = df['NWG']
    features['KWG'] = df['KWG']
    features['MDIMC'] = df['MDIMC']
    features['NDIMC'] = df['NDIMC']
    features['MDIMA'] = df['MDIMA']
    features['NDIMB'] = df['NDIMB']
    features['KWI'] = df['KWI']
    features['VWM'] = df['VWM']
    features['VWN'] = df['VWN']
    features['STRM'] = df['STRM']
    features['STRN'] = df['STRN']
    features['SA'] = df['SA']
    features['SB'] = df['SB']
    
    # ===== 2. WORKGROUP & THREADING FEATURES =====
    # Total threads per workgroup
    features['total_threads'] = df['MDIMC'] * df['NDIMC']
    
    # Work per thread
    features['work_per_thread_M'] = df['MWG'] / df['MDIMC']
    features['work_per_thread_N'] = df['NWG'] / df['NDIMC']
    features['work_per_thread_total'] = features['work_per_thread_M'] * features['work_per_thread_N']
    
    # Tiling efficiency
    features['tile_size'] = df['MWG'] * df['NWG']
    features['tile_aspect_ratio'] = df['MWG'] / df['NWG']
    features['tile_aspect_ratio_log'] = np.log1p(features['tile_aspect_ratio'])
    
    # Thread block efficiency
    features['thread_block_aspect'] = df['MDIMC'] / df['NDIMC']
    
    # ===== 3. MEMORY ACCESS PATTERNS =====
    # Vector load efficiency
    features['vector_load_total'] = df['VWM'] * df['VWN']
    features['vector_load_ratio'] = df['VWM'] / df['VWN']
    features['vector_load_efficiency_M'] = df['MWG'] / df['VWM']
    features['vector_load_efficiency_N'] = df['NWG'] / df['VWN']
    
    # Memory transactions per thread
    features['mem_transactions_M'] = (df['MWG'] / df['MDIMC']) / df['VWM']
    features['mem_transactions_N'] = (df['NWG'] / df['NDIMC']) / df['VWN']
    
    # ===== 4. LOCAL MEMORY (SHARED MEMORY) FEATURES =====
    # Local memory shape vs workgroup tile
    features['local_mem_M_ratio'] = df['MDIMA'] / df['MWG']
    features['local_mem_N_ratio'] = df['NDIMB'] / df['NWG']
    features['local_mem_size'] = df['MDIMA'] * df['NDIMB']
    
    # Bank conflict potential
    features['local_mem_aspect'] = df['MDIMA'] / df['NDIMB']
    features['threads_per_local_M'] = df['MDIMA'] / df['MDIMC']
    features['threads_per_local_N'] = df['NDIMB'] / df['NDIMC']
    
    # ===== 5. CACHE & MEMORY HIERARCHY =====
    # Caching strategy effectiveness
    features['cache_both'] = df['SA'] * df['SB']
    features['cache_either'] = (df['SA'] + df['SB']).clip(0, 1)
    features['cache_mismatch'] = np.abs(df['SA'] - df['SB'])
    
    # Cache size estimation (when enabled)
    features['cache_size_A'] = df['SA'] * df['MWG'] * df['KWG']
    features['cache_size_B'] = df['SB'] * df['KWG'] * df['NWG']
    features['total_cache_size'] = features['cache_size_A'] + features['cache_size_B']
    
    # ===== 6. COMPUTATIONAL INTENSITY =====
    # FLOPs vs memory access ratio
    matrix_size = 2048
    features['total_flops'] = 2 * matrix_size**3  # constant, but useful for ratios
    
    # Memory footprint
    features['memory_footprint'] = (df['MWG'] * df['KWG'] + df['KWG'] * df['NWG']) * 4  # bytes
    features['compute_intensity'] = df['KWG'] / features['memory_footprint']
    
    # Inner dimension work
    features['k_iterations'] = matrix_size / df['KWG']
    features['k_work_per_iter'] = df['KWG'] * df['KWI']
    features['total_k_work'] = features['k_iterations'] * features['k_work_per_iter']
    
    # ===== 7. OCCUPANCY & RESOURCE UTILIZATION =====
    # Threads per SM (assuming warp size of 32)
    features['warps_per_workgroup'] = np.ceil(features['total_threads'] / 32)
    features['thread_utilization'] = features['total_threads'] / (features['warps_per_workgroup'] * 32)
    
    # Work distribution
    features['workgroups_M'] = matrix_size / df['MWG']
    features['workgroups_N'] = matrix_size / df['NWG']
    features['total_workgroups'] = features['workgroups_M'] * features['workgroups_N']
    
    # ===== 8. STRIDE & COALESCING =====
    features['stride_both'] = df['STRM'] * df['STRN']
    features['stride_either'] = (df['STRM'] + df['STRN']).clip(0, 1)
    features['stride_mismatch'] = np.abs(df['STRM'] - df['STRN'])
    
    # Interaction with vector loads
    features['stride_vector_M'] = df['STRM'] * df['VWM']
    features['stride_vector_N'] = df['STRN'] * df['VWN']
    
    # ===== 9. CRITICAL INTERACTIONS =====
    # Tiling vs threading
    features['tiles_per_thread'] = features['tile_size'] / features['total_threads']
    features['thread_tile_efficiency'] = (df['MWG'] * df['NWG']) / (df['MDIMC'] * df['NDIMC'])
    
    # Cache + vector interactions
    features['cache_vector_M'] = df['SA'] * df['VWM']
    features['cache_vector_N'] = df['SB'] * df['VWN']
    
    # Unrolling efficiency
    features['unroll_k_ratio'] = df['KWI'] / df['KWG']
    features['unroll_work'] = df['KWI'] * df['MWG'] * df['NWG']
    
    # Local memory + cache
    features['local_cache_M'] = df['MDIMA'] * df['SA']
    features['local_cache_N'] = df['NDIMB'] * df['SB']
    
    # ===== 10. POWER FEATURES (non-linear relationships) =====
    # Log transforms for skewed distributions
    features['log_MWG'] = np.log2(df['MWG'])
    features['log_NWG'] = np.log2(df['NWG'])
    features['log_KWG'] = np.log2(df['KWG'])
    features['log_tile_size'] = np.log2(features['tile_size'])
    features['log_threads'] = np.log2(features['total_threads'])
    
    # Square features for quadratic effects
    features['MWG_sq'] = df['MWG'] ** 2
    features['NWG_sq'] = df['NWG'] ** 2
    features['threads_sq'] = features['total_threads'] ** 2
    features['VWM_sq'] = df['VWM'] ** 2
    features['VWN_sq'] = df['VWN'] ** 2
    
    # Square root features
    features['sqrt_work_per_thread'] = np.sqrt(features['work_per_thread_total'])
    features['sqrt_tile_size'] = np.sqrt(features['tile_size'])
    
    # ===== 11. RATIO & EFFICIENCY METRICS =====
    features['memory_thread_ratio'] = features['memory_footprint'] / features['total_threads']
    features['work_memory_ratio'] = features['tile_size'] / features['memory_footprint']
    features['vector_thread_ratio'] = features['vector_load_total'] / features['total_threads']
    
    # Alignment features (powers of 2 are better)
    features['MWG_alignment'] = df['MWG'] % 64
    features['NWG_alignment'] = df['NWG'] % 64
    features['vector_alignment'] = (df['VWM'] * df['VWN']) % 4
    
    # ===== 12. COMPLEX INTERACTIONS =====
    # Three-way interactions for critical parameters
    features['tile_thread_vector_M'] = df['MWG'] * df['MDIMC'] * df['VWM']
    features['tile_thread_vector_N'] = df['NWG'] * df['NDIMC'] * df['VWN']
    features['cache_stride_vector_M'] = df['SA'] * df['STRM'] * df['VWM']
    features['cache_stride_vector_N'] = df['SB'] * df['STRN'] * df['VWN']
    
    # Local memory efficiency
    features['local_thread_efficiency'] = (df['MDIMA'] * df['NDIMB']) / (df['MDIMC'] * df['NDIMC'])
    features['local_tile_coverage'] = (df['MDIMA'] * df['NDIMB']) / (df['MWG'] * df['NWG'])
    
    # K-dimension interactions
    features['k_vector_product'] = df['KWG'] * df['KWI']
    features['k_cache_product'] = df['KWG'] * df['SA'] * df['SB']
    features['k_local_product'] = df['KWG'] * features['local_mem_size']
    
    # ===== 13. CATEGORICAL COMBINATIONS =====
    # Binary feature interactions
    features['SA_STRM'] = df['SA'] * df['STRM']
    features['SB_STRN'] = df['SB'] * df['STRN']
    features['SA_SB_STRM_STRN'] = df['SA'] * df['SB'] * df['STRM'] * df['STRN']
    
    # ===== 14. MEMORY BANDWIDTH ESTIMATION =====
    # Bytes transferred per workgroup
    features['bytes_read_A'] = df['MWG'] * df['KWG'] * 4 * (1 - df['SA'])
    features['bytes_read_B'] = df['KWG'] * df['NWG'] * 4 * (1 - df['SB'])
    features['bytes_written'] = df['MWG'] * df['NWG'] * 4
    features['total_bandwidth'] = features['bytes_read_A'] + features['bytes_read_B'] + features['bytes_written']
    
    # Bandwidth per thread
    features['bandwidth_per_thread'] = features['total_bandwidth'] / features['total_threads']
    
    # ===== 15. WORKLOAD BALANCE =====
    # How evenly work is distributed
    features['workgroup_balance'] = np.minimum(features['workgroups_M'], features['workgroups_N']) / \
                                    np.maximum(features['workgroups_M'], features['workgroups_N'])
    
    features['dimension_balance'] = np.minimum(df['MWG'], df['NWG']) / np.maximum(df['MWG'], df['NWG'])
    
    # ===== 16. ADVANCED INTERACTIONS =====
    # Multiplicative interactions between key parameters
    features['MWG_NWG'] = df['MWG'] * df['NWG']
    features['MWG_KWG'] = df['MWG'] * df['KWG']
    features['NWG_KWG'] = df['NWG'] * df['KWG']
    features['MDIMC_NDIMC'] = df['MDIMC'] * df['NDIMC']
    features['MDIMA_NDIMB'] = df['MDIMA'] * df['NDIMB']
    features['VWM_VWN'] = df['VWM'] * df['VWN']
    
    # Division interactions
    features['MWG_div_MDIMC'] = df['MWG'] / df['MDIMC']
    features['NWG_div_NDIMC'] = df['NWG'] / df['NDIMC']
    features['MWG_div_VWM'] = df['MWG'] / df['VWM']
    features['NWG_div_VWN'] = df['NWG'] / df['VWN']
    features['MDIMA_div_MDIMC'] = df['MDIMA'] / df['MDIMC']
    features['NDIMB_div_NDIMC'] = df['NDIMB'] / df['NDIMC']
    
    # ===== 17. WARP-LEVEL FEATURES =====
    # GPU execution happens at warp granularity
    features['threads_per_warp_utilization'] = (features['total_threads'] % 32) / 32
    features['warp_divergence_potential'] = features['thread_block_aspect']
    
    # ===== 18. POLYNOMIAL FEATURES FOR KEY INTERACTIONS =====
    # Create polynomial features for most critical parameters
    key_params = ['MWG', 'NWG', 'KWG', 'VWM', 'VWN']
    for i, p1 in enumerate(key_params):
        for p2 in key_params[i+1:]:
            features[f'{p1}_{p2}_interaction'] = df[p1] * df[p2]
            features[f'{p1}_{p2}_ratio'] = df[p1] / (df[p2] + 1e-8)

    return features

def extract_features_chatgpt_new2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build composite features for SGEMM kernel parameter tuning.
    Input:  pandas DataFrame with columns:
        ['MWG','NWG','KWG','MDIMC','NDIMC','MDIMA','NDIMB','KWI','VWM','VWN','STRM','STRN','SA','SB']
        (it may also contain 'Run_time' which will be ignored)
    Output: pandas DataFrame with ~500 engineered features (float32), suitable for linear regression.
    """
    # Ensure required columns exist
    required = ['MWG','NWG','KWG','MDIMC','NDIMC','MDIMA','NDIMB','KWI','VWM','VWN','STRM','STRN','SA','SB']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Initialize
    X = pd.DataFrame(index=df.index)

    # Base numeric and binary columns
    num_cols = ['MWG','NWG','KWG','MDIMC','NDIMC','MDIMA','NDIMB','KWI','VWM','VWN']
    bin_cols = ['STRM','STRN','SA','SB']

    # Copy base features (as floats)
    for col in num_cols:
        X[col] = df[col].astype(np.float32)
    for col in bin_cols:
        X[col] = df[col].astype(np.float32)

    # Log2 transforms (all inputs are powers-of-two, so log2 is integer-like)
    for col in num_cols:
        X[f'l2_{col}'] = np.log2(df[col].astype(np.float32))

    # Per-variable transforms
    MAT = 2048.0
    for col in num_cols:
        v = df[col].astype(np.float32)
        X[f'inv_{col}'] = 1.0 / v
        X[f'sq_{col}'] = v * v
        X[f'cube_{col}'] = v * v * v
        # Matrix-size alignment features
        X[f'ratio_2048_{col}'] = MAT / v
        X[f'tiles_2048_{col}'] = (MAT // v).astype(np.float32)
        X[f'mod_2048_{col}'] = (MAT % v).astype(np.float32)
        X[f'is_div_2048_{col}'] = ((MAT % v) == 0).astype(np.float32)
        # Log2 parity (even/odd exponent)
        X[f'log2_parity_{col}'] = (X[f'l2_{col}'] % 2).astype(np.float32)

    # Pairwise features for key pairs (balance/aspect/alignment)
    key_pairs = [
        ('MWG','NWG'),
        ('MDIMC','NDIMC'),
        ('MDIMA','NDIMB'),
        ('VWM','VWN'),
    ]
    for a, b in key_pairs:
        va = df[a].astype(np.float32)
        vb = df[b].astype(np.float32)
        X[f'{a}_eq_{b}'] = (va == vb).astype(np.float32)
        X[f'{a}_min_{b}'] = np.minimum(va, vb)
        X[f'{a}_max_{b}'] = np.maximum(va, vb)
        X[f'{a}_sum_{b}'] = va + vb
        X[f'{a}_prod_{b}'] = va * vb
        X[f'{a}_ratio_{b}'] = va / vb
        X[f'{b}_ratio_{a}'] = vb / va
        X[f'{a}_absdiff_{b}'] = np.abs(va - vb)
        X[f'{a}_gm_{b}'] = np.sqrt(va * vb)
        X[f'{a}_hm_{b}'] = 2.0 / (1.0/va + 1.0/vb)
        X[f'l2diff_{a}_{b}'] = np.abs(X[f'l2_{a}'] - X[f'l2_{b}'])

    # Workgroup and tile geometry
    X['WG_AREA'] = df['MWG'].astype(np.float32) * df['NWG'].astype(np.float32)
    X['WG_EDGE_SUM'] = df['MWG'].astype(np.float32) + df['NWG'].astype(np.float32)
    X['WG_ASPECT'] = df['MWG'].astype(np.float32) / df['NWG'].astype(np.float32)
    X['WG_VOLUME_M'] = df['MWG'].astype(np.float32) * df['KWG'].astype(np.float32)
    X['WG_VOLUME_N'] = df['NWG'].astype(np.float32) * df['KWG'].astype(np.float32)
    X['WG_VOLUME'] = X['WG_AREA'] * df['KWG'].astype(np.float32)

    # Local workgroup size and occupancy proxies
    X['LOCAL_SIZE'] = df['MDIMC'].astype(np.float32) * df['NDIMC'].astype(np.float32)
    X['OUTPUT_PER_WG'] = X['WG_AREA']
    X['OUTPUT_PER_ITEM'] = X['OUTPUT_PER_WG'] / (X['LOCAL_SIZE'] + 1e-6)
    X['OCCUPANCY_PROXY'] = X['LOCAL_SIZE'] / (X['WG_AREA'] + 1e-6)
    # Warp-based proxies (NVIDIA warp=32)
    X['warps'] = X['LOCAL_SIZE'] / 32.0
    X['warp_int'] = np.floor(X['warps'])
    X['warp_frac'] = X['warps'] - X['warp_int']
    X['mdimc_div_32'] = (df['MDIMC'] % 32 == 0).astype(np.float32)
    X['ndimc_div_32'] = (df['NDIMC'] % 32 == 0).astype(np.float32)

    # Vector width and memory access
    X['VEC_TOTAL'] = df['VWM'].astype(np.float32) + df['VWN'].astype(np.float32)
    X['VEC_PROD'] = df['VWM'].astype(np.float32) * df['VWN'].astype(np.float32)
    X['mdimc_per_lane'] = df['MDIMC'].astype(np.float32) / df['VWM'].astype(np.float32)
    X['ndimc_per_lane'] = df['NDIMC'].astype(np.float32) / df['VWN'].astype(np.float32)
    X['vec_match_m_flag'] = (df['MDIMC'] % df['VWM'] == 0).astype(np.float32)
    X['vec_match_n_flag'] = (df['NDIMC'] % df['VWN'] == 0).astype(np.float32)
    X['ld_span_m'] = df['MDIMC'].astype(np.float32) * df['VWM'].astype(np.float32)
    X['ld_span_n'] = df['NDIMC'].astype(np.float32) * df['VWN'].astype(np.float32)
    X['WG_AREA_by_vecprod'] = X['WG_AREA'] / (X['VEC_PROD'] + 1e-6)

    # Unrolling and K dimension
    X['KWI_log2'] = X['l2_KWI']
    X['KWG_log2'] = X['l2_KWG']
    X['KWG_times_KWI'] = df['KWG'].astype(np.float32) * df['KWI'].astype(np.float32)
    X['K_total_unrolled'] = MAT / (X['KWG_times_KWI'] + 1e-6)
    X['KWG_over_KWI'] = df['KWG'].astype(np.float32) / df['KWI'].astype(np.float32)

    # Off-chip memory pressure proxies and arithmetic intensity proxies
    X['mem_tiles_A'] = (df['MWG'].astype(np.float32) * df['KWG'].astype(np.float32)) / (df['VWM'].astype(np.float32) + 1e-6)
    X['mem_tiles_B'] = (df['NWG'].astype(np.float32) * df['KWG'].astype(np.float32)) / (df['VWN'].astype(np.float32) + 1e-6)
    X['mem_tiles_total'] = X['mem_tiles_A'] + X['mem_tiles_B']
    X['ai_proxy'] = (df['MWG'].astype(np.float32) * df['NWG'].astype(np.float32) * df['KWG'].astype(np.float32)) / (X['mem_tiles_total'] + 1e-6)
    X['ai_proxy_kwi'] = X['ai_proxy'] * df['KWI'].astype(np.float32)

    # Stride and caching flags + interactions
    X['stride_any'] = ((df['STRM'] > 0) | (df['STRN'] > 0)).astype(np.float32)
    X['stride_both'] = ((df['STRM'] > 0) & (df['STRN'] > 0)).astype(np.float32)
    X['cache_any'] = ((df['SA'] > 0) | (df['SB'] > 0)).astype(np.float32)
    X['cache_both'] = ((df['SA'] > 0) & (df['SB'] > 0)).astype(np.float32)
    X['stride_m_vwm'] = df['STRM'].astype(np.float32) * df['VWM'].astype(np.float32)
    X['stride_n_vwn'] = df['STRN'].astype(np.float32) * df['VWN'].astype(np.float32)
    X['cache_m_mwg'] = df['SA'].astype(np.float32) * df['MWG'].astype(np.float32)
    X['cache_n_nwg'] = df['SB'].astype(np.float32) * df['NWG'].astype(np.float32)
    X['cache_m_kwi'] = df['SA'].astype(np.float32) * df['KWI'].astype(np.float32)
    X['cache_n_kwi'] = df['SB'].astype(np.float32) * df['KWI'].astype(np.float32)
    X['ai_proxy_stride'] = X['ai_proxy'] / (1.0 + X['stride_any'])

    # Matrix tiling counts along M/N and total workgroups
    X['tiles_m'] = (MAT // df['MWG']).astype(np.float32)
    X['tiles_n'] = (MAT // df['NWG']).astype(np.float32)
    X['tiles_total'] = X['tiles_m'] * X['tiles_n']
    X['m_ratio'] = df['MWG'].astype(np.float32) / (df['MDIMC'].astype(np.float32) + 1e-6)
    X['n_ratio'] = df['NWG'].astype(np.float32) / (df['NDIMC'].astype(np.float32) + 1e-6)

    # Alignment of tiles with local memory shapes
    X['mem_align_A'] = (df['MWG'] % df['MDIMA'] == 0).astype(np.float32)
    X['mem_align_B'] = (df['NWG'] % df['NDIMB'] == 0).astype(np.float32)
    X['align_m_local'] = (df['MWG'] % (df['MDIMC'] * df['VWM']) == 0).astype(np.float32)
    X['align_n_local'] = (df['NWG'] % (df['NDIMC'] * df['VWN']) == 0).astype(np.float32)

    # Additional pair features across M/N tiling and local/caching
    X['m_over_local'] = df['MWG'].astype(np.float32) / (df['MDIMC'].astype(np.float32) + 1e-6)
    X['n_over_local'] = df['NWG'].astype(np.float32) / (df['NDIMC'].astype(np.float32) + 1e-6)
    X['m_over_memA'] = df['MWG'].astype(np.float32) / (df['MDIMA'].astype(np.float32) + 1e-6)
    X['n_over_memB'] = df['NWG'].astype(np.float32) / (df['NDIMB'].astype(np.float32) + 1e-6)
    X['m_mod_memA'] = (df['MWG'] % df['MDIMA']).astype(np.float32)
    X['n_mod_memB'] = (df['NWG'] % df['NDIMB']).astype(np.float32)
    X['cache_align_A'] = X['mem_align_A'] * df['SA'].astype(np.float32)
    X['cache_align_B'] = X['mem_align_B'] * df['SB'].astype(np.float32)

    # Interactions: log2 variables with binary flags
    for b in bin_cols:
        for col in num_cols:
            X[f'l2_{col}_x_{b}'] = X[f'l2_{col}'] * df[b].astype(np.float32)

    # Polynomial features on log2-transformed numeric variables (degree 2 and 3, with repetition)
    log_cols = [f'l2_{c}' for c in num_cols]
    # degree-2
    for i in range(len(log_cols)):
        for j in range(i, len(log_cols)):
            a, b = log_cols[i], log_cols[j]
            name = f'{a}*{b}' if a != b else f'{a}^2'
            X[f'poly2_{name}'] = X[a] * X[b]
    # degree-3
    for i in range(len(log_cols)):
        for j in range(i, len(log_cols)):
            for k in range(j, len(log_cols)):
                a, b, c = log_cols[i], log_cols[j], log_cols[k]
                if a == b == c:
                    name = f'{a}^3'
                else:
                    name = f'{a}*{b}*{c}'
                X[f'poly3_{name}'] = X[a] * X[b] * X[c]

    # A few more targeted interactions to round out the feature count
    targeted = [
        ('WG_AREA', 'KWI'), ('WG_VOLUME', 'KWI'),
        ('LOCAL_SIZE', 'VWM'), ('LOCAL_SIZE', 'VWN'),
        ('OUTPUT_PER_ITEM', 'SA'), ('OUTPUT_PER_ITEM', 'SB'),
        ('ai_proxy', 'SA'), ('ai_proxy', 'SB'),
        ('m_ratio', 'STRM'), ('n_ratio', 'STRN'),
        ('mem_tiles_total', 'STRM'), ('mem_tiles_total', 'STRN'),
    ]
    for a, b in targeted:
        X[f'{a}_x_{b}'] = X[a] * df[b].astype(np.float32)

    # Cast to float32 to keep memory reasonable
    X = X.astype(np.float32)

    # Do not include the target if present
    if 'Run_time' in X.columns:
        X = X.drop(columns=['Run_time'])

    return X

def load_and_preprocess_data(data_file: str = "data/train.csv"):
    dataset = pd.read_csv(data_file)
    """
    Divide the dataset into features and target

    You can do all possible modifications to features, but DO NOT change the targets

    return:
        features (np.ndarray): Input features, shape [num_samples, in_features]
        targets (np.ndarray): Target values, shape [num_samples]
    """

    # Separate features and targets
    targets = dataset['Run_time'].to_numpy()
    dataset = dataset.drop(columns=['Run_time'])

    dataset = extract_features_chatgpt(dataset)

    """
    # Multiply `KWI`, `VWM`, `VWN` by 4 to avoid log2 issues when =1
    dataset[['KWI', 'VWM', 'VWN']] *= 4

    # Construct LOG2 and INV features
    idxs = ['MWG', 'NWG', 'KWG', 'MDIMC', 'NDIMC', 'MDIMA', 'NDIMB',
            'KWI', 'VWM', 'VWN']
    transformed_features = {}
    idxs01 = ['STRM', 'STRN', 'SA', 'SB'] # features of [0, 1] range
    for col in idxs:
        transformed_features.update({#f'log2({col})': np.log2(dataset[col]),
                                     f'inv({col})': 1.0 / dataset[col]})
    dataset = pd.concat([dataset, pd.DataFrame(transformed_features, index=dataset.index)], axis=1)

    # Construct composite features MUL(X,Y)
    composite_idxs = idxs + idxs01 + list(transformed_features.keys())
    mul_features = {}
    for i, j in itertools.combinations(composite_idxs, 2):
        mul_features[f'mul({i},{j})'] = dataset[i] * dataset[j]
    for i, j, k in itertools.combinations(composite_idxs, 3):
        mul_features[f'mul({i},{j},{k})'] = dataset[i] * dataset[j] * dataset[k]

    dataset = pd.concat([dataset, pd.DataFrame(mul_features, index=dataset.index)], axis=1)

    """

    idxs = list(dataset.columns)
    inv_features = {}
    for col in idxs:
        inv_features.update({f'inv({col})': 1.0 / dataset[col]})
    dataset = pd.concat([dataset, pd.DataFrame(inv_features, index=dataset.index)], axis=1)
    # Construct composite features MUL(X,Y)
    composite_idxs = list(dataset.columns)
    mul_features = {}
    for i, j in itertools.combinations(composite_idxs, 2):
        mul_features[f'mul({i},{j})'] = dataset[i] * dataset[j]
    dataset = pd.concat([dataset, pd.DataFrame(mul_features, index=dataset.index)], axis=1)

    # Normalize the dataset
    # We assume that the first call of this function is for training set
    global means, stds
    if means is None or stds is None:
        means, stds = dataset.mean().to_numpy(), dataset.std().to_numpy()
        # Avoid division by zero: replace 0 std with 1 (no normalization for constant features)
        stds = np.where(stds < 1e-10, 1.0, stds)
    dataset = (dataset - means) / stds

    # print(dataset.mean())
    # print(dataset.std())

    features = dataset.to_numpy()
    print(f"Data size: {features.shape[0]}. Features num: {features.shape[1]}")
    return features, targets

class LinearRegressionModel(LinearModel):
    def __init__(self, in_features: int, out_features: int, l1_lambda: float = 0.0, l2_lambda: float = 0.0):
        """
        Linear regression model, inherits from LinearModel.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features (usually 1).
            l1_lambda (float): L1 regularization coefficient.
            l2_lambda (float): L2 regularization coefficient.
        """
        self.weight = np.random.randn(in_features, out_features) * 1e-6
        self.bias = np.zeros((1, out_features))
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the output given input.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
        """
        return features @ self.weight + self.bias

    def gradient(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> tuple:
        """
        Compute gradients for MSE loss with L1 and L2 regularization.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True values, shape [batch_size, out_features].
            predictions (np.ndarray): Predicted values, shape [batch_size, out_features].

        Returns:
            tuple: (dw, db), gradients for weights and bias.
        """
        m = features.shape[0] # Batch size 
        error = predictions - targets
        dw = features.T @ error / float(m)
        db = np.sum(error, axis=0, keepdims=True) / float(m)
        
        # L1 regularization gradient: lambda * sign(weight)
        if self.l1_lambda > 0:
            dw += self.l1_lambda * np.sign(self.weight)

        # L2 regularization gradient: weight * lambda
        if self.l2_lambda > 0:
            dw += self.l2_lambda * self.weight
        
        return dw, db

    def backpropagation(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray, learning_rate: float = 0.01) -> float:
        """
        Perform backpropagation, compute MSE loss with regularization and update parameters.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True values, shape [batch_size, out_features].
            predictions (np.ndarray): True values, shape [batch_size, out_features].
            learning_rate (float): Learning rate, default 0.01.
        """
        m = features.shape[0] # Batch size 
        
        #  MSE loss
        loss = (1/(2*m)) * np.sum((predictions - targets) ** 2)
        
        # L1 regularization
        if self.l1_lambda > 0:
            loss += self.l1_lambda * np.sum(np.abs(self.weight))
        
        # L2 regularization
        if self.l2_lambda > 0:
            loss += (self.l2_lambda / 2) * np.sum(self.weight ** 2)
        
        dw, db = self.gradient(features, targets, predictions)

        self.weight -= learning_rate * dw
        self.bias -= learning_rate * db
        # print(f"Weight norm: {np.linalg.norm(self.weight)}, Bias norm: {np.linalg.norm(self.bias)}")
        return loss

class LinearRegressionTrainer(Trainer):
    def __init__(self, model, train_dataloader, eval_dataloader=None, 
                 save_dir=None, learning_rate=0.01, eval_strategy="epoch", 
                 eval_steps=100, num_epochs=10, eval_metric="mae"):
        super().__init__(model, train_dataloader, eval_dataloader, save_dir, 
                         learning_rate*0+0.01, eval_strategy, eval_steps, num_epochs, eval_metric)

    def compute_loss(self, batch_pred, batch_grd):
        """
        Compute loss based on model type with detailed checks for linear regression.

        Args:
            batch_pred: Predicted values, shape [batch_size, out_features].
            batch_grd: True values/labels, shape [batch_size, out_features].

        Returns:
            float: Mean loss for the batch.
        """
        return np.mean((batch_pred - batch_grd) ** 2)

def linear_regression_analytic(X, y):
    """
    Calculate the analytical linear regression results.

    Args:
        X (np.ndarray): Input features, shape [num_samples, in_features]
        y (np.ndarray): True values, shape [num_samples, out_features]

    Return:
        weight (np.ndarray): Model weight
        bias (np.ndarray | float): Model bias
    """
    # Add a column of ones to X for the bias term
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])  # shape [n, d+1]
    # Use pseudo-inverse to handle singular matrix
    coef = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y  # shape [d+1, 1]
    bias = coef[0].reshape(1, 1)
    weight = coef[1:]
    return weight, bias

class LogisticRegressionModel(LinearModel):
    def __init__(self, in_features: int, out_features: int):
        """
        Logistic regression model, inherits from LinearModel.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features (usually 1 for binary classification).
        """
        self.weight = np.random.randn(in_features, out_features) * 1e-2
        self.bias = np.zeros((1, out_features))

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid function.

        Args:
            x (np.ndarray): Input values.

        Returns:
            np.ndarray: Sigmoid output.
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the output given input.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
        """
        return self._sigmoid(features @ self.weight + self.bias)

    def gradient(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> tuple:
        """
        Compute gradients for binary cross-entropy loss.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True labels (0 or 1), shape [batch_size, out_features].
            predictions (np.ndarray): Predicted probabilities, shape [batch_size, out_features].

        Returns:
            tuple: (dw, db), gradients for weights and bias.
        """
        m = float(features.shape[0])  # Batch size
        error = predictions - targets
        dw = (1/m) * features.T @ error
        db = (1/m) * np.sum(error, axis=0, keepdims=True)
        return dw, db
    
    def backpropagation(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray, learning_rate: float = 0.01) -> float:
        """
        Perform backpropagation, compute binary cross-entropy loss and update parameters.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True labels (0 or 1), shape [batch_size, out_features].
            learning_rate (float): Learning rate, default 0.01.

        Returns:
            float: Binary cross-entropy loss for the batch.
        """
        m = float(features.shape[0])  # Batch size
        epsilon = 1e-15  # To avoid log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = - (1/m) * np.sum(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        dw, db = self.gradient(features, targets, predictions)
        self.weight -= learning_rate * dw
        self.bias -= learning_rate * db
        return loss
        
class LogisticRegressionTrainer(Trainer):
    def __init__(self, model, train_dataloader, eval_dataloader=None, 
                 save_dir=None, learning_rate=0.01, eval_strategy="epoch", 
                 eval_steps=100, num_epochs=10, eval_metric="f1"):
        super().__init__(model, train_dataloader, eval_dataloader, save_dir, 
                         learning_rate, eval_strategy, eval_steps, num_epochs, eval_metric)
        
    def compute_loss(self, batch_pred, batch_grd):
        m = float(batch_grd.shape[0])  # Batch size
        epsilon = 1e-15  # To avoid log(0)
        batch_pred = np.clip(batch_pred, epsilon, 1 - epsilon)
        loss = - (1/m) * np.sum(batch_grd * np.log(batch_pred) + (1 - batch_grd) * np.log(1 - batch_pred))
        return loss
