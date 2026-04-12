from dataclasses import dataclass


@dataclass
class NetworkConfig:
    num_blocks: int = 10
    num_filters: int = 128
    se_ratio: int = 4
    input_planes: int = 112
    policy_size: int = 1858
    value_size: int = 3  # WDL: win, draw, loss
    policy_conv_filters: int = 80
    value_conv_filters: int = 32
    value_fc_size: int = 128
    # Moves-left head
    mlh_conv_filters: int = 8
    mlh_fc_size: int = 128
    # Attention policy head
    use_attention_policy: bool = True
    policy_embedding_size: int = 64
    policy_d_model: int = 64
