from .transformer import GraphTransformer
from omegaconf import DictConfig
from .brainnetcnn import BrainNetCNN
from .fbnetgen import FBNETGEN
from .BNT import BrainNetworkTransformer
from .BNT.bnt_mbs import BrainNetworkTransformer_MBS  # newly added
from .BNT.bnt_mbs_cluster import BrainNetworkTransformer_MBS_Cluster



def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    return eval(config.model.name)(config).cuda()
