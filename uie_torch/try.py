from uie_predictor import UIEPredictor
from pprint import pprint

schema = ['肿瘤的大小', '肿瘤的个数', '肝癌级别', '脉管内癌栓分级'] # Define the schema for entity extraction
ie = UIEPredictor(model='uie-base', schema=schema)
pprint(ie("（右肝肿瘤）肝细胞性肝癌（II-III级，梁索型和假腺管型），肿瘤包膜不完整，紧邻肝被膜，侵及周围肝组织，未见脉管内癌栓（MVI分级：M0级）及卫星子灶形成。（肿物1个，大小4.2×4.0×2.8cm）。")) # Better print results using pprint