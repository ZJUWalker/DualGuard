#将模型拆分成三份，head，server，tail
from dataclasses import dataclass,field

class Intermediate(object):
    HEAD2SERVER_ACTIVATION = 'A_H2S'
    SERVER2HEAD_GRADIANT = 'G_S2H'
    SERVER2TAIL_ACTIVATION = 'A_S2T'
    TAIL2SERVER_GRADIANT = 'G_T2S'
    HEAD2TAIL_ACTIVATION = 'A_S2T'
    TAIL2HEAD_GRADIANT = 'G_S2H'
    
@dataclass
class SplitModelConfig:
    # def __init__(self, head_layer_num=1,server_layer_num=8, tail_layer_num=1,with_server=True,logicl_load=False):
    #     self.head_layer_num = head_layer_num
    #     self.server_layer_num = server_layer_num
    #     self.tail_layer_num = tail_layer_num
    #     self.with_server = with_server
    #     self.logicl_load=logicl_load #Ture表示拆分后模型其实还是用原先模型的参数，只是赋一个指针
    head_layer_num: int = field(default=2)
    server_layer_num: int = field(default=-1) #如果不设置会在拆分时自动根据头尾层数计算
    tail_layer_num: int = field(default=2)
    with_server: bool = field(default=True)
    logicl_load: bool = field(default=True)
    
    @property
    def total_hidden_layers(self):
        return self.head_layer_num+self.server_layer_num+self.tail_layer_num
