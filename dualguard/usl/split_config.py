from dataclasses import dataclass, field


class Intermediate(object):
    HEAD2SERVER_ACTIVATION = 'A_H2S'
    SERVER2HEAD_GRADIANT = 'G_S2H'
    SERVER2TAIL_ACTIVATION = 'A_S2T'
    TAIL2SERVER_GRADIANT = 'G_T2S'
    HEAD2TAIL_ACTIVATION = 'A_S2T'
    TAIL2HEAD_GRADIANT = 'G_S2H'


@dataclass
class SplitModelConfig:
    head_layer_num: int = field(default=2)
    server_layer_num: int = field(default=-1)
    tail_layer_num: int = field(default=2)
    with_server: bool = field(default=True)
    logicl_load: bool = field(default=True)

    @property
    def total_hidden_layers(self):
        return self.head_layer_num + self.server_layer_num + self.tail_layer_num
