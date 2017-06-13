from GameTheory_Game.ConstantSumGame import ConstantSumGame
import numpy as np


class ZerosumGame(ConstantSumGame):
    def __init__(self, maximum_int=10, minimum_int=-10, lin=np.random.randint(2, 5), col=np.random.randint(2, 5)):
        self.__c = 0
        ConstantSumGame.__init__(self, self.__c, maximum_int=maximum_int, minimum_int=minimum_int, lin=lin, col=col)
