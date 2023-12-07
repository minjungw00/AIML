import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

class ClassificationModel:
    def __init__(self):
        self._model = None
        
    @property
    def features(self):
        return self._features
    
    @features.setter
    def features(self, features):
        self._features = features

    
    def train(self, df):
        X = df[self._features]
        y = df["성별"]

        self._model.fit(X, y)
    
    
    def predict(self, df):
        X = df[self._features]
        pred = self._model.predict(X)
        return pred
        

    

class Model01(ClassificationModel):
    def __init__(self):
        self._model = MLPClassifier(random_state=9756)
    
        self._features = [
            '키',
'눈높이',
'목뒤높이',
'어깨높이',
'어깨가쪽높이',
'겨드랑높이',
'(팔굽힌)팔꿈치높이',
'엉덩이높이',
'주먹높이',
'허리높이',
'배꼽수준허리높이',
'위앞엉덩뼈가시높이',
'정강뼈위점높이',
'샅높이',
'허리너비',
'배꼽수준허리너비',
'엉덩이너비',
'겨드랑두께',
'허리두께',
'배꼽수준허리두께',
'엉덩이두께',
'몸통수직길이',
'엉덩이수직길이',
'체중(몸무게)',
'앞중심길이',
'배꼽수준앞중심길이',
'겨드랑앞벽사이길이',
'겨드랑앞접힘사이길이',
'목둘레',
'목밑둘레',
'허리둘레',
'배꼽수준허리둘레',
'배둘레',
'엉덩이둘레',
'배돌출점기준엉덩이둘레',
'어깨길이',
'목뒤등뼈위겨드랑수준길이',
'등길이',
'배꼽수준등길이',
'넙다리직선길이',
'어깨사이길이',
'어깨가쪽사이길이',
'겨드랑뒤벽사이길이',
'겨드랑뒤접힘사이길이',
'위팔길이(어깨점)',
'팔길이',
'팔안쪽길이',
'겨드랑둘레',
'엉덩이옆길이',
'다리가쪽길이',
'몸통세로둘레',
'샅앞뒤길이',
'배꼽수준샅앞뒤길이',
'앉은키',
'앉은눈높이',
'앉은목뒤높이',
'앉은어깨높이',
'앉은팔꿈치높이',
'앉은넙다리높이',
'앉은무릎높이',
'앉은오금높이',
'앉은엉덩이무릎수평길이',
'앉은엉덩이오금수평길이',
'앉은엉덩이배두께',
'(팔굽힌)위팔수직길이',
'(팔굽힌)아래팔수평길이',
'(팔굽힌)팔꿈치손끝수평길이',
'(팔굽힌)팔꿈치주먹수평길이',
'앉은배두께',
'어깨너비',
'위팔사이너비',
'팔꿈치사이너비',
'앉은엉덩이너비',
'머리수직길이',
'얼굴수직길이/코뿌리-턱끝수직길이',
'벽면앞으로뻗은주먹수평길이',
'벽면어깨수평길이',
'눈동자사이너비',
'눈구석사이너비',
'눈살눈확아래사이수직길이',
'손직선길이',
'손바닥직선길이',
'손너비/손안쪽가쪽직선길이',
'검지손가락길이',
'검지손가락중간마디너비',
'검지손가락끝마디너비',
'손두께',
'막대쥔손안둘레',
'손둘레',
'머리둘레',
'귀구슬사이머리위길이',
'눈살-머리마루-뒤통수길이',
'머리두께/눈살-뒤통수돌출직선길이',
'머리너비',
'얼굴너비',
'아래턱사이너비',
'발너비',
'발직선길이',
'가쪽복사높이',
'넙다리둘레',
'넙다리중간둘레',
'무릎둘레',
'무릎아래둘레',
'장딴지둘레',
'종아리최소둘레',
'발목최대둘레',
'발꿈치너비',
'(편)위팔둘레',
'(편)팔꿈치둘레',
'손목둘레',
'머리위로뻗은주먹높이'
        ]
        
    # def train(self, df):
    #   You can override this function...
    
    
    # def predict(self, df):
    #   You can override this function...    
    
    
    
class Model02(ClassificationModel):
    def __init__(self):
        self._model = GaussianNB()
    
        self._features = [
            "눈높이",
            "목뒤높이",
            "손목둘레"
        ]
        
    # def train(self, df):
    #   You can override this function...
    
    
    # def predict(self, df):
    #   You can override this function...    
    
    
    

class Model03(ClassificationModel):
    def __init__(self):
        self._model = GaussianNB()
    
        self._features = [
            "눈높이",
            "목뒤높이",
            "손목둘레"
        ]
        
    # def train(self, df):
    #   You can override this function...
    
    
    # def predict(self, df):
    #   You can override this function...    
    
    