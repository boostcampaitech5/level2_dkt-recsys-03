from features.feature_manager import FeatureManager
from features.assessments import AssessmentNum, UserSolvedCnt, RepeatCnt
from features.tags import TagExposureCnt
from features.tests import TestNum, TestType
from features.timestamps import Month, Hour, DayOfWeek
from features.solving_time import SolvingTime
from features.matrix_factorization import AssessmentLatentFactor


def feature_manager(csv_path: str) -> FeatureManager:
    fm = FeatureManager(csv_path, feature_processors=[
        AssessmentNum(),
        UserSolvedCnt(),
        RepeatCnt(),
        
        TagExposureCnt(),
        
        TestNum(),
        TestType(),
        
        Month(),
        Hour(),
        DayOfWeek(),
        
        SolvingTime(),
        AssessmentLatentFactor()
    ])
    
    
    if fm.need_feature_creation():
        # 여기서 피쳐 생성 여부를 자동으로 감지~
        # 이걸로 그냥 자동화 할 수 있지 않을까?
        # 예를 들면 prepare_df할때 자동으로 생성 후 로드 한다던가..
        pass
    
    return fm