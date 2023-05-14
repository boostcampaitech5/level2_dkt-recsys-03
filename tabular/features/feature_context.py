from features.feature_manager import FeatureManager
from features.assessments import AssessmentNum, UserSolvedCnt, RepeatCnt
from features.tags import TagExposureCnt
from features.tests import TestNum, TestType
from features.timestamps import Month, Hour, DayOfWeek
from features.rolling_time import RollingTime
from features.memory import UserAcc, UserItemAcc, UserTag1Acc, UserTag2Acc, UserLastTag1Correct, UserLastTag2Correct


def feature_manager(csv_path: str) -> FeatureManager:
    """
    FeatureProcssor 객체 주입
    """
    fm = FeatureManager(csv_path, feature_processors=[
        # assessments.py
        AssessmentNum(),
        UserSolvedCnt(),
        RepeatCnt(),
        # tags.py
        TagExposureCnt(),
        # tests.py
        TestNum(),
        TestType(),
        # timestamps.py
        Month(),
        Hour(),
        DayOfWeek(),
        # rolling_time.py
        RollingTime(),
        # memory.py
        UserAcc(),
        UserItemAcc(),
        UserTag1Acc(),
        UserTag2Acc(),
        UserLastTag1Correct(),
        UserLastTag2Correct()
    ])
    
    return fm