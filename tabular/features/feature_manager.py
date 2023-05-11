import pandas as pd
from itertools import product
from typing import List, Union
from features.preprocessor import PreProcessor
from os import path


class FeatureManager:
    
    def __init__(self, csv_path: str = './features.csv', feature_processors: List[PreProcessor] = []) -> None:
        self.csv_path = csv_path
        self.feature_processors = feature_processors
    
    
    ####### 피쳐 업데이트 확인
    def need_feature_creation(self, is_train: bool = True):
        '''
        피쳐 생성이 필요한가?
        
        True: 피쳐가 추가된 경우, 피쳐 csv가 없는 경우 
        False: 피쳐 csv가 이미 있고, 현재 최신상태인 경우
        '''
        if is_train:
            csv_path_ = self.__train_feature_csv_path()
        else:
            csv_path_ = self.__test_feature_csv_path()
        
        if not path.exists(csv_path_):
            return True
        if not path.exists(f'{csv_path_}.chk'):
            return True
        
        #### feature_processors로 해시 값을 구한뒤, 이 값으로 업데이트 여부를 판단
        curr_process_col_v = str(self.feature_columns())
        with open(f'{csv_path_}.chk', 'r') as chk_f:
            col_v = chk_f.readline().rstrip()
            
            if col_v != curr_process_col_v:
                return True
            
        return False
    
    
    ####### 피쳐 생성
    def create_features(self, df_: pd.DataFrame, isTrain: bool = True) -> pd.DataFrame:
        '''
        피쳐를 생성함.
        '''
        df = df_.copy(deep=True)
        
        for processor in self.feature_processors:
            col = processor.columns()
            opt = processor.options()
            
            ### 여러개의 컬럼을 만드는 경우
            if type(col) is list:
                cols = col
                
                ### 옵션이 없는 경우
                if not opt:
                    series_list = processor.create_feature(df)
                    for i, series in enumerate(series_list):
                        df.loc[:, cols[i]] = series
                    continue
                
                ### 옵션이 있는 경우, 컬럼#인자 로 저장
                keys = list(opt.keys())
                values_by_key = list(opt.values())
                for args in product(*values_by_key):
                    kwargs = {key: values for key, values in zip(keys, args)}
                    series_list = processor.create_feature(df, **kwargs)
                    
                    for i, series in enumerate(series_list):
                        col_ = self.__col_opt(cols[i], args)
                        df.loc[:, col_] = series
                
                continue
            
            ### 옵션이 없는 경우
            if not opt:
                df.loc[:, col] = processor.create_feature(df)
                continue
            
            ### 옵션이 있는 경우, 컬럼#인자 로 저장
            keys = list(opt.keys())
            values_by_key = list(opt.values())
            for args in product(*values_by_key):
                kwargs = {key: values for key, values in zip(keys, args)}
                
                col_ = self.__col_opt(col, args)
                df.loc[:, col_] = processor.create_feature(df, **kwargs)
                
        self.__dump_features_to_csv(df, isTrain)
        return df
    
    
    def __dump_features_to_csv(self, df: pd.DataFrame, is_train: bool = True) -> None:
        if is_train:
            csv_path_ = self.__train_feature_csv_path()
        else:
            csv_path_ = self.__test_feature_csv_path()
        
        df.to_csv(csv_path_)
        
        with open(f'{csv_path_}.chk', 'w') as chk_f:
            curr_process_col_v = str(self.feature_columns())
            chk_f.write(curr_process_col_v)
    
    
    ####### 피쳐 로딩
    def prepare_df(self, option: dict, selected_columns: list[str], df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        '''피쳐 목록 선택'''
        f_df = self.__load_feature_df(is_train)
        
        for processor in self.feature_processors:
            col = processor.columns()
            
            if type(col) == list:
                cols = col
                
                opt = processor.options()
            
                if not opt:
                    for col_ in cols:
                        if col_ not in selected_columns:
                            continue
                        df.loc[:, col_] = f_df[col_]
                    continue
                
                arg_keys = list(processor.options().keys())
                arg_vals = [option[arg_key] for arg_key in arg_keys]
                
                for col_ in cols:
                    if col_ not in selected_columns:
                        continue
                    
                    col__ = self.__col_opt(col_, arg_vals)
                    df.loc[:, col_] = f_df[col__]
                
                continue
            
            # 선택된 컬럼만 업데이트 함
            if col not in selected_columns:
                continue
            
            opt = processor.options()
            
            if not opt:
                df.loc[:, col] = f_df[col]
                continue
            
            arg_keys = list(processor.options().keys())
            arg_vals = [option[arg_key] for arg_key in arg_keys]
            
            col_ = self.__col_opt(col, arg_vals)
            df.loc[:, col] = f_df[col_]
            
        return df
    
    
    def feature_columns(self) -> list[str]:
        return [p.columns() for p in self.feature_processors]
    
    
    def __load_feature_df(self, is_train: bool = True) -> pd.DataFrame:
        if is_train:
            csv_path_ = self.__train_feature_csv_path()
        else:
            csv_path_ = self.__test_feature_csv_path()
        
        f_df = pd.read_csv(csv_path_, parse_dates=['Timestamp'])
        return f_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
    
    
    def __col_opt(self, col: str, args: list) -> str:
        args = tuple(args)
        return f'{col}#{args}'
    
    
    def __test_feature_csv_path(self) -> str:
        tokens = self.csv_path.split('.')
        return '.'.join(tokens[:-1] + ['test'] + [tokens[-1]])
    
    
    def __train_feature_csv_path(self) -> str:
        tokens = self.csv_path.split('.')
        return '.'.join(tokens[:-1] + ['train'] + [tokens[-1]])