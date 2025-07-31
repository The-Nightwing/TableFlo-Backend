import pandas as pd
import recordlinkage
import numpy as np
from typing import Dict, List, Any, Tuple

class FuzzyMatcher:
    def __init__(self, request_data: Dict[str, Any]):
        """
        Initialize FuzzyMatcher with request data containing matching configuration
        """
        self.files = request_data.get('files', [])
        self.keys = request_data.get('keys', [])
        self.settings = request_data.get('settings', {})
        self.values = request_data.get('values', [])
        self.cross_reference = request_data.get('cross_reference', [])
        self.output_file = request_data.get('output_file', 'reconciliation_output')

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare dataframe by handling missing values
        """
        df = df.copy()
        df[df.select_dtypes(include=['object']).columns] = df.select_dtypes(include=['object']).fillna('')
        df[df.select_dtypes(include=['number']).columns] = df.select_dtypes(include=['number']).fillna(0)
        return df

    def process_keys(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process keys based on case sensitivity and special character handling
        """
        df1, df2 = df1.copy(), df2.copy()
        
        for key in self.keys:
            if key.get('case_sensitive') == 'no':
                df1[key['file1']] = df1[key['file1']].astype(str).str.upper()
                df2[key['file2']] = df2[key['file2']].astype(str).str.upper()

            if key.get('ignore_special') == 'yes':
                df1[key['file1']] = df1[key['file1']].astype(str).str.replace(r'[^a-zA-Z0-9]', '', regex=True)
                df2[key['file2']] = df2[key['file2']].astype(str).str.replace(r'[^a-zA-Z0-9]', '', regex=True)

        return df1, df2

    def handle_many_relations(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Handle many-to-many, one-to-many, or many-to-one relationships
        """
        left_keys = [key['file1'] for key in self.keys]
        right_keys = [key['file2'] for key in self.keys]

        if self.settings['method'] == 'one-to-many':
            agg_dict = {value['file2']: 'sum' for value in self.values}
            df2 = df2.groupby(right_keys).agg(agg_dict).reset_index()
            self._add_group_ids(df2, 'file2')

        elif self.settings['method'] == 'many-to-one':
            agg_dict = {value['file1']: 'sum' for value in self.values}
            df1 = df1.groupby(left_keys).agg(agg_dict).reset_index()
            self._add_group_ids(df1, 'file1')

        elif self.settings['method'] == 'many-to-many':
            agg_dict_left = {value['file1']: 'sum' for value in self.values}
            agg_dict_right = {value['file2']: 'sum' for value in self.values}
            
            df1 = df1.groupby(left_keys).agg(agg_dict_left).reset_index()
            df2 = df2.groupby(right_keys).agg(agg_dict_right).reset_index()
            
            self._add_group_ids(df1, 'file1')
            self._add_group_ids(df2, 'file2')

        return df1, df2

    def _add_group_ids(self, df: pd.DataFrame, file_key: str):
        """
        Add group IDs to dataframe for cross-referencing
        """
        for i, ref in enumerate(self.cross_reference):
            if ref[file_key] == 0:
                df['Group-ID'] = 'Group-' + (df.index + 1).astype(str)
                self.cross_reference[i][file_key] = 'Group-ID'

    def perform_matching(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Perform the fuzzy matching using recordlinkage
        """
        # Setup blocking keys
        left_block = [key['file1'] for key in self.keys if key['criteria'] == 'exact']
        right_block = [key['file2'] for key in self.keys if key['criteria'] == 'exact']

        # Create indexer and perform blocking
        indexer = recordlinkage.Index()
        indexer.block(left_on=left_block, right_on=right_block)
        comparisons = indexer.index(df1, df2)

        # Setup comparison
        compare = recordlinkage.Compare()
        for key in self.keys:
            if key['criteria'] == 'exact':
                compare.exact(key['file1'], key['file2'], 
                            label=f"{key['file1']}-{key['file2']}")
            elif key['criteria'] == 'fuzzy':
                compare.string(key['file1'], key['file2'], 
                             method='jarowinkler', 
                             threshold=0.85,
                             label=f"{key['file1']}-{key['file2']}")

        # Compute matches
        result = compare.compute(comparisons, df1, df2)
        return result.reset_index()

    def process_results(self, result: pd.DataFrame, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process matching results and update dataframes with reconciliation information
        """
        # Add cross-reference columns
        self._add_cross_reference_columns(df1, df2)
        
        # Process matches
        while len(result) > 0:
            self._process_single_match(result, df1, df2)
            result = result[
                (result['level_0'] != result['level_0'].iloc[0]) & 
                (result['level_1'] != result['level_1'].iloc[0])
            ]

        # Calculate differences and match status
        self._calculate_differences(df1, df2)
        
        return df1, df2

    def _add_cross_reference_columns(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """
        Add cross-reference columns to dataframes
        """
        for ref in self.cross_reference:
            if ref['file1'] == ref['file2']:
                df1[f"{ref['file2']}-2"] = np.nan
                df2[f"{ref['file1']}-1"] = np.nan
            else:
                df1[ref['file2']] = np.nan
                df2[ref['file1']] = np.nan

        for value in self.values:
            if value['file1'] == value['file2']:
                df1[f"{value['file2']}-2"] = np.nan
                df2[f"{value['file1']}-1"] = np.nan
            else:
                df1[value['file2']] = np.nan
                df2[value['file1']] = np.nan

        df1['Reco_Status'] = np.nan
        df2['Reco_Status'] = np.nan 