"""
Feature Engineering Module

Creates features for inverse safety modeling from crash data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Engineers features for crash analysis and safety modeling.
    
    Creates features in categories:
    - Temporal features (time, day, season)
    - Environmental features (weather, lighting)
    - Location features (road type, area, speed limit)
    - VRU features (pedestrian/cyclist specific)
    - Driver/Vehicle features
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        pass
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: DataFrame with MONTH, DAY_WEEK, HOUR columns
            
        Returns:
            DataFrame with additional temporal features
        """
        df = df.copy()
        
        # Hour of day features
        if 'HOUR' in df.columns:
            df['HOUR'] = pd.to_numeric(df['HOUR'], errors='coerce')
            df['IS_RUSH_HOUR'] = df['HOUR'].isin([7, 8, 9, 16, 17, 18]).astype(int)
            df['IS_NIGHT'] = df['HOUR'].isin(range(20, 24)) | df['HOUR'].isin(range(0, 6))
            df['IS_NIGHT'] = df['IS_NIGHT'].astype(int)
        
        # Day of week features
        if 'DAY_WEEK' in df.columns:
            df['DAY_WEEK'] = pd.to_numeric(df['DAY_WEEK'], errors='coerce')
            df['IS_WEEKEND'] = df['DAY_WEEK'].isin([1, 7]).astype(int)  # 1=Sunday, 7=Saturday
        
        # Season features
        if 'MONTH' in df.columns:
            df['MONTH'] = pd.to_numeric(df['MONTH'], errors='coerce')
            df['SEASON'] = df['MONTH'].apply(self._get_season)
        
        logger.info("Created temporal features")
        return df
    
    @staticmethod
    def _get_season(month: int) -> str:
        """Map month to season."""
        if pd.isna(month):
            return 'Unknown'
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def create_environmental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather and lighting features.
        
        Args:
            df: DataFrame with weather and lighting columns
            
        Returns:
            DataFrame with environmental features
        """
        df = df.copy()
        
        # Weather features
        if 'WEATHER' in df.columns:
            # Adverse weather indicator
            adverse_weather_codes = [2, 3, 4, 5, 10, 11, 12]  # Rain, snow, fog, etc.
            df['ADVERSE_WEATHER'] = df['WEATHER'].isin(adverse_weather_codes).astype(int)
        
        # Lighting features
        if 'LGT_COND' in df.columns:
            # Poor lighting indicator
            poor_lighting_codes = [2, 3, 4]  # Dark (not lighted, lighted, unknown)
            df['POOR_LIGHTING'] = df['LGT_COND'].isin(poor_lighting_codes).astype(int)
        
        # Combined adverse conditions
        if 'ADVERSE_WEATHER' in df.columns and 'POOR_LIGHTING' in df.columns:
            df['ADVERSE_CONDITIONS'] = (df['ADVERSE_WEATHER'] & df['POOR_LIGHTING']).astype(int)
        
        logger.info("Created environmental features")
        return df
    
    def create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create location-based features.
        
        Args:
            df: DataFrame with road and location columns
            
        Returns:
            DataFrame with location features
        """
        df = df.copy()
        
        # Road type features
        if 'ROUTE' in df.columns:
            # Interstate/highway indicator
            df['IS_INTERSTATE'] = df['ROUTE'].isin([1]).astype(int)
        
        # Urban/rural
        if 'RUR_URB' in df.columns:
            df['IS_URBAN'] = (df['RUR_URB'] == 1).astype(int)
        
        # Speed limit features
        if 'SPD_LIM' in df.columns:
            df['SPD_LIM'] = pd.to_numeric(df['SPD_LIM'], errors='coerce')
            df['HIGH_SPEED_ROAD'] = (df['SPD_LIM'] >= 55).astype(int)
            df['LOW_SPEED_ROAD'] = (df['SPD_LIM'] <= 35).astype(int)
        
        # Traffic control device
        if 'TYP_INT' in df.columns:
            # Intersection type
            df['HAS_TRAFFIC_SIGNAL'] = df['TYP_INT'].isin([1]).astype(int)
        
        logger.info("Created location features")
        return df
    
    def create_vru_features(self, accident_df: pd.DataFrame, person_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create VRU-specific features.
        
        Args:
            accident_df: Accident-level data
            person_df: Person-level data
            
        Returns:
            DataFrame with VRU features added
        """
        df = accident_df.copy()
        
        # Count pedestrians and cyclists per crash
        vru_counts = person_df[person_df['PER_TYP'].isin([5, 6])].groupby('CASENUM').agg({
            'PER_TYP': lambda x: {
                'pedestrian_count': (x == 5).sum(),
                'cyclist_count': (x == 6).sum(),
                'total_vru': len(x)
            }
        })
        
        # Flatten the nested dict
        vru_counts_df = pd.DataFrame(vru_counts['PER_TYP'].tolist(), index=vru_counts.index)
        
        # Merge with accident data
        df = df.merge(vru_counts_df, left_on='CASENUM', right_index=True, how='left')
        df[['pedestrian_count', 'cyclist_count', 'total_vru']] = df[
            ['pedestrian_count', 'cyclist_count', 'total_vru']
        ].fillna(0)
        
        # VRU injury severity (from person data)
        if 'INJ_SEV' in person_df.columns:
            vru_severity = person_df[person_df['PER_TYP'].isin([5, 6])].groupby('CASENUM')['INJ_SEV'].agg([
                ('max_vru_injury', 'max'),
                ('fatal_vru', lambda x: (x == 4).any().astype(int))
            ])
            df = df.merge(vru_severity, left_on='CASENUM', right_index=True, how='left')
        
        logger.info("Created VRU features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between variables.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # Night + Poor lighting
        if 'IS_NIGHT' in df.columns and 'POOR_LIGHTING' in df.columns:
            df['NIGHT_AND_DARK'] = (df['IS_NIGHT'] & df['POOR_LIGHTING']).astype(int)
        
        # Urban + High speed
        if 'IS_URBAN' in df.columns and 'HIGH_SPEED_ROAD' in df.columns:
            df['URBAN_HIGH_SPEED'] = (df['IS_URBAN'] & df['HIGH_SPEED_ROAD']).astype(int)
        
        # Weekend + Night
        if 'IS_WEEKEND' in df.columns and 'IS_NIGHT' in df.columns:
            df['WEEKEND_NIGHT'] = (df['IS_WEEKEND'] & df['IS_NIGHT']).astype(int)
        
        logger.info("Created interaction features")
        return df
    
    def engineer_features_pipeline(
        self, 
        accident_df: pd.DataFrame, 
        person_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            accident_df: Accident-level data
            person_df: Person-level data (optional, for VRU features)
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("\n=== Starting Feature Engineering Pipeline ===")
        
        df = accident_df.copy()
        
        # Create features
        df = self.create_temporal_features(df)
        df = self.create_environmental_features(df)
        df = self.create_location_features(df)
        
        if person_df is not None:
            df = self.create_vru_features(df, person_df)
        
        df = self.create_interaction_features(df)
        
        logger.info(f"\nFeature engineering complete")
        logger.info(f"Total features: {len(df.columns)}")
        logger.info(f"Records: {len(df):,}")
        
        return df


if __name__ == "__main__":
    # Example usage
    from data_loader import CRSSDataLoader
    from preprocessing import CrashPreprocessor
    
    loader = CRSSDataLoader()
    datasets = loader.load_complete_dataset()
    
    preprocessor = CrashPreprocessor()
    processed = preprocessor.preprocess_pipeline(datasets)
    
    engineer = FeatureEngineer()
    featured_df = engineer.engineer_features_pipeline(
        processed['accident'], 
        processed['person']
    )
    
    print("\nFeature Engineering Complete!")
    print(f"Features created: {len(featured_df.columns)}")
