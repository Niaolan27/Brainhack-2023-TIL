#this is where you place the speaker codes
from abstract_ai_services import AbstractDigitDetectionService, AbstractSpeakerIDService, AbstractObjectReIDService
from tilsdk.cv.types import *
from typing import Tuple, List
import numpy as np



class SpeakerIDService():
    '''Implementation of the Speaker ID service.
    '''
    def __init__(self, model_dir:str):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        pass
    
    def identify_speaker(self, audio_waveform: np.array, sampling_rate: int) -> str:
        '''
        Parameters
        ----------
        audio_waveform : np.array
            input waveform.
        sampling_rate : int
            the sampling rate of the audio file.
            
        Returns
        -------
        result : str
            string representing the speaker's ID corresponding to the list of speaker IDs in the training data set.
        '''
        if audio_waveform[0] == 1:
            return "TeamName1_Member1"
        else:
            return "TeamName2_Member3"


