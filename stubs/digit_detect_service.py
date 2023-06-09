# This is where you place the ASR codes




class DigitDetectionService():
    '''Implementation of the Digit Detection Service based on Automatic Speech Recognition.
    '''
    def __init__(self, model_dir:str):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        pass

    def transcribe_audio_to_digits(self, audio_waveform: np.array) -> Tuple[int]:
        '''Transcribe audio waveform to a tuple of ints.
        
        Parameters
        ----------
        audio_waveform : numpy.array
            Numpy array of floats that represent the audio file. It is assumed that the sampling rate of the audio is 16K.
        Returns
        -------
        results  :
            The ordered tuple of digits found in the input audio file.
        '''
        return (1,2)  # mock value
