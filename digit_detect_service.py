# This is where you place the ASR codes
pip install tqdm==4.65.0
!pip install jiwer==3.0.1   
!pip install librosa==0.9.1
!pip install pandas==2.0.0rc

# download specific version of torch and torchaudio
!pip install torch==1.12.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116 

# we only take account torch and torchaudio library here
!pip list | grep torch

import os
import json
from tqdm import tqdm
from jiwer import wer, cer
from time import time
import pandas as pd
from typing import Tuple, Dict, List

import torch
import torch.nn.functional as F
import torchaudio

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import json

# setting the random seed for reproducibility
SEED = 2022


class CustomSpeechDataset(torch.utils.data.Dataset):
    
    """
    Custom torch dataset class to load the dataset 
    """
    
    def __init__(self, manifest_file: str, audio_dir: str, is_test_set: bool=False) -> None:

        """
        manifest_file: the json file that contains the filename of the audio, and also the annotation if is_test_set is set to False
        audio_dir: the root directory of the audio datasets
        is_test_set: the flag variable to switch between loading of the train and the test set. Train set loads the annotation whereas test set does not
        """

        self.audio_dir = audio_dir
        self.is_test_set = is_test_set

        with open(manifest_file, 'r') as f:
            self.manifest = json.load(f)

        
    def __len__(self) -> int:
        
        """
        To get the number of loaded audio files in the dataset
        """

        return len(self.manifest)
    
    
    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:

        """
        To get the values required to do the training
        """

        if torch.is_tensor(index):
            index.tolist()
            
        audio_path = self._get_audio_path(index)
        signal, sr = torchaudio.load(audio_path)
        
        if not self.is_test_set:
            annotation = self._get_annotation(index)
            return audio_path, signal, annotation
        
        return audio_path, signal
    
    
    def _get_audio_path(self, index: int) -> str:

        """
        Helper function to retrieve the audio path from the json manifest file
        """
        
        path = os.path.join(self.audio_dir, self.manifest[index]['path'])

        return path
    
    
    def _get_annotation(self, index: int) -> str:

        """
        Helper function to retrieve the annotation from the json manifest file
        """

        return self.manifest[index]['annotation']
        

class TextTransform:

    """
    Map characters to integers and vice versa (encoding/decoding)
    """
    
    def __init__(self) -> None:

        char_map_str = """
            <SPACE> 0
            A 1
            B 2
            C 3
            D 4
            E 5
            F 6
            G 7
            H 8
            I 9
            J 10
            K 11
            L 12
            M 13
            N 14
            O 15
            P 16
            Q 17
            R 18
            S 19
            T 20
            U 21
            V 22
            W 23
            X 24
            Y 25
            Z 26
        """
        
        self.char_map = {}
        self.index_map = {}
        
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch

        self.index_map[0] = ' '


    def get_char_len(self) -> int:

        """
        Gets the number of characters that are being encoded and decoded in the prediction
        Returns:
        --------
            the number of characters defined in the __init__ char_map_str
        """

        return len(self.char_map)
    

    def get_char_list(self) -> List[str]:

        """
        Gets the list of characters that are being encoded and decoded in the prediction
        
        Returns:
        -------
            a list of characters defined in the __init__ char_map_str
        """

        return list(self.index_map.values())
    

    def text_to_int(self, text: str) -> List[int]:

        """
        Use a character map and convert text to an integer sequence 
        Returns:
        -------
            a list of the text encoded to an integer sequence 
        """
        
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)

        return int_sequence
    

    def int_to_text(self, labels) -> str:

        """
        Use a character map and convert integer labels to an text sequence 
        
        Returns:
        -------
            the decoded transcription
        """
        
        string = []
        for i in labels:
            string.append(self.index_map[i])

        return ''.join(string).replace('<SPACE>', ' ')
        
        
class GreedyDecoder:

    """
    Decodes the logits into characters to form the final transciption using the greedy decoding approach
    """

    def __init__(self) -> None:
        pass


    def decode(
            self, 
            output: torch.Tensor, 
            labels: torch.Tensor=None, 
            label_lengths: List[int]=None, 
            collapse_repeated: bool=True, 
            is_test: bool=False
        ):
        
        """
        Main method to call for the decoding of the text from the predicted logits
        """
        
        text_transform = TextTransform()
        arg_maxes = torch.argmax(output, dim=2)
        decodes = []

        # refer to char_map_str in the TextTransform class -> only have index from 0 to 26, hence 27 represents the case where the character is decoded as blank (NOT <SPACE>)
        decoded_blank_idx = text_transform.get_char_len()

        if not is_test:
            targets = []

        for i, args in enumerate(arg_maxes):
            decode = []

            if not is_test:
                targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))

            for j, char_idx in enumerate(args):
                if char_idx != decoded_blank_idx:
                    if collapse_repeated and j != 0 and char_idx == args[j-1]:
                        continue
                    decode.append(char_idx.item())
            decodes.append(text_transform.int_to_text(decode))

        return decodes, targets if not is_test else decodes
        

class DataProcessor:

    """
    Transforms the audio waveform tensors into a melspectrogram
    """

    def __init__(self) -> None:
        pass
    
    
    def _audio_transformation(self, is_train: bool=True):

        return torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
                torchaudio.transforms.TimeMasking(time_mask_param=100)
            ) if is_train else torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
    

    def data_processing(self, data, data_type='train'):

        """
        Process the audio data to retrieve the spectrograms that will be used for the training
        """

        text_transform = TextTransform()
        spectrograms = []
        input_lengths = []
        audio_path_list = []

        audio_transforms = self._audio_transformation(is_train=True) if data_type == 'train' else self._audio_transformation(is_train=False)

        if data_type != 'test':  
            labels = []
            label_lengths = []

            for audio_path, waveform, utterance in data:

                spec = audio_transforms(waveform).squeeze(0).transpose(0, 1)
                spectrograms.append(spec)
                label = torch.Tensor(text_transform.text_to_int(utterance))
                labels.append(label)
                input_lengths.append(spec.shape[0]//2)
                label_lengths.append(len(label))

            spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
            return audio_path, spectrograms, labels, input_lengths, label_lengths

        else:
            for audio_path, waveform in data:

                spec = audio_transforms(waveform).squeeze(0).transpose(0, 1)
                spectrograms.append(spec)
                input_lengths.append(spec.shape[0]//2)
                audio_path_list.append(audio_path)

            spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
            return audio_path_list, spectrograms, input_lengths
            
            
class CNNLayerNorm(torch.nn.Module):
    
    """
    Layer normalization built for CNNs input
    """
    
    def __init__(self, n_feats: int) -> None:
        super(CNNLayerNorm, self).__init__()

        self.layer_norm = torch.nn.LayerNorm(n_feats)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x of dimension -> (batch, channel, feature, time)
        """
        
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)

        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 


class ResidualCNN(torch.nn.Module):

    """
    Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf except with layer norm instead of batch norm
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, dropout: float, n_feats: int) -> None:
        super(ResidualCNN, self).__init__()

        self.cnn1 = torch.nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = torch.nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Model building for the Residual CNN layers
        
        Input x of dimension -> (batch, channel, feature, time)
        """

        residual = x
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual

        return x # (batch, channel, feature, time)


class BidirectionalGRU(torch.nn.Module):

    """
    The Bidirectional GRU composite code block which will be used in the main SpeechRecognitionModel class
    """
    
    def __init__(self, rnn_dim: int, hidden_size: int, dropout: int, batch_first: int) -> None:
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = torch.nn.GRU(
            input_size=rnn_dim, 
            hidden_size=hidden_size,
            num_layers=1, 
            batch_first=batch_first, 
            bidirectional=True
        )
        self.layer_norm = torch.nn.LayerNorm(rnn_dim)
        self.dropout = torch.nn.Dropout(dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Transformation of the layers in the Bidirectional GRU block
        """

        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)

        return x


class SpeechRecognitionModel(torch.nn.Module):

    """
    The main ASR Model that the main code will interact with
    """
    
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1) -> None:
        super(SpeechRecognitionModel, self).__init__()
        
        n_feats = n_feats//2
        self.cnn = torch.nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = torch.nn.Sequential(*[
            ResidualCNN(
                in_channels=32, 
                out_channels=32, 
                kernel=3, 
                stride=1, 
                dropout=dropout, 
                n_feats=n_feats
            ) for _ in range(n_cnn_layers)
        ])
        self.fully_connected = torch.nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = torch.nn.Sequential(*[
            BidirectionalGRU(
                rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                hidden_size=rnn_dim, 
                dropout=dropout, 
                batch_first=i==0
            ) for i in range(n_rnn_layers)
        ])
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(rnn_dim, n_class)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Transformation of the layers in the ASR model block
        """

        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        
        return x
        
        
        
def infer(hparams, test_dataset, model_path) -> Dict[str, str]:
    
    print('\ngenerating inference ...')

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(SEED)
    
    greedy_decoder = GreedyDecoder()
    data_processor = DataProcessor()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=lambda x: data_processor.data_processing(x, 'test'),
        **kwargs
    )
    
    # load the pretrained model
    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], 
        hparams['n_rnn_layers'], 
        hparams['rnn_dim'],
        hparams['n_class'], 
        hparams['n_feats'], 
        hparams['stride'], 
        hparams['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    output_dict = {}
    
    with torch.no_grad():
        for i, _data in tqdm(enumerate(test_loader)):
            audio_path, spectrograms, input_lengths = _data
            spectrograms = spectrograms.to(device)
            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class) 
            decoded_preds_batch = greedy_decoder.decode(output.transpose(0, 1), labels=None, label_lengths=None, is_test=True)
            
            # batch prediction
            for decoded_idx in range(len(decoded_preds_batch[0])):
                output_dict[audio_path[decoded_idx]] = decoded_preds_batch[0][decoded_idx]
                
    print('done!\n')
    return output_dict
    

class DigitDetectionService():
    '''Implementation of the Digit Detection Service based on Automatic Speech Recognition.
    '''
    def __init__(self, model_dir:str):
    
    	self.model_dir = model_dir

        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        pass

    def check_digits(word_list):
    	numbers_dict = {'1' : 'ONE', '2': 'TWO', '3' : 'THREE', '4' : 'FOUR', '5' : 'FIVE', '6' : 'SIX'. '7' : 'SEVEN'. '8' : 'EIGHT', '9' : 'NINE'}

	return ([k for k, v in numbers_dict.items() if v in word_list])
    
    def transcribe_audio_to_digits(self, audio_dir) -> Tuple[int]:
    	#create manifest file (use python json library)
    	results_digits = []
    	manifest_dict = {}
    	manifest_file = 'manifest.json'
    	for item in os.scandir(audio_dir):
    		manifest_dict['path'] = item
	with open(manifest_file, 'w') as outfile:
    		json.dump(manifest_dict, outfile)
    	
    	#load test dataset (manifest, audio dir, test=true)
    	dataset_test = CustomSpeechDataset(manifest_file=manifest_file,   
					    audio_dir=audio_dir, 
					    is_test_set=True)
    	#run the infer function (hparams, dataset, self.model_dir) -> dictionary 
    	hparams = {
            "n_cnn_layers": 5,
            "n_rnn_layers": 5,
            "rnn_dim": 512,
            "n_class": 28, # 26 alphabets in caps + <SPACE> + blanks
            "n_feats": 128,
            "stride": 2,
            "dropout": 0.1,
            "learning_rate": 1.04e-4,
            "batch_size": 5,
            "epochs": 300
	}
    	results = infer(hparams, dataset_test, self.model_dir)
    	#{'audio1.wav':[i am gay]}
    	
    	#run through dictionary, extract out 1 digit from each file -> tuple(8,9)
    	sorted_fnames = sorted(results.keys())
    	for fname in sorted_fnames:
    		sentence = results[fname]
    		word_list = sentence.split()
    		digit = check_digits(word_list)[0] if check_digits(word_list) else None #first digit is taken
    		results_digits.append(digit)
	results_digits = tuple(results_digits)
	
	return results_digits
    	
    
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
       
        """

