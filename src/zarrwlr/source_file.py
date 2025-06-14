from dataclasses import dataclass
import subprocess
import json
import pathlib
import hashlib
from typing import Dict, List, Any, Optional
from .utils import safe_int_conversion, safe_float_conversion, can_ffmpeg_decode_codec

# import and initialize logging
from .logsetup import get_module_logger
logger = get_module_logger(__file__)
logger.trace("Module loading...")

# ############################################################
# ############################################################
#
# Class FileBaseParameters
# ========================
# 
# Main audio file parameters for import processing
#
@dataclass
class FileBaseParameters:
    """Main audio file parameters for import processing."""
    file: pathlib.Path = None
    file_size_bytes: int|None = None
    file_sh256: str|None = None
    container_format_name: str|None = None
    selected_audio_stream_list: list[int]|None = None
    nb_channels_in_stream_list: list[int]|None = None
    total_nb_of_channels = 0
    sample_rate: int|None = None
    sample_format: str|None = None
    codec_name: str|None = None
    nb_samples: int|None = None 
    
    def __post_init__(self):
        if self.audio_stream_list is None:
            self.audio_stream_list = []
        if self.nb_channels_in_stream_list is None:
            self.nb_channels_in_stream_list = []

#
# End of Class FileBaseParameters
#
# ############################################################
# ############################################################



# ############################################################
# ############################################################
#
# Class FileParameter
# ===================
#
# Analysis of the audio source file to set it free for import
#

class FileParameter:
    """
    Vollständige Parameter-Analyse einer Audio-Datei
    
    Strukturiert alle Informationen hierarchisch:
    - Container-Level: Format, Dauer, etc.
    - Audio-Streams: Pro Audio-Stream (Codec, Sample-Rate, etc.)
    - Other-Streams: Nicht-Audio-Streams (Video, Subtitle, etc.)
    - General-Meta: Alle nicht-technischen Metadaten (Tags, etc.)
    """
    
    # ################################################
    #
    # Initialization part
    # -------------------
    #
    def __init__(self, file_path: str | pathlib.Path, user_meta: dict = {}, target_format: str = 'flac', audio_stream_selection_list: int|list[int] = [0]):
        """
        Initialisiert FileParameter mit vollständiger ffprobe-Analyse
        
        Args:
            file_path: Pfad zur zu analysierenden Audio-Datei
            user_meta: Zusätzliche benutzerdefinierte Metadaten
        """
        logger.trace(f"Initializing a file parameter instance requested with file_path: {str(file_path)}. See following steps.")
        
        # Parameter validation and Initialisation of variables
        logger.trace("Step 1: Validate arguments...")
        self._base_parameter = FileBaseParameters()
        
        file_path = pathlib.Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio-Datei nicht gefunden: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Pfad ist keine Datei: {file_path}")
        self._base_parameter.file = pathlib.Path(file_path).resolve()
        self._base_parameter.file_size_bytes = self._base_parameter.file.stat().st_size
           
        if not self._validate_and_save_user_meta(user_meta):
            self._user_meta:dict = {}
            
        if not self._validate_and_save_target_format(target_format):
            self._target_format:str = ""
        
        if not self._validate_and_save_audio_stream_selection_list(audio_stream_selection_list):
            self._base_parameter.selected_audio_stream_list = [0]
        logger.trace("Step 1 Done: Validate arguments.")
        
        logger.trace("Step 2: Parameter initialization...")
        # Initialization of other parameters
        self._base_parameter.selected_audio_stream_list = [0]
        self._can_be_imported = False # Can not be True until analyzein was run.
        
        # Initialisierung der Haupt-Datenstrukturen
        self._general_meta: dict = {}
        self._container: dict = {}
        self._audio_streams: List[dict] = []
        self._other_streams: List[dict] = []
        logger.trace("Step 2 Done: Parameter initialization.")
        
        logger.trace("Step 3: Start to calculate the hash of the file data...")
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        self._base_parameter.file_sh256 = hasher.hexdigest()
        logger.trace("Step 3 Done: Hash creation.")
        
        # Vollständige Daten per ffprobe einlesen
        logger.trace("Step 4: Extract all file information...")
        self._extract_file_data()
        logger.trace("Step 4 done: Extract all file information.")
        
        # set non-Stream depending values
        self._base_parameter.container_format_name = self._container["format_name"]
        
        # do a first analysis by using default configuration
        logger.trace("Step 5: Base analysation of parameters...")
        self._analyze(self)
        logger.trace("Step 5 done: Base analysation of parameters. First report can be requested.")
    
    def _validate_and_save_user_meta(self, user_meta: dict|None):
        if not isinstance(user_meta, dict|None):
            raise ValueError("Additional information, given as user_meta, must be structured as dictionary.")
        if user_meta is not None:
            logger.trace(f"user_meta value accepted: {user_meta}")
            self._user_meta = user_meta
            return True # value set
        return False # value not set

    def _validate_and_save_target_format(self, target_format:str|None):
        if not isinstance(target_format, str|None):
            raise ValueError("Target format must be a string value.")
        if target_format is not None:
            ### additional tests needed!!!
            # TODO!
            ### ##########################
            self._target_format = target_format
            return True # value set
        return False # value not set
    
    def _validate_and_save_audio_stream_selection_list(self, audio_stream_selection_list:int|list[int]|None):
        if audio_stream_selection_list is None:
            return False # value not set
        if isinstance(audio_stream_selection_list, int):
            audio_stream_selection_list = [audio_stream_selection_list]
        if not isinstance(audio_stream_selection_list, list):
            raise ValueError("audio_stream_selection_list musst be int or list[int].")
        if not all(isinstance(i, int) for i in audio_stream_selection_list):
            raise ValueError("audio_stream_selection_list: all elements must be integers.")
        if len(audio_stream_selection_list) < 1:
            raise ValueError("At least one audio stream selection needed.")
        # ##################################################
        #
        # TODO for next versions: Accept and process more than one stream
        if len(audio_stream_selection_list) > 1:
            raise ValueError("More than exactly one selected audio stream not yet supported. But, planned for next versions.")
        #
        # ##################################################
        self._base_parameter.selected_audio_stream_list = audio_stream_selection_list
        return True # value set
    
    def _extract_file_data(self):
        """
        Liest vollständige ffprobe-Daten und sortiert sie in die Datenstrukturen ein
        """
        logger.trace(f"Extraction of all meta information by ffprobe requested for file: {self._base_parameter.file.name}. See following steps.")
        try:
            # Vollständiger ffprobe-Befehl für alle Informationen
            cmd = [
                "ffprobe", "-v", "error",
                "-show_format",      # Container-Level Informationen
                "-show_streams",     # Stream-Level Informationen
                "-show_chapters",    # Chapter-Informationen
                "-show_programs",    # Program-Informationen (falls vorhanden)
                "-of", "json",       # JSON-Output für einfache Verarbeitung
                str(self._base_parameter.file)
            ]
            logger.trace(f"Call ffprobe for file {self._base_parameter.file.name} with command list: '{cmd}' as subprocess...")
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    check=True,
                    timeout=30
                )
                logger.trace(f"ffprobe subprocess call finalized successfully.")
            except subprocess.CalledProcessError as e:
                logger.error(f"ffprobe subprocess call failed with return code {e.returncode}")
                logger.error(f"stdout: '{e.stdout}', stderr: '{e.stderr}'")
                logger.error(f"File {self._base_parameter.file.name} could not be successfully analyzed before import.")
                raise RuntimeError(f"Failed to analyze media file {self._base_parameter.file.name}") from e
            except subprocess.TimeoutExpired as e:
                logger.error(f"ffprobe timed out after 30 seconds for file {self._base_parameter.file.name}")
                raise RuntimeError(f"Media file analysis timed out: {self._base_parameter.file.name}") from e
            
            # JSON-Daten parsen
            logger.trace(f"Parse for data in ffprobe response...")
            ffprobe_data = json.loads(result.stdout)
            logger.trace(f"Parsing done.")
            
            # Datenstrukturen füllen
            logger.trace("Call extractors to read information from ffprobe results in detail.")
            self._extract_container_info(ffprobe_data, self._base_parameter.file.name)
            self._extract_stream_info(ffprobe_data, self._base_parameter.file.name)
            self._extract_general_meta(ffprobe_data, self._base_parameter.file.name)
            
        except subprocess.CalledProcessError as e:
            raise ValueError(f"ffprobe-Analyse fehlgeschlagen: {e}")
        except subprocess.TimeoutExpired:
            raise ValueError(f"ffprobe-Timeout bei Datei: {self._base_parameter.file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Ungültige JSON-Ausgabe von ffprobe: {e}")
        except Exception as e:
            raise ValueError(f"Unerwarteter Fehler bei ffprobe-Analyse: {e}")
    
    def _extract_container_info(self, ffprobe_data: dict, file: str):
        """
        Extrahiert Container-Level Informationen aus ffprobe-Daten
        
        Args:
            ffprobe_data: Vollständige ffprobe JSON-Daten
            file: Dateiname für Logging
        """
        logger.trace(f"Extracting container information for file '{file}' requested.")
        format_info = ffprobe_data.get("format", {})
        
        self._container = {
            # Basis Container-Informationen
            "format_name": format_info.get("format_name"),
            "format_long_name": format_info.get("format_long_name"),
            "duration": safe_float_conversion(format_info.get("duration")),
            "size": safe_int_conversion(format_info.get("size")),
            "bit_rate": safe_int_conversion(format_info.get("bit_rate")),
            "probe_score": safe_int_conversion(format_info.get("probe_score")),
            
            # Zusätzliche Container-Eigenschaften
            "start_time": safe_float_conversion(format_info.get("start_time")),
            "nb_streams": safe_int_conversion(format_info.get("nb_streams")),
            "nb_programs": safe_int_conversion(format_info.get("nb_programs"))
        }
        logger.trace(f"Done: Extracting container information for file '{file}'.")
    
    def _extract_stream_info(self, ffprobe_data: dict, file: str):
        """
        Extrahiert Stream-Level Informationen und trennt Audio- von Nicht-Audio-Streams
        
        Args:
            ffprobe_data: Vollständige ffprobe JSON-Daten
            file: Dateiname für Logging
        """
        all_streams = ffprobe_data.get("streams", [])
        
        # Reset der Stream-Listen
        self._audio_streams = []
        self._other_streams = []
        
        for stream_data in all_streams:
            # Trennung nach Audio und Nicht-Audio mit spezifischen Funktionen
            if stream_data.get("codec_type") == "audio":
                stream_info = self._create_audio_stream_info_dict(stream_data, file)
                self._audio_streams.append(stream_info)
            else:
                stream_info = self._create_other_stream_info_dict(stream_data, file)
                self._other_streams.append(stream_info)
                
        # Sortierung nach 'index' parameter:
        self._audio_streams.sort(key=lambda x: x["index"] if x["index"] is not None else 999)
        self._other_streams.sort(key=lambda x: x["index"] if x["index"] is not None else 999)
    
    def _create_audio_stream_info_dict(self, stream_data: dict, file: str) -> dict:
        """
        Erstellt ein standardisiertes Audio-Stream-Info Dictionary
        
        Args:
            stream_data: Rohe Audio-Stream-Daten von ffprobe
            file: Dateiname für Logging
            
        Returns:
            Strukturiertes Audio-Stream-Info Dictionary
        """
        logger.trace(f"Extracting audio stream information of file '{file}'.")
        return {
            # Stream-Identifikation
            "index": safe_int_conversion(stream_data.get("index")),  # Sollte immer Integer sein
            "id": stream_data.get("id"),  # data type can differ from container format to container format
            
            # Codec-Informationen
            "codec_name": stream_data.get("codec_name"),
            "codec_long_name": stream_data.get("codec_long_name"),
            "codec_type": stream_data.get("codec_type"),
            "codec_tag": stream_data.get("codec_tag"),
            "codec_tag_string": stream_data.get("codec_tag_string"),
            
            # Audio-Parameter (technische Eigenschaften)
            "sample_rate": safe_int_conversion(stream_data.get("sample_rate")),
            "sample_fmt": stream_data.get("sample_fmt"),
            "channels": safe_int_conversion(stream_data.get("channels")),
            "channel_layout": stream_data.get("channel_layout"),
            "bits_per_sample": safe_int_conversion(stream_data.get("bits_per_sample")),
            "bit_rate": safe_int_conversion(stream_data.get("bit_rate")),
            
            # Timing-Informationen
            "duration": safe_float_conversion(stream_data.get("duration")),
            "start_time": safe_float_conversion(stream_data.get("start_time")),
            "time_base": stream_data.get("time_base"),
            "start_pts": safe_int_conversion(stream_data.get("start_pts")),
            "duration_ts": safe_int_conversion(stream_data.get("duration_ts")),
            
            # Stream-Eigenschaften
            "disposition": stream_data.get("disposition", {})
        }
    
    def _create_other_stream_info_dict(self, stream_data: dict, file: str) -> dict:
        """
        Erstellt ein standardisiertes Nicht-Audio-Stream-Info Dictionary
        
        Args:
            stream_data: Rohe Nicht-Audio-Stream-Daten von ffprobe
            file: Dateiname für Logging
            
        Returns:
            Strukturiertes Nicht-Audio-Stream-Info Dictionary
        """
        logger.trace(f"Extracting non-audio stream information of file '{file}'.")
        return {
            # Stream-Identifikation
            "index": safe_int_conversion(stream_data.get("index")),  # Sollte immer Integer sein
            "id": stream_data.get("id"),  # data type can differ from container format to container format
            
            # Codec-Informationen
            "codec_name": stream_data.get("codec_name"),
            "codec_long_name": stream_data.get("codec_long_name"),
            "codec_type": stream_data.get("codec_type"),
            "codec_tag": stream_data.get("codec_tag"),
            "codec_tag_string": stream_data.get("codec_tag_string"),
            
            # Timing-Informationen
            "duration": safe_float_conversion(stream_data.get("duration")),
            "start_time": safe_float_conversion(stream_data.get("start_time")),
            "time_base": stream_data.get("time_base"),
            "start_pts": safe_int_conversion(stream_data.get("start_pts")),
            "duration_ts": safe_int_conversion(stream_data.get("duration_ts")),
            
            # Stream-Eigenschaften
            "disposition": stream_data.get("disposition", {})
        }
    
    def _extract_general_meta(self, ffprobe_data: dict, file: str):
        """
        Extrahiert alle nicht-technischen Metadaten durch strukturellen Vergleich
        
        Kopiert die komplette ffprobe-Struktur und entfernt alle bereits
        in _container und _audio_streams/_other_streams verarbeiteten Pfade.
        
        Args:
            ffprobe_data: Vollständige ffprobe JSON-Daten
            file: Dateiname für Logging
        """
        import copy
        logger.trace(f"Extracting all the other information of file '{file}'.")
        
        # Vollständige Kopie der ffprobe-Daten als Ausgangsbasis
        self._general_meta = copy.deepcopy(ffprobe_data)
        
        # Entferne bereits verarbeitete Container-Pfade
        self._remove_processed_container_paths(self._general_meta)
        
        # Entferne bereits verarbeitete Stream-Pfade
        self._remove_processed_stream_paths(self._general_meta)
        
        # Füge Benutzerdefinierte Metadaten hinzu
        self._general_meta["user_meta"] = self._user_meta
    
    def _remove_processed_container_paths(self, meta_data: dict):
        """
        Entfernt bereits in _container verarbeitete Pfade aus meta_data
        
        Args:
            meta_data: Meta-Daten Dictionary (wird modifiziert)
        """
        format_section = meta_data.get("format", {})
        
        # Liste der bereits verarbeiteten Container-Keys
        processed_container_keys = [
            "format_name", "format_long_name", "duration", "size", 
            "bit_rate", "probe_score", "start_time", "nb_streams", "nb_programs"
        ]
        
        # Entferne verarbeitete Keys aus format-Sektion
        for key in processed_container_keys:
            format_section.pop(key, None)
    
    def _remove_processed_stream_paths(self, meta_data: dict):
        """
        Entfernt bereits in _audio_streams/_other_streams verarbeitete Pfade
        
        Args:
            meta_data: Meta-Daten Dictionary (wird modifiziert)
        """
        streams_section = meta_data.get("streams", [])
        
        # Liste der bereits verarbeiteten Audio-Stream-Keys
        processed_audio_keys = [
            "index", "id", "codec_name", "codec_long_name", "codec_type",
            "codec_tag", "codec_tag_string", "sample_rate", "sample_fmt",
            "channels", "channel_layout", "bits_per_sample", "bit_rate",
            "duration", "start_time", "time_base", "start_pts", "duration_ts",
            "disposition"
        ]
        
        # Liste der bereits verarbeiteten Nicht-Audio-Stream-Keys
        processed_other_keys = [
            "index", "id", "codec_name", "codec_long_name", "codec_type",
            "codec_tag", "codec_tag_string", "duration",
            "start_time", "time_base", "start_pts", "duration_ts", "disposition"
        ]
        
        # Bereinige jeden Stream
        for stream in streams_section:
            codec_type = stream.get("codec_type")
            
            if codec_type == "audio":
                # Entferne Audio-spezifische Keys
                for key in processed_audio_keys:
                    stream.pop(key, None)
            else:
                # Entferne Nicht-Audio-spezifische Keys
                for key in processed_other_keys:
                    stream.pop(key, None)
     
    #
    # Ende of initialization part
    #               
    # ################################################
    

    # ################################################
    #
    # Analyzing part
    # --------------
    #
    def _analyse(self, target_format: str|None = None, audio_stream_selection_list: int|list[int]|None = None) -> bool:
        """
        Führt Import-Analyse durch und bestimmt Import-Bereitschaft
        
        Analysiert die relevanten Daten der übergebenen Audio-Streams:
            - entnimmt die relevanten Daten des Audio-Streams mit dem kleinsten Index
              (gewährleistet durch die Sortierung in '_extract_stream_info()')
              - return False, wenn nicht alle Daten verfügbar sind
            - falls mehr als nur ein Index übergeben wurde:
                - vergleiche, ob die anderen Stream identische Parameter haben; Ausnahme: Kanalzahl
                - return False, wenn Vergleich über alle Streams hinweg mindestens einmal nicht erfolgreich
            alles ok: 
                - erstelle self._base_parameter aus den relevanten Daten
                - return True
        
        Args:
            audio_stream_selection_list: Stream-Index(e) für die Analyse (echte ffprobe-Indizes)
            
        Returns:
            True wenn Import möglich, False sonst
        """
        logger.trace(f"Import analysis requested for audio stream indices: {audio_stream_selection_list}")
        self._can_be_imported = True
        
        if target_format is not None:
            self._validate_and_save_target_format(target_format)
        
        if audio_stream_selection_list is not None:
            self._validate_and_save_audio_stream_selection_list(audio_stream_selection_list)
        
        # check if selected audio stream exist in file
        known_indices = [stream["index"] for stream in self._audio_streams if stream["index"] is not None]
        for index in self._base_parameter.selected_audio_stream_list:
            if index not in known_indices:
                self._can_be_imported = False
                logger.error(f"Audio file '{str(self._base_parameter.file.name)}': Missing given Index '{index}' in Audio indices of file. Audio indices of file: {known_indices}")
        
        if not self._can_be_imported:
            logger.trace("Break audio file analysis.")
            return False
        
        # read out parameters of streams and if more than one stream: compare if all
        # selected streams are compatible
        self._base_parameter.total_nb_of_channels = 0
        self._base_parameter.nb_channels_in_stream_list = []
        is_first_stream = True
        for index in self._base_parameter.selected_audio_stream_list:
            stream = next((d for d in self._audio_streams if d["index"] == index), None)
            if stream is None:
                self._can_be_imported = False
                logger.warning(f"Audio file '{str(self._base_parameter.file.name)}': Could not find audio stream with file index {index}")
                logger.trace("Break audio file analysis.")
                break
            
            if is_first_stream:
                # initialize all values
                is_first_stream = False # reset the flag
                
                self._base_parameter.sample_rate = stream.get("sample_rate")
                self._base_parameter.sample_format = stream.get("sample_fmt")
                self._base_parameter.codec_name = stream.get("codec_name")
                self._base_parameter.nb_channels_in_stream_list.append(stream.get("channels"))
                if    self._base_parameter.sample_rate is None \
                   or self._base_parameter.sample_rate < 1:
                    logger.error(f"Audio file '{str(self._base_parameter.file.name)}': Missing required parameter 'sample_rate' in audio stream with index {index}.")
                    self._can_be_imported = False
                    break
                if self._base_parameter.sample_format is None:
                    logger.error(f"Audio file '{str(self._base_parameter.file.name)}': Missing required parameter 'sample_fmt' in audio stream with index {index}.")
                    self._can_be_imported = False
                    break
                if self._base_parameter.codec_name is None:
                    logger.error(f"Audio file '{str(self._base_parameter.file.name)}': Missing required parameter 'codec_name' in audio stream with index {index}.")
                    self._can_be_imported = False
                    break
                if    self._base_parameter.nb_channels_in_stream_list[-1] is None \
                   or self._base_parameter.nb_channels_in_stream_list[-1] < 1:
                    logger.error(f"Audio file '{str(self._base_parameter.file.name)}': Missing required parameter 'channels' in audio stream with index {index}.")
                    self._can_be_imported = False
                    break
                self._base_parameter.total_nb_of_channels += self._base_parameter.nb_channels_in_stream_list[-1]

                # since there is no standard parameter for sample count, we have to calculate
                self._base_parameter.nb_samples = self._calculate_total_samples(index)
                if    self._base_parameter.nb_samples is None \
                   or self._base_parameter.nb_samples < 1:
                    self._can_be_imported = False
                    logger.error(f"Audio file '{str(self._base_parameter.file.name)}': Could not calculate total samples for stream {index}.")
                    break
            else:
                # compare with first stream (außer Kanalzahl)
                if self._base_parameter.sample_rate != stream.get("sample_rate"):
                    logger.error(f"Audio file '{str(self._base_parameter.file.name)}': Sample rate of audio stream with index '{index}' differs to the value of stream '{self._base_parameter.audio_stream_list[0]}'.")
                    self._can_be_imported = False
                    break
                if self._base_parameter.sample_format != stream.get("sample_fmt"):
                    logger.error(f"Audio file '{str(self._base_parameter.file.name)}': Sample format of audio stream with index '{index}' differs to the value of stream '{self._base_parameter.audio_stream_list[0]}'.")
                    self._can_be_imported = False
                    break
                if self._base_parameter.codec_name != stream.get("codec_name"):
                    logger.error(f"Audio file '{str(self._base_parameter.file.name)}': Codec name of audio stream with index '{index}' differs to the value of stream '{self._base_parameter.audio_stream_list[0]}'.")
                    self._can_be_imported = False
                    break
                self._base_parameter.nb_channels_in_stream_list.append(stream.get("channels"))
                if    self._base_parameter.nb_channels_in_stream_list[-1] is None \
                   or self._base_parameter.nb_channels_in_stream_list[-1] < 1:
                    logger.error(f"Audio file '{str(self._base_parameter.file.name)}': Missing required parameter 'channels' in audio stream with index {index}.")
                    self._can_be_imported = False
                    break
                self._base_parameter.total_nb_of_channels += self._base_parameter.nb_channels_in_stream_list[-1]

                # since there is no standard parameter for sample count, we have to calculate
                if self._base_parameter.nb_samples != self._calculate_total_samples(index):
                    self._can_be_imported = False
                    logger.error(f"Audio file '{str(self._base_parameter.file.name)}': Could not calculate total samples for stream {index}.")
                    break
        if not self._can_be_imported:
            logger.trace("Break audio file analysis.")
            return False
        
        if not can_ffmpeg_decode_codec(self._base_parameter.codec_name):
            self._can_be_imported = False
            logger.error(f"Audio file '{str(self._base_parameter.file.name)}': Codec '{self._base_parameter.codec_name}' is not decodable by FFMPEG. Check your FFMPEG version and if a newer version of FFMPEG can decode this. Or use an external tool to decode the file into an intermediate codec before import. In such case, use a lossless codec to maintain sound quality.")       

        # Check compatibility to requested target format
        target_format


        logger.trace(   f"Audio file '{str(self._base_parameter.file.name)}' analysis results: "
                        f"file path={str(self._base_parameter.file.parent)}, "
                        f"file size={self._base_parameter.file_size_bytes} bytes, "
                        f"file hash={self._base_parameter.file_sh256}, "
                        f"container format name={self._base_parameter.container_format_name}, "
                        f"selected audio streams for import (indices):{self._base_parameter.selected_audio_stream_list}, "
                        f"number of channels per selected stream={self._base_parameter.nb_channels_in_stream_list}, "
                        f"total number of selected audio channels={self._base_parameter.total_nb_of_channels}, "
                        f"sample rate={self._base_parameter.sample_rate}, "
                        f"codec name={self._base_parameter.codec_name}, "
                        f"samples={self._base_parameter.nb_samples}, ")
                        f"Ready for import={self._can_be_imported}"

        if self._can_be_imported:
            logger.success(f"Audio file '{str(self._base_parameter.file.name)}': ✅ Import analysis successful for {len(audio_stream_selection_list)} audio stream(s) of file '{str(self._base_parameter.file.name)}'. It is very likely that the file can be imported without errors.")
            
        
        return self._can_be_imported
    
    
    def _calculate_total_samples(self, audio_stream_index: int = 0) -> Optional[int]:
        """
        Berechnet die Gesamtzahl der Samples in einem Audio-Stream. Nutzt dabei die best möglichen
        Methoden, um aus den Metadaten die Samplezahl zu berechnen. In absoluten Ausnahmefällen ist
        es theoretisch möglich, dass die Samplezahl doch anders ist.
        
        Args:
            audio_stream_index: Index im audio_streams Array (0 = erster Audio-Stream)
                               NICHT der echte Stream-Index aus der Datei!
            
        Returns:
            Anzahl Samples oder None wenn Berechnung nicht möglich
        """
        if audio_stream_index >= len(self._audio_streams):
            return None
        
        stream = self._audio_streams[audio_stream_index]
        
        # METHODE 1: duration_ts (BESTE - direkter Sample-Count)
        duration_ts = stream.get("duration_ts")
        time_base = stream.get("time_base")
        sample_rate = stream.get("sample_rate")
        
        if duration_ts and time_base and sample_rate:
            try:
                # Parse time_base: "1/48000" → 1/48000
                if '/' in str(time_base):
                    num, den = map(int, str(time_base).split('/'))
                    # Wenn time_base = 1/sample_rate, dann duration_ts = samples
                    if den == sample_rate and num == 1:
                        return duration_ts  # EXAKTE SAMPLE-ANZAHL!
            except (ValueError, ZeroDivisionError):
                pass
        
        # METHODE 2: duration * sample_rate (STANDARD - sehr gut)
        duration = stream.get("duration")
        if duration and sample_rate:
            return int(duration * sample_rate)  # SEHR ZUVERLÄSSIG
        
        # METHODE 3: Fallback für Container-Duration
        container_duration = self._container.get("duration")
        if container_duration and sample_rate:
            return int(container_duration * sample_rate)
        
        return None
    
    def _find_audio_stream_by_file_index(self, file_stream_index: int) -> Optional[int]:
        """
        Findet den Array-Index eines Audio-Streams basierend auf dem echten Stream-Index
        
        Args:
            file_stream_index: Echter Stream-Index aus ffprobe
            
        Returns:
            Array-Index oder None wenn nicht gefunden
        """
        for i, stream in enumerate(self._audio_streams):
            if stream.get("index") == file_stream_index:
                return i
        return None
    
    # End of analyzing part
    #
    # ##############################################################
    
    
    # ##############################################################
    #
    # User API
    # --------
    #
    # Properties (Setter, Getter)
    #
    
    @property
    def base_parameter(self) -> FileBaseParameters:
        """Base parameters for import.
        
        All specific parameters used during import.
        """
        return self._base_parameter
    
    @property
    def container(self) -> dict:
        """Container level information"""
        return self._container.copy()
    
    @property
    def audio_streams(self) -> List[dict]:
        """List of parameters of audio streams."""
        return [stream.copy() for stream in self._audio_streams]
    
    @property
    def other_streams(self) -> List[dict]:
        """List of parameters of non-audio streams."""
        return [stream.copy() for stream in self._other_streams]
    
    @property
    def general_meta(self) -> dict:
        """General meta data (Tags, etc.)"""
        return self._general_meta.copy()
    
    @property
    def user_meta(self) -> dict:
        """User defined meta data."""
        return self._user_meta.copy()
    
    @user_meta.setter
    def user_meta(self, user_meta:dict):
        self._validate_and_save_user_meta(user_meta)
        logger.trace("user_meta data set. Rerun analyzer.")
        self._analyse()
       
    @property
    def target_format(self) -> str:
        return self._target_format
    
    @target_format.setter 
    def target_format(self, target_format:str):
        self._validate_and_save_target_format(target_format)
        logger.trace("target_format set. Rerun analyzer.")
        self._analyse()
    
    @property
    def audio_stream_selection_list(self) -> list[int]:
        """Selected audio streams in order to import"""
        return self._selected_audio_stream_list.copy()
    
    @audio_stream_selection_list.setter
    def audio_stream_selection_list(self, audio_stream_selection_list:list[int]):
        self._validate_and_save_audio_stream_selection_list(audio_stream_selection_list)
        logger.trace("audio_stream_selection_list set. Rerun analyzer.")
        self._analyse()
    
    @property
    def can_be_imported(self) -> bool:
        """Flag if the import is permitted.
        
        Analyzing must have done with positive result.
        """
        return self._can_be_imported.copy() # True, if completed analysis was positive

    @property
    def has_audio(self) -> bool:
        """Prüft ob Audio-Streams vorhanden sind"""
        return len(self._audio_streams) > 0
    
    @property
    def number_of_audio_stream(self) -> int:
        """Gibt die Anzahl der Audio-Streams zurück"""
        return len(self._audio_streams)
    
    #
    # End of Properties
    #
    # User methods
    #
    
    def analyze(self, audio_stream_selection_list: int|list[int]|None = None) -> bool:
        return self._analyse(self, audio_stream_selection_list)
    # 
    # End of user methods
    #
    # End of User-API
    
    

# End of Class FileParameter
#    
# ############################################################
# ############################################################
    
    

logger.debug("Module loaded.")