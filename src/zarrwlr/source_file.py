from dataclasses import dataclass
import subprocess
import json
import pathlib
import hashlib
from typing import Dict, List, Any, Optional

# import and initialize logging
from .logsetup import get_module_logger
logger = get_module_logger(__file__)
logger.trace("Module loading...")

@dataclass
class FileBaseParameters:
    """Main parameters for import step."""
    file: pathlib.Path = None
    file_size_bytes: int|None = None
    file_sh256: str|None = None
    container_format_name: str|None = None
    audio_stream_list: list[int]|None = None
    nb_channels_in_stream_list: list[int]|None = None
    sampling_rate: int|None = None
    sample_format: str|None = None
    bit_rate: int|None = None
    codec_name: str|None = None
    nb_samples: int|None = None  # Hinzugefügt für Sample-Anzahl
    
    def __post_init__(self):
        if self.audio_stream_list is None:
            self.audio_stream_list = []
        if self.nb_channels_in_stream_list is None:
            self.nb_channels_in_stream_list = []


class FileParameter:
    """
    Vollständige Parameter-Analyse einer Audio-Datei
    
    Strukturiert alle Informationen hierarchisch:
    - Container-Level: Format, Dauer, etc.
    - Audio-Streams: Pro Audio-Stream (Codec, Sample-Rate, etc.)
    - Other-Streams: Nicht-Audio-Streams (Video, Subtitle, etc.)
    - General-Meta: Alle nicht-technischen Metadaten (Tags, etc.)
    """
    
    def __init__(self, file_path: str | pathlib.Path, user_meta: dict = None):
        """
        Initialisiert FileParameter mit vollständiger ffprobe-Analyse
        
        Args:
            file_path: Pfad zur zu analysierenden Audio-Datei
            user_meta: Zusätzliche benutzerdefinierte Metadaten
        """
        logger.trace(f"Initializing a file parameter instance requested with file_path: {str(file_path)}. See following steps.")
        
        # Validierung der Datei
        logger.trace("Step 1: Validate file...")
        file_path = pathlib.Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio-Datei nicht gefunden: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Pfad ist keine Datei: {file_path}")
        logger.trace("Step 1: Validation done.")
        
        # Initialisierung der Basis-Informationen
        self._base_parameter = FileBaseParameters()
        self._base_parameter.file = pathlib.Path(file_path).resolve()
        self._user_meta: dict = user_meta or {}
        logger.trace("Step 2: User meta data included.")
        
        # Initialisierung der Haupt-Datenstrukturen
        self._general_meta: dict = {}
        self._container: dict = {}
        self._audio_streams: List[dict] = []
        self._other_streams: List[dict] = []
        
        # Initialisierung der allgemeinsten File-bezogenen Parameter
        self._base_parameter.file_size_bytes = self._base_parameter.file.stat().st_size
        logger.trace(f"Step 3: File size calculated ({self._base_parameter.file_size_bytes/1000} kByte).")
        logger.trace("Step 4: Start to calculate the hash of the file data...")
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        self._base_parameter.file_sh256 = hasher.hexdigest()
        logger.trace("Step 4: Hash creation done.")
        
        # Vollständige Daten per ffprobe einlesen
        self._extract_file_data()
        
        # set non-Stream values
        self._base_parameter.container_format_name = self._container["format_name"]
    
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
            "duration": self._safe_float_conversion(format_info.get("duration")),
            "size": self._safe_int_conversion(format_info.get("size")),
            "bit_rate": self._safe_int_conversion(format_info.get("bit_rate")),
            "probe_score": self._safe_int_conversion(format_info.get("probe_score")),
            
            # Zusätzliche Container-Eigenschaften
            "start_time": self._safe_float_conversion(format_info.get("start_time")),
            "nb_streams": self._safe_int_conversion(format_info.get("nb_streams")),
            "nb_programs": self._safe_int_conversion(format_info.get("nb_programs"))
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
            "index": self._safe_int_conversion(stream_data.get("index")),  # Sollte immer Integer sein
            "id": stream_data.get("id"),  # data type can differ from container format to container format
            
            # Codec-Informationen
            "codec_name": stream_data.get("codec_name"),
            "codec_long_name": stream_data.get("codec_long_name"),
            "codec_type": stream_data.get("codec_type"),
            "codec_tag": stream_data.get("codec_tag"),
            "codec_tag_string": stream_data.get("codec_tag_string"),
            
            # Audio-Parameter (technische Eigenschaften)
            "sample_rate": self._safe_int_conversion(stream_data.get("sample_rate")),
            "sample_fmt": stream_data.get("sample_fmt"),
            "channels": self._safe_int_conversion(stream_data.get("channels")),
            "channel_layout": stream_data.get("channel_layout"),
            "bits_per_sample": self._safe_int_conversion(stream_data.get("bits_per_sample")),
            "bit_rate": self._safe_int_conversion(stream_data.get("bit_rate")),
            
            # Timing-Informationen
            "duration": self._safe_float_conversion(stream_data.get("duration")),
            "start_time": self._safe_float_conversion(stream_data.get("start_time")),
            "time_base": stream_data.get("time_base"),
            "start_pts": self._safe_int_conversion(stream_data.get("start_pts")),
            "duration_ts": self._safe_int_conversion(stream_data.get("duration_ts")),
            
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
            "index": self._safe_int_conversion(stream_data.get("index")),  # Sollte immer Integer sein
            "id": stream_data.get("id"),  # data type can differ from container format to container format
            
            # Codec-Informationen
            "codec_name": stream_data.get("codec_name"),
            "codec_long_name": stream_data.get("codec_long_name"),
            "codec_type": stream_data.get("codec_type"),
            "codec_tag": stream_data.get("codec_tag"),
            "codec_tag_string": stream_data.get("codec_tag_string"),
            
            # Timing-Informationen
            "duration": self._safe_float_conversion(stream_data.get("duration")),
            "start_time": self._safe_float_conversion(stream_data.get("start_time")),
            "time_base": stream_data.get("time_base"),
            "start_pts": self._safe_int_conversion(stream_data.get("start_pts")),
            "duration_ts": self._safe_int_conversion(stream_data.get("duration_ts")),
            
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
                        return duration_ts  # ✅ EXAKTE SAMPLE-ANZAHL!
            except (ValueError, ZeroDivisionError):
                pass
        
        # METHODE 2: duration * sample_rate (STANDARD - sehr gut)
        duration = stream.get("duration")
        if duration and sample_rate:
            return int(duration * sample_rate)  # ✅ SEHR ZUVERLÄSSIG
        
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
    
    def import_analysis(self, audio_stream_indices: int|list[int] = 0) -> bool:
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
            audio_stream_indices: Stream-Index(e) für die Analyse (echte ffprobe-Indizes)
            
        Returns:
            True wenn Import möglich, False sonst
        """
        logger.trace(f"Import analysis requested for audio stream indices: {audio_stream_indices}")
        
        # Normalisierung der Eingabe
        if isinstance(audio_stream_indices, int):
            audio_stream_indices = [audio_stream_indices]
        
        assert isinstance(audio_stream_indices, list), "audio_stream_indices muss int oder list[int] sein"
        assert all(isinstance(i, int) for i in audio_stream_indices), "Alle Elemente müssen integers sein"
        
        # Überprüfe ob alle angegebenen Indizes existieren
        known_indices = [stream["index"] for stream in self._audio_streams if stream["index"] is not None]
        for index in audio_stream_indices:
            if index not in known_indices:
                logger.error(f"Missing given Index '{index}' in Audio indices of file. Audio indices of file: {known_indices}")
                return False
        
        # Sortiere die zu analysierenden Streams nach ihrer ursprünglichen Reihenfolge im File
        # (basierend auf der bereits sortierten self._audio_streams Liste)
        sorted_stream_info = []
        for stream in self._audio_streams:
            stream_index = stream.get("index")
            if stream_index in audio_stream_indices:
                array_index = self._find_audio_stream_by_file_index(stream_index)
                sorted_stream_info.append({
                    'file_index': stream_index,
                    'array_index': array_index,
                    'stream': stream
                })
        
        logger.trace(f"Processing streams in file order: {[s['file_index'] for s in sorted_stream_info]}")
        
        # Verarbeite Streams in der korrekten Reihenfolge
        reference_stream = None
        reference_array_index = None
        reference_nb_samples = None
        channels_list = []
        processed_file_indices = []
        
        for stream_info in sorted_stream_info:
            file_index = stream_info['file_index']
            array_index = stream_info['array_index']
            stream = stream_info['stream']
            
            if array_index is None:
                logger.error(f"Could not find audio stream with file index {file_index}")
                return False
            
            # Prüfe ob alle erforderlichen Parameter vorhanden sind
            required_params = ["sample_rate", "sample_fmt", "codec_name", "channels"]
            for param in required_params:
                if stream.get(param) is None:
                    logger.error(f"Missing required parameter '{param}' in audio stream {file_index}")
                    return False
            
            # Berechne Sample-Anzahl für diesen Stream
            current_nb_samples = self._calculate_total_samples(array_index)
            if current_nb_samples is None:
                logger.error(f"Could not calculate total samples for stream {file_index}")
                return False
            
            # Setze Referenz-Stream (erster in der File-Reihenfolge)
            if reference_stream is None:
                reference_stream = stream
                reference_array_index = array_index
                reference_nb_samples = current_nb_samples
                logger.trace(f"Using stream {file_index} as reference stream with {reference_nb_samples} samples")
            else:
                # Vergleiche mit Referenz-Stream (außer Kanalzahl)
                compare_params = ["sample_rate", "sample_fmt", "codec_name", "bit_rate"]
                for param in compare_params:
                    if stream.get(param) != reference_stream.get(param):
                        logger.error(f"Parameter '{param}' differs between streams: "
                                   f"{reference_stream.get(param)} vs {stream.get(param)}")
                        return False
                
                # WICHTIG: Vergleiche Sample-Anzahl
                if current_nb_samples != reference_nb_samples:
                    logger.error(f"Sample count differs between streams: "
                               f"reference stream has {reference_nb_samples} samples, "
                               f"stream {file_index} has {current_nb_samples} samples")
                    return False
                
                logger.trace(f"Stream {file_index} matches reference: {current_nb_samples} samples")
            
            # Sammle Kanalzahlen IN FILE-REIHENFOLGE
            channels = stream.get("channels")
            if channels is not None:
                channels_list.append(channels)
            
            # Sammle verarbeitete Indizes IN FILE-REIHENFOLGE
            processed_file_indices.append(file_index)
        
        # Verwende die bereits berechnete Sample-Anzahl
        nb_samples = reference_nb_samples
        
        # Setze die Base-Parameter IN FILE-REIHENFOLGE
        self._base_parameter.audio_stream_list = processed_file_indices  # Bereits in korrekter Reihenfolge
        self._base_parameter.nb_channels_in_stream_list = channels_list   # Bereits in korrekter Reihenfolge
        self._base_parameter.nb_channels_in_stream_list = channels_list
        self._base_parameter.sampling_rate = reference_stream.get("sample_rate")
        self._base_parameter.sample_format = reference_stream.get("sample_fmt")
        self._base_parameter.bit_rate = reference_stream.get("bit_rate")
        self._base_parameter.codec_name = reference_stream.get("codec_name")
        self._base_parameter.nb_samples = nb_samples
        
        logger.success(f"Import analysis successful for {len(audio_stream_indices)} audio stream(s)")
        logger.trace(f"Parameters: sampling_rate={self._base_parameter.sampling_rate}, "
                    f"codec={self._base_parameter.codec_name}, "
                    f"channels={self._base_parameter.nb_channels_in_stream_list}, "
                    f"samples={self._base_parameter.nb_samples}")
        
        return True
    
    # Properties für Zugriff auf die Daten
    
    @property
    def base_parameter(self) -> FileBaseParameters:
        """Basis-Parameter für Import"""
        return self._base_parameter
    
    @property
    def container(self) -> dict:
        """Container-Level Informationen"""
        return self._container.copy()
    
    @property
    def audio_streams(self) -> List[dict]:
        """Liste aller Audio-Streams"""
        return [stream.copy() for stream in self._audio_streams]
    
    @property
    def other_streams(self) -> List[dict]:
        """Liste aller Nicht-Audio-Streams"""
        return [stream.copy() for stream in self._other_streams]
    
    @property
    def general_meta(self) -> dict:
        """Allgemeine Metadaten (Tags, etc.)"""
        return self._general_meta.copy()
    
    @property
    def user_meta(self) -> dict:
        """Benutzerdefinierte Metadaten"""
        return self._user_meta.copy()
    
    # Hilfsmethoden
    
    def get_audio_stream_mapping(self) -> List[dict]:
        """
        Zeigt die Zuordnung zwischen Array-Index und echtem Stream-Index
        
        Returns:
            Liste mit Mapping-Informationen
        """
        mapping = []
        for i, stream in enumerate(self._audio_streams):
            mapping.append({
                "audio_array_index": i,
                "file_stream_index": stream.get("index"),
                "codec": stream.get("codec_name"),
                "sample_rate": stream.get("sample_rate"),
                "channels": stream.get("channels")
            })
        return mapping
    
    def has_audio(self) -> bool:
        """Prüft ob Audio-Streams vorhanden sind"""
        return len(self._audio_streams) > 0
    
    def get_audio_stream_count(self) -> int:
        """Gibt die Anzahl der Audio-Streams zurück"""
        return len(self._audio_streams)
    
    @staticmethod
    def _safe_int_conversion(value: Any) -> Optional[int]:
        """
        Sichere Konvertierung zu int mit None-Fallback
        
        Args:
            value: Zu konvertierender Wert
            
        Returns:
            int oder None
        """
        if value is None or value == "":
            return None
        
        try:
            if isinstance(value, str):
                # Entferne mögliche Einheiten oder Zusätze
                cleaned = value.split()[0] if ' ' in value else value
                return int(float(cleaned))  # float() für Dezimalzahlen als Strings
            return int(value)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _safe_float_conversion(value: Any) -> Optional[float]:
        """
        Sichere Konvertierung zu float mit None-Fallback
        
        Args:
            value: Zu konvertierender Wert
            
        Returns:
            float oder None
        """
        if value is None or value == "":
            return None
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


logger.debug("Module loaded.")