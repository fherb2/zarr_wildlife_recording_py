# Umbauplan

Nach der Implementierung der flac- und aac-Codec Prozeduren zur Ablage von Audio-Daten in eine Zarr3-Datenbank, soll nun die Ebene zum Nutzer und zur Zarr-Datenbank umgeschrieben werden. Das ist ehemals das, was bei aimport implementiert und von einigen anderen Funktionen/Python-Modulen unterstützt wurden.

BEACHTE 'umbau_status.md' bezüglich aktualisierten und feiner spezifierten Anforderungen.

Für den Umbau vorbereitet:

## audio_coding.py

--> Übertragen nach 'import_utils.py'

* class TargetFormats – Definiert die Möglichkeiten, das Quell-Audio in den Zielcodec umzukodieren. Neben flac und aac wird noch eine Numpy-Variante vorgesehenm, die noch nicht implementiert ist. Für flac und aac gibt es jeweils die Variante, dass die Zielsamplingrate nicht vorgegeben wird, sowie eine Vorauswahl möglicher Samplingraten. Wenn die Samplingrate nicht vorgegeben wird, soll beim Check des Quellfiles mit FileParameter.analyze() eine sinnvolle Samplingrate voreingestellt werden. Das ist dort noch nicht implementiert: TODO
* class TargetSamplingTransforming – Definiert, wie bei unterschiedlichen Quell- und Zielsamplingraten umgegangen werden soll. 'EXACTLY' ist das typische Verfahren, wenn die Samplingfrequenz der Quelle mit dem Ziel-Codec verarbeitbar ist. Andernfalls gibt es die Methode Resampling mit verschiedenen Ziel-Sampling-Frequenzen, bei der das Ursprüngliche Audiosignal interpolativ auf eine andere Samplingrate kodiert wird. Die dritte Variante Reinterpreting ist vor allem für hohe Samplingfequenzen gedacht, wie sie für die Aufnahme von Ultraschallsignalen, z.B. von Fledermäusen, verwendet wird. In diesem Fall werden die Samples zwar mit einer niedrigen Samplingrate kodiert, allerdings so, als wäre das Signal bereits mit dieser Samplingfrequenz aufgenommen wurden. Damit dehnt sich die Aufnahmezeit zwar, das wird aber bei der Rückkodierung zu PCM rückgängig gemacht, indem dem PCM-Stream wieder die ursprüngliche Samplingfequenz mitgegeben wird. Ohne Resampling. Somit ist es möglich, auch Ultraschallsignale verlustbehaftet zu kodieren. Vorausgesetzt, dass die Methoden dabei keine Artefakte erzeugt, die bei der Reinterprtierung in Erscheinung treten.
*   class AudioCompressionBaseType – Dient der grundlegenden Unterscheidung, um es sich um unkomprimierte, verlustbehaftete oder verlustfreie Komprimierung handelt. 
* get_audio_codec_compression_type() ist die Methode, die aus dem eigentlichen codec-Namen eines Audio-Streams den AudioCompressionBaseType bestimmt.

## source_file.py

-> Umbenannt und erweitert zu import_utils.py

* class FileParameter – Mit dieser Klasse wird das Audio-Quellfile untersucht und neben einem vollständigen File-Meta-Daten-Überblick auch die grundlegenden Parameter extrahiert. Diese werden mit den gewünschten Importierungsdaten (Target-Format, usw. ) gegengeprüft bzw. wenn nicht vorgegeben auch die Zielparameter (codec, Sampling rate...) vorgeschlagen. Diese Funktion ist noch nicht ganz vollständig für diese Aufgaben.
Die Verwendung ist so gedacht, dass der Nutzer (oder die Import-Funktion, wenn nicht vorher durch den Nutzer erfolgt und das Ergebnis beim Import nicht mitgeliefert) damit die Verwendung mit dem Zielformat verifiziert. Weiterhin soll die Funktion um eine Methode erweitert werden, bei der der Nutzer die extrahierten Daten ausdrucken kann. 

## aimport.py

* Das ist die bisherige High-Level-Datei, mit der die aac und flac Dateien entwickelt wurden. Diese Beiden bleiben unverändert und geben mit ihrer API das Zugriffsmusster vor. Die aimport.py wird ausgewechselt durch eine vergleichbare API, die aber das Nutzererlebnis mit den implementierten und geplanten Hilfsmechanismen vereinfacht.

### Verbleibende Feature von aimport.py:

- init_original_audio_group()
- check_if_original_audio_group()
- safe_get_sample_format_dtype() ist eine Hilfsfunktion; soll neu implementiert werden: in audio_coding.py; zu überprüfen, ob sie so noch verwendet wird bzw. verwendet werden soll.
- is_audio_in_original_audio_group() – soll aber die Feature, die FileParameter ausliest nutzen.
- _get_source_params() wird wahrscheinlich nicht mehr benötigt
- import_original_audio_file() – Das ist die eigentliche zentrale Funktion zum Import. Bezüglich der Parametrierung, Analyse und der darin noch zu implementierenden Vorschlagparameter für den Import (Format, sampling rate, resampling/reinterpreting)(die der Nutzer überschreiben kann) mit class FileParameter ist sie entsprechend abzuändern.