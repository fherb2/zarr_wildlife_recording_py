# Umbau voon aimport

@Claude: Chat-Sprache ist deutsch. Auch wenn hier größtenteils englisch beschrieben. Coding-Kommentare sind ab jetzt immer in Englisch.
@Claude: Es braucht keine Rückwärts-Kompatibilität, da wir noch keine Datenbanken haben.
@Claude: Ich möchte die High-Level-User-API nicht als eine Klasse definieren, was ja durchaus üblich ist. Ich möchte einige wenige High-Level-Funktionen, die der Nutzer je nach eigenem Tun und Lassen nutzen kann. Sodass auch nicht Python-Afine Nutzer mit Anfänger-Programmierwissen damit umgehen können. Das Niveau der Dokumentation in aimport sollte entsprechend angepasst sein, was programmiertechnische Fachbegriffe anbelangt. 
Claude: Im Gegensatz zu manch anderen Funktionen in unteren Leveln solten die Docstrings gut und umfassend über die jeweilige Funktion informieren. -> Aber sämtliche Basisinformationen, die allgemeingültiger Art sind, wie auch funktionsübergreifende Nutzungsbeispiele schreiben wir schon vorher in den Modul Docstring am Anfang der Datei. Hier können durchaus Inhalte nochmals aufgeführt werden, die den anderen Mid-/Low-Level-Modulen schon stehen, da in aimport quasi die Nutzer-Dokumentation für den Audio-Import enthalten sein soll. Das betrifft insbesondere auch die Ausführungen zur Analyse und zur Nutzer-Information mit der FileParameter Klasse.
Claude: Grundsätzlich bei allen Funktionen, die entweder ein Sigle-Parameter oder eine Liste solcher Parameter übergeben bekommen können, gilt: Die Abarbeitung bei einer Liste als Übergabe erfolgt in gleichartigen Subprozessen (eigene Systemprozesse mit eigenem GIL), die jeweils einen einzelnen Parameter bearbeiten. Dazu wird innerhalb der Funktion eine Worker-Funktion (von mir auch als Klasse, wenn das sinvoll ist) mit dem eigentlichen Arbeitsinhalt erstellt. Bei Single-Parameter-Aufruf wird der Worker dann nicht als Subprozess sondern unmittelbar aufgerufen. Das heißt, in diesem Fall kommuniziert er nicht über Pipes sondern gibt das Ergebnis direkt zurück. Vermutlich ist damit für den Worker eine kleine Klasse mit beiden Kommunikationsvarianten gar nicht verkehrt. Die Entscheidung liegt bei Dir.

# Zarr-bezogene Aufgaben

## new: open_zarr_audio_grp(store_path: str|pathlib.Path, group_path: str|pathlib.Path|None = None, create:bool=True) -> zarr.Group

- Should do what the name says.
- Uses init_original_audio_group() and check_if_original_audio_group()
- if create is True, so the group may created if not exist, otherwise raise an exception
- if group exist but is not recognized as zarr audio group, raise an exception
- by using check_and_update_zarr_audio_group(), an upgrade would run, if group exist but outdated. This can run independing if the create flag is set or no. Upgrade are possible every time we open such an audio group with this function.
- The return value is the zarr.Group object as handle for all other functions. 

@Claude: Important: In all other high-level and middle-level functions or methods, we use exactly this zarr.Group object. To be save to do this, we should create a simple Type Class AGroup (or during use in applications: zarrwlr.AGroup). This class should be usable in isinstance(). So, we use in all other functions not more 'zarr.Group', we use 'AGroup'. And we could implement a type check at the beginning of functions or methods. For example, we could allow an API argument as 'AGroup|List[AGroup]': so we are checkable that we don't get an parameter like 'List[zarr.Group]' with a non checked group if it is valid as defined in 'check_and_update_zarr_audio_group()' and the initializer function.

### init_original_audio_group(store_path: str|pathlib.Path, group_path: str|pathlib.Path|None = None) -> zarr.Group

- rename to init_zarr_audio_grp()

Content description: Make a Zarr group (can be the storage entry point) to a identificable group for original audi data. In case it is already such a group, do nothing. In both cases: give back a zarr.Group handle of this group.

- function as implemented inclusive the initializing sub function

### check_and_update_zarr_audio_group(group:zarr.Group) -> bool

Content description: Check if the magic_id is correct and if it is the same version as the package version (means version from Config; not the project.toml/setup-Version). Calls version update function, if needed.

- rename to check_if_zarr_audio_grp()
- change Interface to 'check_if_zarr_audio_grp(group:zarr.Group) -> bool'
- remove the exception raisings, instead:

    - if is it not a valid zarr audio group, return false
    - if only the version is wrong, call a new function 'upgrade_zarr_audio_grp(group:zarr.Group, version:tuple)->zarr.Group'. But implement only as empty template with a return value of 'True'.


## Audio-/Audio-Group bezogene Funktionen

## safe_get_sample_format_dtype()

- wird offenbar nicht mehr benötigt und kann wohl weg


## is_audio_in_original_audio_group()

- neues Interface und umbenennen:

is_audio_in_zarr_audio_group(zarr_audio_group:AGroup, 'file oder file_parameter oder Listen davon:str|pathlib.Path|FileParameter|list[str]|list[pathlib.Path]|list[FileParameter]) -> tuple[bool, FileParameter]|list[tuple[bool, FileParameter]

Ziel ist, wie auch bei einigen nachfolgenden Funktionen, dass der Nutzer grundsätzlich ein einzelnes File oder mehrere Files gleichzeitig behandeln kann (single value or list[]), jedoch auch entscheiden kann, ob er die Audiofiles direkt übergibt oder bereits mit FileParameter untersuchte Audiofile-Parameter übergibt. Wenn er nur File-/Filepaths übergibt, dann ermittelt die Funktion selbsttätig mit FileParameter die notwendigen Parameter, um zu prüfen, ob das File/die Files schon in der Datenbank enthalten sind. Andernfalls wird das übergebene FileParameter direkt benutzt, ohne eine Analyse nochmal anzustoßen (diese wird immer von FileParameter selbsttätig getan, sobald  ein Parameter über die Schnittstelle geändert wird, der ein anderes Analyseergebnis hervorrufen könnte. Das gilt auch für File-Parameter, die unabhängig vom Analyseergebnis sind: Existiert eine FileParameter Instanz, wurden auch die Parameter des zugehörigen Files bereits ermittelt. Als Ergebnis wird ein boolscher Wert pro File(parameter) zurück gegeben, der True ist, wenn das File ganz offenbar schon mal importiert wurde sowie auch FileParameter selbst. Egal ob die FileParameter bereits übergeben wurden oder erst innerhalb des Aufrufs initialisiert wurde.

Der typische usecase wäre nämlich für aimport am Anfang (bei einem einzigen File):

```
aGroup = open_zarr_audio_grp(zarr_group_path)
was_imported, file_params = is_audio_in_zarr_audio_group(aGroup, mein_file.mp3)
if not was_imported:
   print(file_params)
else:
    print("File was already imported.")
```

An Hand der Ausgabe entscheidet der Nutzer, ob und wie er das File importieren will. Konfiguriert eventuell die Zielforgabe in file_parameter um und am Schluß würde er dann aufrufen:

```
aimport(aGroup, file_parameter)
```

## _get_source_params

Sollte entfallen, da wir eigentlicvh alles über FileParameter abwickeln.


## aimport

- ersetzt dann das bisherige import_original_audio_file()

Claude: Wenn DU bei aimport bist, dann schreibe mir erst mal kurz nur ein Konzept der Änderung. Danach entscheiden wir, ob wir es so machen oder hier oder da noch etwas anpassen.

## weitere Funktionen:

- import_original_audio_file_with_config
- validate_aac_import_parameters
- _get_aac_config_for_import
- _log_import_performance
- test_aac_integration

Müssten auch entfallen können. Oder gibt es noch Inhalt, den wir andernorts umsetzen müssen?

## Extractions

extract_audio_segment() und parallel_extract_audio_segment() sind bereits keine Import-Funktionen mehr und tauchen in diesem File nur im Rahmen der bisherigen Entwicklung auf. Diese Funktionen implementieren wir in einem zukünftigen file aaccess.py neu. Um den bisherigen Inhalt aber als Kontext zu erhelten, verschieben wir die beiden Funktionen nur in dieses File. <-- Das habe ich im Vorfeld schon getan.

## Der Aufruf...
```
# We do this check during import to brake the program if ffmpeg is missing as soon as possible.
check_ffmpeg_tools()
logger.debug("Module loaded.")
```

bleibt am Ende des Files erhalten.

@Claude: At the end:

- check the imports; remove what we don't use
- review your code finally