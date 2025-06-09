from dataclasses import dataclass
import numpy as np

@dataclass
class Note:
    Duration: float | int
    Lyric: list[tuple[list[str], str, list[str]]] | list[str]
    Pitch: int
    Text: str
    Tech: np.ndarray | None

def Note_to_Dict(note: Note):
    return {
        'Duration': note.Duration,
        'Lyric': note.Lyric,
        'Pitch': note.Pitch,
        'Text': note.Text,
        'Tech': note.Tech
        }

def Dict_to_Note(note: dict):
    return Note(
        Duration= note['Duration'],
        Lyric= note['Lyric'],
        Pitch= note['Pitch'],
        Text= note['Text'],
        Tech= note['Tech']
        )

def Convert_Duration_Second_to_Frame(
    music: list[Note],
    sample_rate: int,
    hop_size: int
    ) -> list[Note]:
    previous_used = 0

    frame_based_music = []

    for note in music:
        duration = round(note.Duration * sample_rate) + previous_used
        
        if duration <= 0:
            print(note)
            assert False
        
        previous_used = duration % hop_size
        duration = duration // hop_size
        note.Duration = duration
        frame_based_music.append(note)

    return frame_based_music

def Convert_Lyric_Syllable_to_Sequence(
    music: list[Note]
    ) -> list[Note]:
    for note in music:
        if note.Lyric[0] == '<X>':
            continue
        note.Lyric = note.Lyric[0] + [note.Lyric[1]] + note.Lyric[2]

    return music