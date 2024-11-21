from pydub import AudioSegment
import os
import shutil

def create_folder(FolderName):
  if not os.path.exists(FolderName):
    os.makedirs(FolderName)
  else :
    shutil.rmtree(FolderName)
    os.makedirs(FolderName)

def cut_mp3(path, startSec, endSec,OutputFolder):
    startMin = 0
    endMin = 0
    startTime = startMin*60*1000+startSec*1000
    endTime = endMin*60*1000+endSec*1000

    # Opening file and extracting segment
    song = AudioSegment.from_file(path)
    extract = song[startTime:endTime]
    
    create_folder(OutputFolder+"/out/")

    OutputFileName =OutputFolder+'/out/Cutted {} From Sec {} - {}.wav'.format(path.replace(".mp3","").replace('wav',""),startSec,endSec)
    # Saving
    extract.export(OutputFileName, format="wav")
    return OutputFileName


def cut_mp3s(path, chunks, output_folder, file_name=None):
    if file_name is None:
        file_name = os.path.splitext(os.path.basename(path))[0]
    song = AudioSegment.from_file(path)
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        extrack = song[chunk[0]*1000:chunk[1]*1000]
        p = os.path.join(output_folder, f'{file_name}_{i}_{chunk[0]}_{chunk[1]}.wav')
        extrack.export(p, format="wav")
        chunk_paths.append(p)
    return chunk_paths


def cut_original_song(path, startSec, endSec, filename):
    startMin = 0
    endMin = 0
    startTime = startMin*60*1000+startSec*1000
    endTime = endMin*60*1000+endSec*1000

    # Opening file and extracting segment
    song = AudioSegment.from_file(path)
    extract = song[startTime:endTime]
    print(filename, startSec, endSec)
    # Saving
    extract.export(filename, format="wav")


