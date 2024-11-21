import os
import math
import glob
from pydub import AudioSegment
from pydub.silence import split_on_silence ,detect_nonsilent
from spleeter.separator import Separator
from os.path import splitext, basename, exists, join
import sys
import timeit
import tensorflow as tf
import subprocess
import concurrent.futures
import multiprocessing
import logging


from music_id.tools.music_cutter import cut_mp3, cut_mp3s
import shutil

# from split_vocal import test_separate_to_file

BACKENDS = "tensorflow"
# MODELS = ['spleeter:2stems', 'spleeter:4stems', 'spleeter:5stems']
MODELS = 'spleeter:2stems'

MODEL_TO_INST = {
    'spleeter:2stems': ('vocals', 'accompaniment'),
    'spleeter:4stems': ('vocals', 'drums', 'bass', 'other'),
    'spleeter:5stems': ('vocals', 'drums', 'bass', 'piano', 'other'),
}

global separator


separator = None

def pydub_music_detection(file_path):
    filename_p = os.path.splitext(os.path.basename(file_path[0]))[0]

    accompaniment_path = os.path.join(file_path[1], filename_p, 'accompaniment.wav')
    
    sound_file = AudioSegment.from_wav(accompaniment_path)
    audio_chunks = detect_nonsilent(sound_file, 
        # must be silent for at least half a second
        min_silence_len=500,
        # consider it silent if quieter than -16 dBFS
        silence_thresh=-32
    )
    
    # Convert to second
    audio_chunks = [[c[0]/1000, c[1]/1000] for c in audio_chunks if abs(c[0]/1000 - c[1]/1000) >= file_path[2]]
    detail = []

    # tmp_chunk_paths = cut_mp3s(accompaniment_path, audio_chunks, output_folder, filename_p + '_accompaniment')
    chunk_paths = cut_mp3s(file_path[0], audio_chunks, file_path[1], filename_p)
    # chunk_paths.extend(tmp_chunk_paths)
    
    for i, audio in enumerate(audio_chunks):
      detail.append({
        'start_time' :audio[0] ,
        'end_time'  : audio[1] ,
        'segment_path': chunk_paths[i]
      })

    return detail

def detect_music(file_paths,output_folder, THRESHHOLD):

    terminal_file = " ".join("'{0}'".format(w) for w in file_paths)
    
    logging.info('Start split vocal beat')
    start = timeit.default_timer()

    process = subprocess.run(
      f"spleeter separate -i {terminal_file} -o {output_folder}", 
      shell=True, 
      check=True, 
      stdout=subprocess.PIPE, 
      universal_newlines=True)

    output = process.stdout
    end = timeit.default_timer()
    logging.info(f"Split vocal beat in {end - start} (s)")
    # path_parts = file_path.split("/")
    # filename =  path_parts[-1]

    start = timeit.default_timer()

    output_folder_array =[output_folder for i in range(len(file_paths))]
    min_matched_sec_array =[THRESHHOLD for i in range(len(file_paths))]
    sumpath = list(zip(file_paths,output_folder_array,min_matched_sec_array)) 

    cpu_count = multiprocessing.cpu_count()
    logging.info(f"Start detect music chunks in {cpu_count} process(es).")
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        results = [executor.submit(pydub_music_detection, chunk_path).result() for chunk_path in sumpath]

    end = timeit.default_timer()
    logging.info(f"Detected music in {end - start} (s)")

    return results
