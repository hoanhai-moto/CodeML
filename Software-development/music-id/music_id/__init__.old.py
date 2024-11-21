from __future__ import print_function, absolute_import, division

import json
import glob
import os
import sys
import time
import logging
import traceback
import math
import warnings
import gc
import subprocess
import uuid
import shutil
import re

from typing import Union, List, Tuple
from datetime import timedelta
from enum import Enum

dir_path = os.path.dirname(os.path.abspath(__file__))
dejavu_core_path = dir_path + "/libs/dejavu_core"
audfprint_core_path = dir_path + "/libs/audfprint_core"
sys.path.append(dir_path)
sys.path.append(dir_path + "/preprocess")

from music_id import configs
from music_id.tools import music_cutter
from music_id.tools.util import get_output_subprocess
from music_id.tools import audfprint_output 


global  Dejavu, \
        decoder, \
        FileRecognizer, \
        SONG_ID, \
        SONG_NAME, \
        RESULTS, \
        TOTAL_TIME,\
        FINGERPRINT_TIME, \
        QUERY_TIME, \
        ALIGN_TIME,\
        FINGERPRINTED_CONFIDENCE, \
        FIELD_FILE_SHA1,\
        audfprint_new_db_cmd, \
        audfprint_add_db_cmd,\
        audfprint_match_db_cmd,\
        audfprint_list_db_cmd,\
        audfprint_db_dir


def load_core_lib(lib_name, config):
    if lib_name == CoreLibEnum.DEJAVU:
        global Dejavu, decoder, FileRecognizer, \
                SONG_ID, SONG_NAME, RESULTS, TOTAL_TIME,\
                FINGERPRINT_TIME, QUERY_TIME, ALIGN_TIME,\
                FINGERPRINTED_CONFIDENCE, FIELD_FILE_SHA1

        sys.path.append(dejavu_core_path)

        from dejavu import Dejavu
        from dejavu.logic import decoder
        from dejavu.logic.recognizer.file_recognizer import FileRecognizer
        from dejavu.config.settings import (SONG_ID,
                                            SONG_NAME,
                                            OFFSET_SECS,
                                            RESULTS,
                                            TOTAL_TIME,
                                            FINGERPRINT_TIME,
                                            QUERY_TIME,
                                            ALIGN_TIME,
                                            FINGERPRINTED_CONFIDENCE,
                                            FIELD_FILE_SHA1)
        return None
        
    elif lib_name == CoreLibEnum.AUDIFPRINT:
        global  audfprint_new_db_cmd, \
                audfprint_add_db_cmd,\
                audfprint_match_db_cmd,\
                audfprint_list_db_cmd,\
                audfprint_db_dir

        audf_config = config.get('audfprint', {})
        audfprint_db_dir = dir_path + '/../db/'
        db_path = audfprint_db_dir + '/%s' % audf_config.get('db_name')

        def get_cmd(action):
            return [
                'python', 
                audfprint_core_path + '/audfprint.py', 
                action, 
                '--dbase', db_path,
                '--ncores', str(audf_config.get('ncores')),
                '--density', str(audf_config.get('density')),
                '--hashbits', str(audf_config.get('hashbits'))
            ]

        audfprint_new_db_cmd = get_cmd('new')
        audfprint_add_db_cmd = get_cmd('add')
        audfprint_match_db_cmd = get_cmd('match')
        audfprint_list_db_cmd = get_cmd('list')

        sys.path.append(audfprint_core_path)

        return db_path

    else:
        raise Exception('The core library is not found!')


class CoreLibEnum(Enum):
    DEJAVU = 1
    AUDIFPRINT = 2


class MusicID(object):
    def __init__(self, config_obj=None, config_path=None, core_library_name: CoreLibEnum = CoreLibEnum.DEJAVU):
        if (config_obj is None):
            config_obj = configs.init(config_path)
            self.db_config = config_obj.get('database', None)
            self.runtime_config = config_obj.get('runtime', None)

        self.core_library_name = core_library_name
        self.config = config_obj

        db_path = load_core_lib(core_library_name, config_obj)
        self.db_path = db_path
        self.db_drop_ratio = 0

        if (core_library_name == CoreLibEnum.AUDIFPRINT):
            return

        self.dejavu = Dejavu(config_obj)

    def fingerprint_directory_to_db(self, path: str, extensions: List[str], nprocesses:Union[int, None]=None):
        '''
        Fingerprint all tracks from a directory into database

        Parameters:
            path: The path of directory
            extensions: The list of file extensions by string. Ex: '.mp3', '.wav'
            nprocesses: The number of processes for parallel computting

        Return: None
        '''

        global  audfprint_new_db_cmd, \
                audfprint_add_db_cmd,\
                audfprint_db_dir\

        if (self.core_library_name == CoreLibEnum.AUDIFPRINT):
            db_path = audfprint_db_dir + '/%s' % self.config.get('audfprint', {}).get('db_name')
            if not os.path.exists(db_path):
                cmd = audfprint_new_db_cmd.copy()
            else:
                cmd = audfprint_add_db_cmd.copy()

            filepaths = []
            for ext in extensions:
                filepaths.extend(glob.glob(os.path.join(path, "*" + ext)))

            cmd.extend(filepaths)
            # subprocess.run(cmd)
            out_lines = get_output_subprocess(cmd)
            out_lines.reverse()
            for line in out_lines:
                match = re.search(r'\([0-9\.]*%\sdropped\)$', line)
                if match is not None:
                    logging.info(line)
                    drop_ratio = match[0].strip('(% dropped)')
                    self.db_drop_ratio = float(drop_ratio)
                    break

        elif self.core_library_name == CoreLibEnum.DEJAVU:
            self.dejavu.fingerprint_directory(path, extensions, nprocesses)


    def fingerprint_files_to_db(self, filepaths: List[str], song_name: str = None):
        '''
        Fingerprint an audio into database

        Parameters:
            filepath: The path of file
            song_name: The name of the track

        Return: None
        '''
        global  audfprint_new_db_cmd, \
                audfprint_add_db_cmd,\
                audfprint_db_dir\

        if self.core_library_name == CoreLibEnum.AUDIFPRINT:
            db_path = audfprint_db_dir + '/%s' % self.config.get('audfprint', {}).get('db_name')
            if not os.path.exists(db_path):
                cmd = audfprint_new_db_cmd.copy()
            else:
                cmd = audfprint_add_db_cmd.copy()

            # paths_join = ' '.join(['"%s"' % f for f in filepaths])
            cmd.extend(filepaths)
            # subprocess.run(cmd)
            out_lines = get_output_subprocess(cmd)
            out_lines.reverse()
            for line in out_lines:
                match = re.search(r'\([0-9\.]*%\sdropped\)$', line)
                if match is not None:
                    logging.info(line)
                    drop_ratio = match[0].strip('(% dropped)')
                    self.db_drop_ratio = float(drop_ratio)
                    break
            
        elif self.core_library_name == CoreLibEnum.DEJAVU:
            self.dejavu.fingerprint_file(filepath, song_name)


    def get_fingerprint(self, file_path: str):
        '''
        Build fingerprint for an audio file

        Parameters:
            file_path: The path of an audio file needed build fingerprint

        Return
            (song_name, fingerprints, file_hash, duration, hash_time)
        '''
        if self.core_library_name == CoreLibEnum.AUDIFPRINT:
            return None
        return Dejavu._fingerprint_worker(filepath)

    def list_fingerprinted_songs(self):
        '''
        List the number of fingerprinted songs stored in database

        Return:
            Number
        '''
        global audfprint_list_db_cmd
        
        lst = []

        if self.core_library_name == CoreLibEnum.AUDIFPRINT:
            out_lines = get_output_subprocess(audfprint_list_db_cmd)
            for line in out_lines:
                if re.search(r'.+\([0-9]+\shashes\)$', line) is not None:
                    re_match = re.search(r'\([0-9]+\shashes\)$', line)
                    total_hashes = int(re_match[0].strip('( hashes)'))
                    file_name = line[:re_match.start() - 1]
                    file_name = os.path.basename(file_name)
                    file_name = os.path.splitext(file_name)[0]
                    lst.append({
                        'song_name': file_name,
                        'total_hashes': total_hashes
                    })

        elif self.core_library_name == CoreLibEnum.DEJAVU: 
            lst = self.dejavu.songs

        return lst


    def get_song_by_name(self, file_name: str = None):
        '''
        Get a song from database by name

        Parameters:
            file_name: The name of a song

        Return:
            Information of the song stored in database
        '''
        global audfprint_list_db_cmd

        song = None

        if (self.core_library_name == CoreLibEnum.AUDIFPRINT):
            out_lines = get_output_subprocess(audfprint_list_db_cmd)
            
            for line in out_lines:
                if re.search(r'.+\([0-9]+\shashes\)$', line) is not None and file_name in line:
                    total_hashes = re.search(r'\([0-9]+\shashes\)$', line)[0]
                    total_hashes = int(total_hashes.strip('( hashes)'))
                    return {
                        'song_name': file_name,
                        'total_hashes': total_hashes
                    }

        elif self.core_library_name == CoreLibEnum.DEJAVU: 
            song = self.dejavu.db.get_song_by_name(filename)

        return song


    def detect(self, input_file: str, chunk: int = 0):
        '''
        Detect the audio chunks matched with tracks on database

        Parameters:
            input_file: The path of query audio
            chunk: The second number of duration for small audio splited from query audio. 
                   Value Less than or equal to 0 mean it don't split.
                   Ex: 0
                   Ex: 240

        Return:
            A list of annotations 
        '''

        from preprocess.detect_music import detect_music

        startsec = 0
        annotations = []

        END_SONG = self.runtime_config.get('end_song')
        THRESHOLD = self.runtime_config.get('threshold_confidence')
        WINDOW_SIZE = self.runtime_config.get('window_size')
        SLIDING_STEP = self.runtime_config.get('sliding_step')

        tmp_dir = "/tmp/music-id-" + uuid.uuid4().__str__()
        audio_name = os.path.splitext(os.path.basename(input_file))[0]
        splited_paths = split_large_audio(input_file, tmp_dir, audio_name, chunk)

        first_period_time = 0
        part_count = 0
        part_total = len(splited_paths)

        for splited_path in splited_paths:
            part_count += 1

            logging.info('''
            ----------------------
            [Chunk %d/%d]
            ----------------------''', 
            part_count, part_total)

            duration_f, annots = self.__detect_chunk(splited_path, first_period_time)
            
            gc.collect()
            first_period_time += duration_f

        # remove temp resources
        try:
            shutil.rmtree(tmp_dir)
        except OSError as e:
            logging.info("Error: %s : %s" % (dir_path, e.strerror))

        return annotations


    def __audfprint_detect_chunk(self, chunk_path: str, start_offset: float):
        logging.info('Reading audio file part...')
        annotations = []

        # in second
        batches = detect_music(tmp_dir, chunk_path)
        


    def __detect_chunk(self, chunk_path: str, start_offset: float):
        logging.info('Reading audio file part...')

        decoder_file = decoder.read_audiofile(chunk_path)
        duration_f = float(decoder_file['duration'])
        duration = int(math.ceil(duration_f))
        annotations = []

        logging.info('Read audio completed.')
        logging.info('Duration: %d (s)', duration)

        # in second
        batches = detect_music(tmp_dir, chunk_path)
        # filter batches greater than or equal to WINDOW_SIZE
        batches = list(filter(lambda batch: batch[1] - batch[0] >= WINDOW_SIZE, batches))
        
        total = len(batches)
        count = 0

        for time_batch in batches:
            try:
                count += 1

                logging.info(f'Progress: {count}/{total} 
                            ({str(timedelta(seconds=time_batch[0]))} - {str(timedelta(seconds=time_batch[1]))})')

                matches, total_time = self.__detect_batch(time_batch[0], time_batch[1])
                
                matches = result.get(RESULTS)
            except:
                logging.info(traceback.format_exc())
            else:
                if matches is None or len(matches) <= 0:
                    continue

                first_period_time_tmp = first_period_time if chunk > 0 else 0
                # duration_last = time_batch[1] - duration_f
                # duration_last = 0 if duration_last <= 0 else duration_last

                # print(json.dumps(matches, indent=2))
                # match = max(matches, key=lambda match: match.get(FINGERPRINTED_CONFIDENCE))
                musics = []
                for match in matches:
                    if match[OFFSET_SECS] < 0 or match[FINGERPRINTED_CONFIDENCE] < THRESHOLD:
                        continue

                    logging.info(json.dumps(match, indent=2))
                
                    music_start = max(0, (float(match[OFFSET_SECS])))
                    
                    musics.append({
                        "start":  str(timedelta(seconds=music_start)),
                        "end":  str(timedelta(seconds=(music_start + (time_batch[1] - time_batch[0])))), # - duration_last,
                        "id": str(match[SONG_ID]),
                        "name": str(match[SONG_NAME]),
                        "confidence": match[FINGERPRINTED_CONFIDENCE]
                    })

                musics = sorted(musics, key=lambda match: match.get('confidence'), reverse=True)

                anno_curr = {
                    "source": {
                        "start": str(timedelta(seconds=first_period_time_tmp + time_batch[0])),
                        "end": str(timedelta(seconds=first_period_time_tmp + min(time_batch[1], duration_f)))
                    },
                    "musics": musics
                }

                annotations.append(anno_curr)

                # if len(annotations) <= 0:
                #     annotations.append(anno_curr)
                #     continue

                # anno_prev = annotations[-1]
                
                # Combine anno_curr and anno_prev if possible
                # condition = (anno_prev['music']['id'] == anno_curr['music']['id'] and
                #     anno_prev['music']['start'] <= anno_curr['music']['start'] <= anno_prev['music']['end'] and
                #     anno_prev['source']['start'] <= anno_curr['source']['start'] <= anno_prev['source']['end'])

                # if condition == True:
                #     anno_prev['source']['end'] = anno_curr['source']['end']
                #     anno_prev['music']['end'] = anno_curr['music']['end']
                #     anno_prev['confidence'] += anno_curr['confidence']
                #     anno_prev['confidence'] /= 2
                # else:
                #     annotations.append(anno_curr)

        return duration_f, annotations

    def __audfprint_detect_batchs(self, batch_paths: List[str]):
        global audfprint_match_db_cmd

        cmd = audfprint_match_db_cmd.copy()
        cmd.extend(batch_paths)

        out_lines = get_output_subprocess(cmd)

        result = audfprint_output.get_result(batch_paths, out_lines)

        logging.info({
            TOTAL_TIME: "{0:.2f}".format(total_time)
        })

        return result

    def __dejavu_detect_batch(self, src_audio, from_sec: float, to_sec: float):
        result = self.dejavu.recognize(FileRecognizer,
                        audiofile = decoder_file['audiofile'], 
                        is_24bit_wav = decoder_file['is_24bit_wav'],
                        from_second = time_batch[0], 
                        to_second = time_batch[1])
        result_info = {
            TOTAL_TIME: "{0:.2f}".format(result.get(TOTAL_TIME)),
            FINGERPRINT_TIME: "{0:.2f}".format(result.get(FINGERPRINT_TIME)),
            QUERY_TIME: "{0:.2f}".format(result.get(QUERY_TIME)),
            ALIGN_TIME: "{0:.2f}".format(result.get(ALIGN_TIME)),
            # RESULTS: result.get(RESULTS)
        }
        logging.info(result_info)
        return result.get(RESULTS)


def split_large_audio(audio_file: str, outdir: str, song_name: str, chunk: int):
    '''
    Split an audio to many smaller audio

    Parameters:
        audio_file: The path of an audio
        outdir: The path of output directory
        song_name: The name of song
        chunk: The second number of duration for small audio splited from query audio.

    Return:
        List of small audio
    '''

    if chunk <= 0:
        return [audio_file]

    part_dir = os.path.join(os.path.abspath(outdir), "parts")
    if not os.path.exists(part_dir):
        os.makedirs(part_dir)

    extension = os.path.splitext(os.path.basename(audio_file))[1]
    out_path_format = os.path.join(part_dir, song_name + '.%05d' + extension)
    
    segment_command = [
        "ffmpeg",
        "-i", audio_file,
        "-f", "segment",
        "-segment_time", str(chunk),
        "-c", "copy", out_path_format
    ]
    subprocess.call(segment_command)

    __clear_cache()
    return sorted(glob.glob(part_dir + ("/*" + extension)), key=os.path.getmtime)


def __clear_cache():
    '''
    Clear memory cache
    '''
    try:
        os.system('sh -c "echo 3 > /proc/sys/vm/drop_caches"')
        print('Droped caches!')
    finally:
        pass
