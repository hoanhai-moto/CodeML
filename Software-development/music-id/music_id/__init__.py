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
import numpy as np
import re

from typing import Union, List, Tuple
from datetime import timedelta
from enum import Enum

dir_path = os.path.dirname(os.path.abspath(__file__))
audfprint_core_path = dir_path + "/libs/audfprint_core"
sys.path.append(dir_path)
sys.path.append(dir_path + "/preprocess")

from music_id import configs
from music_id.tools import music_cutter
from music_id.tools.util import get_output_subprocess, convert_to_minute
from music_id.tools import audfprint_output 

import concurrent.futures
import multiprocessing


global  audfprint_new_db_cmd, \
        audfprint_add_db_cmd,\
        audfprint_match_db_cmd,\
        audfprint_list_db_cmd,\
        audfprint_db_dir


def load_core_lib(config):
    global  audfprint_new_db_cmd, \
            audfprint_add_db_cmd,\
            audfprint_match_db_cmd,\
            audfprint_list_db_cmd,\
            audfprint_db_dir

    audf_config = config.get('audfprint', {})
    db_dir = audf_config.get('db_dir', './db')

    if not os.path.isabs(db_dir):
        db_dir = dir_path + "/../" + db_dir

    audfprint_db_dir = db_dir
    db_path = audfprint_db_dir + '/%s' % audf_config.get('db_base_name', '')

    print(f"Database space located at: {audfprint_db_dir}")

    def get_cmd(action):
        return [
            'python', 
            audfprint_core_path + '/audfprint.py', 
            action, 
            '--dbase', db_path,
            '--ncores', str(audf_config.get('ncores')),
            '--density', str(audf_config.get('density')),
            '--hashbits', str(audf_config.get('hashbits')),
            '--fanout', str(audf_config.get('fanout')),
            '--bucketsize', str(audf_config.get('bucketsize'))
        ]

    audfprint_new_db_cmd = get_cmd('new')
    audfprint_add_db_cmd = get_cmd('add')
    audfprint_list_db_cmd = get_cmd('list')

    audfprint_match_db_cmd = get_cmd('match')
    audfprint_match_db_cmd.extend([
        "--max-matches", str(audf_config.get('max_matches')),
        "--time-quantile", str(audf_config.get('time_quantile')),
        "--min-count", str(audf_config.get('min_count')),
        "--exact-count" if audf_config.get('exact_count', None) is True else '',
        "--find-time-range" if audf_config.get('find_time_range', None) is True else '',
    ])

    sys.path.append(audfprint_core_path)

    return db_path


class MusicID(object):
    def __init__(self, config_obj=None, config_path=None):
        if (config_obj is None):
            config_obj = configs.init(config_path)
            self.runtime_config = config_obj.get('runtime', None)

        # self.core_library_name = core_library_name
        self.config = config_obj

        db_path = load_core_lib(config_obj)
        self.db_path = db_path
        self.db_drop_ratio = 0


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

        db_path = audfprint_db_dir + '/%s' % self.config.get('audfprint', {}).get('db_base_name')
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

        db_path = audfprint_db_dir + '/%s' % self.config.get('audfprint', {}).get('db_base_name')
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
            

    def get_fingerprint(self, file_path: str):
        '''
        Build fingerprint for an audio file

        Parameters:
            file_path: The path of an audio file needed build fingerprint

        Return
            (song_name, fingerprints, file_hash, duration, hash_time)
        '''
        return None

    def list_fingerprinted_songs(self):
        '''
        List the number of fingerprinted songs stored in database

        Return:
            Number
        '''
        global audfprint_list_db_cmd
        
        lst = []

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

        out_lines = get_output_subprocess(audfprint_list_db_cmd)
        
        for line in out_lines:
            if re.search(r'.+\([0-9]+\shashes\)$', line) is not None and file_name in line:
                total_hashes = re.search(r'\([0-9]+\shashes\)$', line)[0]
                total_hashes = int(total_hashes.strip('( hashes)'))
                return {
                    'song_name': file_name,
                    'total_hashes': total_hashes
                }

        return song



    def detect(self, input_file: str, chunk: int = 0, out_dir: str = None):
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

        tmp_dir = out_dir or "/tmp/music-id-" + uuid.uuid4().__str__()
        tmp_dir = os.path.join(tmp_dir, 'audio_segments')
        audio_name = os.path.splitext(os.path.basename(input_file))[0]
        chunk_paths = split_large_audio(input_file, tmp_dir, audio_name, chunk)
        self.__tmp_dir = tmp_dir

        first_period_time = 0
        part_count = 0
        part_total = len(chunk_paths)
        segments = []

        min_matched_sec = self.config.get('WINDOW_SIZE', 5)

        
        segs_list = detect_music(chunk_paths, self.__tmp_dir, min_matched_sec)

        print(segs_list)
        for segs in segs_list:
            # print(chunk_path)
            # Segment music
            # segs = detect_music(self.__tmp_dir, chunk_path, min_matched_sec)
            # segs = [{
            #     'start_time': 0,
            #     'end_time': 100000,
            #     'segment_path': chunk_path
            # }]

            for seg in segs:
                seg['start_time'] += first_period_time
                seg['end_time'] += first_period_time
            
            first_period_time += chunk

            if len(segs) <= 0:
                continue

            if len(segments) > 0:
                latest_time = segments[-1].get('end_time')
                earliest_time = segs[0].get('start_time')
                if abs(latest_time - earliest_time) < 0.5: #second
                    segments[-1]['end_time'] = segs[0]['end_time']
                    segments.extend(segs[1:])
                    continue

            segments.extend(segs)


        # End loop

        gc.collect()
        

        # Get all database file 

        db_path = audfprint_db_dir + '/%s' % self.config.get('audfprint', {}).get('db_name_shell_pattern')

        db_files = glob.glob(db_path)
        #count number core in cpu 
        cpu_count = multiprocessing.cpu_count()

        number_of_database_files= int(len(db_files) / cpu_count)
        """
        
        If cpu_count = 3 and total of database = 112 then database file will split into 3 part to run 
        Part 1 and 2 has 37
        The last part has 38

        """
        database_files_list = list(divide_chunks(db_files, number_of_database_files)) 
        

        print(database_files_list)
        
        segments_array =[segments for i in range(len(database_files_list))]

        min_matched_sec_array =[min_matched_sec for i in range(len(database_files_list))]

            


        sum_database_files_path = list(zip(segments_array,min_matched_sec_array,database_files_list))
        # print("segment: ",len(sumpath[0][0]))
        # print("database list",len(sumpath[0][2]))

        # print("segment: ",len(sumpath[1][0]))
        # print("database list",len(sumpath[1][2]))

        # annotations = self.__detect_segments(sum_database_files_path[0])
        print(len(sum_database_files_path))
        start = time.time()
        logging.info(f"Start matching in {cpu_count} process(es).")
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
            annotations = [executor.submit(detect_segments , database_path) for database_path in sum_database_files_path]
        # merge chunk results
        annotations = [anno.result() for anno in annotations ]
        total_time = time.time() - start

        logging.info({
            'Total time search database ': "{0:.2f}".format(total_time)
        })
        assets = []

        print(annotations)
        logging.info({
            'Total database ': "{}".format(annotations)
        })
        for annotation_list in annotations:
            for annot in annotation_list:
                try:
                    annot_assets = annot['assets']
                    for asset in annot_assets:
                        asset_filter = list(filter(lambda a: a['title'] == asset['title'], assets))
                        if len(asset_filter) <= 0:
                            assets.extend(annot['assets'])
                            continue
                        asset_filter = asset_filter[0]
                        asset_filter['query_segments'].extend(asset['query_segments'])
                        asset_filter['asset_segments'].extend(asset['asset_segments'])
                except:
                    continue
        
        for asset in assets:
            asset['query_segments'] = convert_to_minute(asset['query_segments'])
            asset['asset_segments'] = convert_to_minute(asset['asset_segments'])

        annotations = {
            'clip_title': audio_name,
            'assets': assets
        }

        # print(annotations)

        # remove temp resources
        # try:
        #     shutil.rmtree(tmp_dir)
        # except OSError as e:
        #     logging.info("Error: %s : %s" % (dir_path, e.strerror))

        return annotations

    



    # def __detect_chunk(self, chunk_path: str, start_offset: float):
    #     logging.info('Reading audio file part...')

    #     decoder_file = decoder.read_audiofile(chunk_path)
    #     duration_f = float(decoder_file['duration'])
    #     duration = int(math.ceil(duration_f))
    #     annotations = []

    #     logging.info('Read audio completed.')
    #     logging.info('Duration: %d (s)', duration)

    #     # in second
    #     batches = detect_music(tmp_dir, chunk_path)
    #     # filter batches greater than or equal to WINDOW_SIZE
    #     batches = list(filter(lambda batch: batch[1] - batch[0] >= WINDOW_SIZE, batches))
        
    #     total = len(batches)
    #     count = 0

    #     for time_batch in batches:
    #         try:
    #             count += 1

    #             logging.info(
    #                 f'Progress: {count}/{total} \
    #                 ({str(timedelta(seconds=time_batch[0]))} - {str(timedelta(seconds=time_batch[1]))})')

    #             matches, total_time = self.__detect_batch(time_batch[0], time_batch[1])
                
    #             matches = result.get(RESULTS)
    #         except:
    #             logging.info(traceback.format_exc())
    #         else:
    #             if matches is None or len(matches) <= 0:
    #                 continue

    #             first_period_time_tmp = first_period_time if chunk > 0 else 0
    #             # duration_last = time_batch[1] - duration_f
    #             # duration_last = 0 if duration_last <= 0 else duration_last

    #             # print(json.dumps(matches, indent=2))
    #             # match = max(matches, key=lambda match: match.get(FINGERPRINTED_CONFIDENCE))
    #             musics = []
    #             for match in matches:
    #                 if match[OFFSET_SECS] < 0 or match[FINGERPRINTED_CONFIDENCE] < THRESHOLD:
    #                     continue

    #                 logging.info(json.dumps(match, indent=2))
                
    #                 music_start = max(0, (float(match[OFFSET_SECS])))
                    
    #                 musics.append({
    #                     "start":  str(timedelta(seconds=music_start)),
    #                     "end":  str(timedelta(seconds=(music_start + (time_batch[1] - time_batch[0])))), # - duration_last,
    #                     "id": str(match[SONG_ID]),
    #                     "name": str(match[SONG_NAME]),
    #                     "confidence": match[FINGERPRINTED_CONFIDENCE]
    #                 })

    #             musics = sorted(musics, key=lambda match: match.get('confidence'), reverse=True)

    #             anno_curr = {
    #                 "source": {
    #                     "start": str(timedelta(seconds=first_period_time_tmp + time_batch[0])),
    #                     "end": str(timedelta(seconds=first_period_time_tmp + min(time_batch[1], duration_f)))
    #                 },
    #                 "musics": musics
    #             }

    #             annotations.append(anno_curr)

    #             # if len(annotations) <= 0:
    #             #     annotations.append(anno_curr)
    #             #     continue

    #             # anno_prev = annotations[-1]
                
    #             # Combine anno_curr and anno_prev if possible
    #             # condition = (anno_prev['music']['id'] == anno_curr['music']['id'] and
    #             #     anno_prev['music']['start'] <= anno_curr['music']['start'] <= anno_prev['music']['end'] and
    #             #     anno_prev['source']['start'] <= anno_curr['source']['start'] <= anno_prev['source']['end'])

    #             # if condition == True:
    #             #     anno_prev['source']['end'] = anno_curr['source']['end']
    #             #     anno_prev['music']['end'] = anno_curr['music']['end']
    #             #     anno_prev['confidence'] += anno_curr['confidence']
    #             #     anno_prev['confidence'] /= 2
    #             # else:
    #             #     annotations.append(anno_curr)

    #     return duration_f, annotations

    # def __audfprint_detect_batchs(self, batch_paths: List[str]):
    #     global audfprint_match_db_cmd

    #     cmd = audfprint_match_db_cmd.copy()
    #     cmd.extend(batch_paths)

    #     out_lines = get_output_subprocess(cmd)

    #     result = audfprint_output.get_result(batch_paths, out_lines)

    #     logging.info({
    #         TOTAL_TIME: "{0:.2f}".format(total_time)
    #     })

    #     return result

    # def __dejavu_detect_batch(self, src_audio, from_sec: float, to_sec: float):
    #     result = self.dejavu.recognize(FileRecognizer,
    #                     audiofile = decoder_file['audiofile'], 
    #                     is_24bit_wav = decoder_file['is_24bit_wav'],
    #                     from_second = time_batch[0], 
    #                     to_second = time_batch[1])
    #     result_info = {
    #         TOTAL_TIME: "{0:.2f}".format(result.get(TOTAL_TIME)),
    #         FINGERPRINT_TIME: "{0:.2f}".format(result.get(FINGERPRINT_TIME)),
    #         QUERY_TIME: "{0:.2f}".format(result.get(QUERY_TIME)),
    #         ALIGN_TIME: "{0:.2f}".format(result.get(ALIGN_TIME)),
    #         # RESULTS: result.get(RESULTS)
    #     }
    #     logging.info(result_info)
    #     return result.get(RESULTS)


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

def divide_chunks(l, n): 
      
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 


def detect_segments(database_path):
    global audfprint_match_db_cmd
    logging.info('Run in 1 core')
    s_time = time.time()
    result = []

    seg_paths = [seg.get('segment_path') for seg in database_path[0]]

    cmd = audfprint_match_db_cmd.copy()
    cmd.extend(seg_paths)

    
    
    db_arg_idx = cmd.index('--dbase') + 1

    for db_file in database_path[2]:
        cmd[db_arg_idx] = db_file

        out_lines = get_output_subprocess(cmd)
        try:
            matches = audfprint_output.get_result(seg_paths, out_lines, database_path[1])
            # print(matches)

            total_time = time.time() - s_time

            logging.info({
                'total_time': "{0:.2f}".format(total_time)
            })

            # standardized matches
            lz = list(zip(matches, database_path[0]))
            for match, seg in lz:
                assets = match['assets']
                if len(assets) <= 0:
                    continue
                for asset in assets:
                    asset['query_segments'] = [list(np.array(s) + seg['start_time']) for s in asset['query_segments']]

            matches, segs = zip(*lz)

            result.extend(matches)

            logging.info('Searched on DB %s', os.path.basename(db_file))
        except:
            pass

    return result
