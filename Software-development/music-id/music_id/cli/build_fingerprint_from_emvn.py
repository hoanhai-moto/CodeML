import os, sys
import click
import time
import warnings
import logging
import requests
import traceback
import functools
import re
import json
import uuid
import shutil
import multiprocessing
import traceback
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from music_id import MusicID

LIMIT = 5000
SHOW = 100
EXTENSION = "mp3"
EMVN_TOKEN = None
HOST = 'http://brentracks.sourceaudio.com'
SEARCH_URL = HOST + '/api/tracks/search'
SEARCH_BY_ID_URL = HOST + '/api/tracks/getById'
DOWNLOAD_URL = HOST + '/api/tracks/download'

global music_id, source_audio_dir, existed_track_list


def config_logging(dir):
    filename = dir + "/default.log"
    logFormatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s',
                                     datefmt='%m/%d/%Y %I:%M:%S %p')
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(filename)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    
    
def add_token(url):
    global EMVN_TOKEN
    return url + '?token=%s' % EMVN_TOKEN


def retrieve_track_meta_data(track_id):
    global SEARCH_BY_ID_URL
    url = add_token(SEARCH_BY_ID_URL)
    query = '&track_id=' + track_id

    res = requests.get(url + query)
    if not res.ok:
        return None
    return json.loads(res.content)


def retrieve_tracks_meta_data(album_id=None, show=SHOW):
    res = None
    page = 0
    query_param_str = ''

    global SEARCH_URL
    url = add_token(SEARCH_URL)

    if album_id is not None:
        query_param_str += '&album_id=%d' % int(album_id)
    if show is not None:
        query_param_str += '&show=%d' % show

    while res is None or int(res['count']) > 0:
        query = query_param_str + '&pg=%d' % page
        page += 1
        res = requests.get(url + query)
        if not res.ok:
            res = None
            yield None, 0
        else:
            res = json.loads(res.content)
            yield res, page


def download_track(track_id, track_filename = None, album_name = None, log_dir = None, format=EXTENSION):
    try:
        track_id, track_filename, album_name, log_dir = track_id
    except:
        pass

    query_param_str = '&track_id=%d' % int(track_id)
    query_param_str += '&format=%s' % format

    global DOWNLOAD_URL, source_audio_dir
    # d = res.headers['content-disposition']
    # file_name = re.findall("filename=\"(.+)\";", d)[0]

    if source_audio_dir is not None:
        album_name = '' if album_name is None else ('Album - ' + album_name.replace('/', '_'))
        track_path = os.path.abspath(source_audio_dir) + ('/%s/%s' % (album_name, track_filename))
        if os.path.exists(track_path):
            return track_path
        os.makedirs(os.path.dirname(track_path), exist_ok=True)
    else:
        track_path = os.path.abspath(log_dir) + ('/' + track_filename)

    url = add_token(DOWNLOAD_URL)

    logging.info('Download track %s', track_id)
    res = requests.get(url + query_param_str)
    if not res.ok or 'audio' not in res.headers.get('Content-Type'):
        return None
    
    with open(track_path, 'wb') as f:
        f.write(res.content)

    return track_path


def is_existed(track_meta):
    global existed_track_list

    filename = os.path.splitext(track_meta['Filename'])[0]

    if existed_track_list is None:
        lst = music_id.list_fingerprinted_songs()
        existed_track_list = [e.get('song_name', '') for e in lst]

    return True if filename in existed_track_list else False


def fingerprint_track(track_id, track_path):
    global music_id
    try:
        music_id.fingerprint_files_to_db([track_path])
    except:
        logging.info('Failed to fingerprint for track %s - %s: %s', 
            str(track_id), 
            os.path.basename(track_path), 
            traceback.format_exc()
        )


# Download tracks
def handle_one_track(track_id, track_filename = None, album_name = None, tmp_dir = None):
    global source_audio_dir, existed_track_list
    try:
        track_id, track_filename, album_name, tmp_dir = track_id
    except:
        pass

    track_path = ""
    try:
        logging.info('Processing for track %s ...', track_id)
        track_path = download_track(track_id, track_filename, album_name, tmp_dir)
        if track_path is not None:
            fingerprint_track(track_id, track_path)
    except:
        print(traceback.format_exc())

    # Remove tmp file
    if source_audio_dir is None:
        try:
            os.remove(track_path)
        except: 
            print(traceback.format_exc())
            
    existed_track_list.append(os.path.splitext(os.path.basename(track_path))[0])
    
    logging.info('Done for track %s.', track_id)

    return 1


def handle_multi_tracks(album_id, tmp_dir, nprocesses=None):
    if album_id is not None:
        logging.info('Processing for album %s ...', album_id)
    else:
        logging.info('Processing all resource on EMVN ...')

    tmp_dir += '/songs'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    count = 0
    num_track = 0
    for meta, page in retrieve_tracks_meta_data(album_id):
        count += SHOW
        if count > LIMIT:
            break

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
                
        if meta is None:
            break
        track_metas = meta['tracks']
        if len(track_metas) <= 0:
            break

        track_ids = []
        track_filenames = []
        album_names = []
        for track_meta in track_metas:

            track_id = track_meta['SourceAudio ID']
            track_filename = track_meta['Filename']
            album_name = track_meta['Album']

            if track_id != track_meta['Master ID']:
                logging.info('Track %s is not a master track', track_id)
                continue

            if re.search(r'_No[^\.]*\.mp3$', track_filename) is not None:
                logging.info('Track %s is a _NoXXX.mp3 track', track_id)
                continue

            if is_existed(track_meta):
                logging.info('Track %s existed!', track_id)
                download_track(track_id, track_filename, album_name, tmp_dir)
                continue

            track_ids.append(track_id)
            track_filenames.append(track_filename)
            album_names.append(album_name)

        del track_metas[:]

        if nprocesses is None:
            try:
                nprocesses = multiprocessing.cpu_count()
            except NotImplementedError:
                nprocesses = 1

        pool = multiprocessing.Pool(nprocesses)

        worker_input = zip(track_ids, track_filenames, album_names,
                           [tmp_dir] * len(track_ids))

        iterator = pool.imap_unordered(download_track, worker_input)

        track_paths = []
        while True:
            try:
                track_path = iterator.next()
                if track_path is not None:
                    track_paths.append(track_path)
                    num_track += 1
            except multiprocessing.TimeoutError:
                continue
            except StopIteration:
                break

        pool.close()
        pool.join()

        # fingerprint a music folder
        logging.info('The album have %d tracks.', len(track_paths))
        chunk = 50
        for i in range(0, len(track_paths), chunk):
            music_id.fingerprint_files_to_db(track_paths[i:i+chunk])

        # Remove tmp file
        try:
            shutil.rmtree(tmp_dir)
        except: 
            print(traceback.format_exc())
        else:
            logging.info('Done for page %s.', page)

    return num_track

                

@click.command()
@click.argument("emvn_token")
@click.option("-t","--track_id", default=None, help="A track id from EMVN source")
@click.option("-a", "--album_id", default=None, help="An album id from EMVN source")
@click.option("-al", "--album_list", default=None, help="The path to the list of album id from EMVN source. The .npy extension.")
@click.option("-alf", "--album_list_from", default=0, help="The start index of album id in the list")
@click.option("-alt", "--album_list_to", default=-1, help="The end index of album id in the list")
@click.option("-ad", "--audio_dir", default=None, help="A directory for storing audio files")
@click.option("-l", "--log_dir", default='/tmp/log/music-id.emvn/', help="The log dir for the process")
@click.option("-c", "--config_path", default=None, help="The path of config file")
def main(emvn_token, 
        track_id, 
        album_id, 
        album_list, 
        album_list_from,
        album_list_to,
        audio_dir, 
        log_dir, 
        config_path):

    global EMVN_TOKEN, music_id, source_audio_dir, existed_track_list

    music_id = MusicID(config_path=config_path)
    source_audio_dir = audio_dir
    existed_track_list = None

    EMVN_TOKEN = emvn_token
    stime = time.time()
    tmp_dir = '/tmp/music-id-' + uuid.uuid4().__str__()
    log_dir = os.path.abspath(log_dir)
    completed_track_path = log_dir + '/completed_tracks.csv'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    os.makedirs(tmp_dir)

    # Config log
    config_logging(log_dir)

    album_ids = [album_id]
    if album_id is None:
      if album_list is not None:
        album_ids = np.load(album_list)

    try:
        if track_id is not None:
            track_meta = retrieve_track_meta_data(track_id)
            track_id = track_meta['SourceAudio ID']
            track_filename = track_meta['Filename']
            album_name = track_meta['Album']

            if is_existed(track_meta):
                logging.info('Track %s existed!', track_id)
                return
            count = handle_one_track(track_id, track_filename, album_name, tmp_dir)
        else:
            count = 0
            num_tracks = 0
            al_from = album_list_from
            album_ids_loop = album_ids[album_list_from: album_list_to]
            for i, album_id in enumerate(album_ids_loop):
                logging.info('index: %d/%d' % (album_list_from + i, len(album_ids)))
                if music_id.db_drop_ratio > 4.5:
                    # Store tracks to an another DB file
                    # Rename the current DB file
                    ext = '.pklz'
                    al_end = np.where(album_ids == album_id)[0][0]
                    post_str = '_%d_%d%s' % (al_from, al_end, ext)
                    new_path = music_id.db_path.replace(ext, post_str)
                    os.rename(music_id.db_path, new_path)

                    logging.info('DB filled with %d albums (%d tracks) at %s' % (al_end - al_from, num_tracks, new_path))

                    num_tracks = 0
                    al_from = al_end

                num = handle_multi_tracks(album_id, tmp_dir, nprocesses=5)
                num_tracks += num
                count += num
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except:
            pass

    # completed
    logging.info("All processes completed in %d(s) for %d tracks", time.time() - stime, count)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()