import youtube_dl
import subprocess
import glob, os, sys, io
import timeit
import shutil
import traceback
import logging
import signal
import datetime
import json
import click
from io import TextIOWrapper, BytesIO


root_path = os.path.dirname(os.path.abspath(__file__)) + "/../../"
sys.path.append(root_path)

from music_id.cli import matcher


yt_out_dir = ''
output_log_dir = ''

old_stdout = sys.stdout
total_annots = []
audf_config = None

start_time = timeit.default_timer()

def config_logging(dir):
    now_str = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")
    filename = os.path.abspath(dir) + f"/multi_matcher_{now_str}.log"
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

    logging.info("Logging at: %s", filename)


@click.command()
@click.option("-ydir", "--youtubedir", default='/tmp/music-id/yt-audios', help='A directory for storing youtube audio source')
@click.option("-ldir", "--logdir", default='/tmp/music-id/logs', help='A directory for storing file logs')
@click.option("-pi", "--playlistindex", default=0, help='Youtube playlist index')
def main(youtubedir, logdir, playlistindex):
    global audf_config, total_annots, yt_out_dir, output_log_dir
    
    if not os.path.exists(youtubedir):
        os.makedirs(youtubedir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    count = playlistindex
    yt_out_dir = youtubedir
    output_log_dir = logdir

    config_logging(output_log_dir)

    while True :    
        ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [
                    {'key': 'FFmpegExtractAudio','preferredcodec': 'mp3',

                    'preferredquality': '192',
                    },
                    {'key': 'FFmpegMetadata'},
                ],
                'playliststart': count,
                'playlistend' : count,
                'outtmpl': yt_out_dir + '/%(title)s-%(id)s.%(ext)s'
            }

        count += 1

        logging.info("Dowloading... (next audio)")

        sys.stdout = TextIOWrapper(BytesIO(), sys.stdout.encoding)
        
        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download(['https://www.youtube.com/watch?v=T7STUgWl3Xs&list=UUbq8aOyj9ZtIqcD-0MwG1fQ'])
                
                sys.stdout.seek(0) 
                stdout = sys.stdout.read()
                sys.stdout.close()
                sys.stdout = old_stdout
                
            prefix = '[ffmpeg] Destination: '
            line = list(filter(lambda x: x.startswith(prefix), stdout.split("\n")))[-1]
            audio_path = line.replace(prefix, "")
            print(stdout)

        except:
            traceback_str = traceback.format_exc().__str__()
            sys.stdout.close()
            sys.stdout = old_stdout
            print(traceback_str)
            continue

        # result = matcher.run(input_file=audio_path, outdir=output_log_dir)
        result = {
            'total_time': 1,
            'config': {},
            'result': {
                'clip_title': "audio_name",
                'assets': []
            }
        }
        # process = subprocess.run(
        #     f"python {root_path + '/music_id/cli/matcher.py'} -f {audio_path} -o {output_log_dir}",
        #     shell=True,
        #     check=True,
        #     stdout=subprocess.PIPE,
        #     universal_newlines=True)

        # output = process.stdout

        if audf_config is None:
            audf_config = result.get('config', None)
        
        annot = result.get('result', None)

        if annot is not None:
            total_annots.append(annot)

        stop_time = timeit.default_timer()

        logging.info(f'{count} - Completed {os.path.basname(audio_path)}')

        if stop_time - start_time %100 ==0 :
            print(f'Program has been run for {stop_time - start_time} second')
        if stop_time - start_time >= 86400:
            break
    
    # save result
    save_result()

## end

def save_result():
    with open(os.path.abspath(output_log_dir) + '/total_result.json', 'w') as f:
        json.dump({
            'total_time': timeit.default_timer() - start_time,
            'config': audf_config,
            'result': total_annots
        }, f, indent=2, ensure_ascii=True)

    logging.info('Total result stored!')


def handle_process_terminate(signalNumber, frame):
    logging.info('The process was terminated by signal number %d', signalNumber)
    save_result()



if __name__ == "__main__":
    signal.signal(signal.SIGHUP, handle_process_terminate)
    signal.signal(signal.SIGINT, handle_process_terminate)
    signal.signal(signal.SIGQUIT, handle_process_terminate)
    signal.signal(signal.SIGILL, handle_process_terminate)
    signal.signal(signal.SIGTRAP, handle_process_terminate)
    signal.signal(signal.SIGABRT, handle_process_terminate)
    signal.signal(signal.SIGBUS, handle_process_terminate)
    signal.signal(signal.SIGFPE, handle_process_terminate)
    #signal.signal(signal.SIGKILL, handle_process_terminate)
    signal.signal(signal.SIGUSR1, handle_process_terminate)
    signal.signal(signal.SIGSEGV, handle_process_terminate)
    signal.signal(signal.SIGUSR2, handle_process_terminate)
    signal.signal(signal.SIGPIPE, handle_process_terminate)
    signal.signal(signal.SIGALRM, handle_process_terminate)
    signal.signal(signal.SIGTERM, handle_process_terminate)
    main()