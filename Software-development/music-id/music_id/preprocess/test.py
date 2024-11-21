import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from music_id.tools.youtube_download import dowload_youtube
from music_id.preprocess.split_vocal import test_separate_to_file
from music_id.preprocess.detect_music import detect_music
BACKENDS = "tensorflow"
# MODELS = ['spleeter:2stems', 'spleeter:4stems', 'spleeter:5stems']
MODELS = 'spleeter:2stems'
base_dir = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-i", "--input",
   help="Input file mp3")
ap.add_argument("-y", "--youtube_link", help="Input youtube link")
ap.add_argument("-o", "--outputfolder", required=True,default = '/', help="output folder")   
args = vars(ap.parse_args())

if __name__ == '__main__':
  if args['input'] :
    name = args['input'];
  if args['youtube_link'] :
    link = args['youtube_link']
    name = dowload_youtube(link)
  
  print(detect_music(args['outputfolder'],name,5))
  
