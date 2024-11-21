import youtube_dl
import subprocess
import glob ,os
import timeit
import shutil
import re
start = timeit.default_timer()
count = 1

def get_id (file):
    file=file.replace(" - ","")
    if len(file.split('-'))>1:
        file = re.sub(r'^.*?-', '',file)
        file = re.sub(r'^.*?-', '',file)
    return file
             
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
            'outtmpl':'./%(title)s-%(id)s.%(ext)s'
        }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download(['https://www.youtube.com/watch?v=FN7ALfpGxiI&list=OLAK5uy_mEUG6FBe9vZQDqoEcleFrtuH1MkFiK0WQ'])
            
    except :
        count+=1 
        continue 
        print('Error download file')
    list_of_files = glob.glob('./*.mp3') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    
    id_vi = get_id(latest_file)
    
    os.rename(latest_file,id_vi)

    process = subprocess.run(
      f"python music_id/cli/matcher.py -c 240 -f '{id_vi}' -o ./output ",
      shell=True,
      check=True,
      stdout=subprocess.PIPE,
      universal_newlines=True)
    
    output = process.stdout

    stop = timeit.default_timer()

    print(f'Added {count} songs')
    count+=1
    if stop - start %100 ==0 :
        print('Program has been run for {stop-start} second')
    if count==860:
        break
