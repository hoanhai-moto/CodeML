from __future__ import unicode_literals
import youtube_dl

ydl_opts = {
          'format': 'bestaudio/best',
          'postprocessors': [
              {'key': 'FFmpegExtractAudio','preferredcodec': 'mp3',
              
              'preferredquality': '192',
              },
              {'key': 'FFmpegMetadata'},
          ],
      }

def dowload_youtube(link):
  with youtube_dl.YoutubeDL(ydl_opts) as ydl:
      # youtube_file = ydl.download(link)
      info = ydl.extract_info(str(link), download=True)
      youtube_file_name = ydl.prepare_filename(info)
      
      youtube_file_name=youtube_file_name.replace(".webm",".mp3")
      print(youtube_file_name)
  return youtube_file_name