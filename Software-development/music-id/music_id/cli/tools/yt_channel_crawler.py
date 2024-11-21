# -*- coding: utf-8 -*-

# Sample Python code for youtube.search.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/guides/code_samples#python

import os, traceback, json

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors

src_path = os.path.dirname(os.path.abspath(__file__)) + "/../../"

scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]

def main():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = src_path + '/configs/client_secret_gg.json"
    yt_crawler_result = src_path + '/cli/yt_crawler_result.json'

    # read result file
    try:
        with open(yt_crawler_result, "r") as f:
            video_metadata = json.load(f)
    except:
        video_metadata = {
            'kind': None,
            'nextPageToken': None,
            'totalResults': None,
            'items': []
        }

    # print(video_metadata)


    # Get credentials and create an API client
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        client_secrets_file, scopes)
    credentials = flow.run_console()
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, credentials=credentials)


    while True:
        try:
            next_page_token = video_metadata.get('nextPageToken', '')
            request = youtube.search().list(
                part="id",
                channelId="UCbq8aOyj9ZtIqcD-0MwG1fQ",
                order="date", 
                maxResults=50,
                pageToken=next_page_token
            )
            response = request.execute()
        except:
            print(traceback.format_exc())
            break
        else:
            if video_metadata.get('kind', None) is None:
                video_metadata['kind'] = response.get('kind', None)
            if video_metadata.get('totalResults', None) is None:
                video_metadata['totalResults'] = response.get('pageInfo', {}).get('totalResults', None)

            nextPageToken = response.get('nextPageToken', None)
            if nextPageToken is None:
                break
            else:
                video_metadata['nextPageToken'] = nextPageToken

            video_metadata.get('items').extend(response.get('items', []))


            print(f"Retrieved {len(video_metadata.get('items'))} video's IDs in total {video_metadata.get('totalResults')} videos. Next page: {next_page_token}")


    print(f"Completed for {len(video_metadata.get('items'))} videos")

    # save to file
    with open(yt_crawler_result, "w") as f:
        json.dump(video_metadata, f, indent=2)

if __name__ == "__main__":
    main()