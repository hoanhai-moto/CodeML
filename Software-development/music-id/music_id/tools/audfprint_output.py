

import glob
import subprocess
import re
import numpy as np
import os, sys
from itertools import groupby
from operator import itemgetter
from datetime import timedelta, datetime
import math
import json
import time


DELTA = 4 # seconds
DELTA_2 = 2 # seconds
FLOAT_REGEX = r"[-+]?\d*\.\d+|\d+"

# source_files = glob.glob('/Volumes/Storage/DeepHub/mp3/yt_sources/KỲ TÀI THÁCH ĐẤU*.mp3')
# result_file = '/Volumes/Storage/DeepHub/audfprint/outputs/result_4.json'


# tmp_file = '/tmp/audfprint_log_%s.txt' % (datetime.now())
# print('Logged at: ', tmp_file)


def get_result(source_files, output_lines, min_matched_sec=0):
    match_infos = __get_matches(output_lines, min_matched_sec)
    result = __reconstructure_result(source_files, match_infos)
    return result


def __get_matches(output_lines, min_matched_sec):
    match_infos = []

    for line in output_lines:
        # print(line)
        if line.startswith('Matched'):
            match_info = []
            matched_sec, _line = line.split(' starting at ')

            matched_sec = float(re.findall(FLOAT_REGEX, matched_sec)[0])
            match_info.append(matched_sec)

            query_start_in, _line = _line.split(' to time ')

            query_start_sec, query_file_name = query_start_in.split(' in ')
            query_start_sec = float(re.findall(FLOAT_REGEX, query_start_sec)[0])
            match_info.append(query_start_sec)
            match_info.append(query_file_name)

            refer_start_in, _line = _line.split(' with ')

            refer_start_sec, refer_file_name = refer_start_in.split(' in ')
            refer_start_sec = float(re.findall(FLOAT_REGEX, refer_start_sec)[0])
            match_info.append(refer_start_sec)
            match_info.append(refer_file_name)

            num_matched_hashes, _line = _line.split(' of ')
            match_info.append(int(num_matched_hashes.strip()))

            num_landmark_hashes, rank = _line.split(' common hashes at rank ')
            match_info.append(int(num_landmark_hashes.strip()))
            match_info.append(int(rank.strip()))

            if match_info[-3]/match_info[-2] > 0.03 and matched_sec > min_matched_sec - 3:
                print(line)
                match_infos.append(match_info)

    # End loop
    return match_infos


def __reconstructure_result(source_files, match_infos):
    s_time = time.time()
    result = []

    for file in source_files:
        title = file.split('/')[-1]
        # if file.split('/')[-1] != 'CHOQUÉ LA CAMIONETA DE DIEGO - LOS RULES - DIEGO CARDENAS - JORGE ANZALDO -.mp3':
        #     continue

        print('---------------------------')
        print(title)
        print('---------------------------')
        match_info_filter = list(filter(lambda v: v[2] == file, match_infos))
        
        match_info_filter.sort(key = itemgetter(4))
        groups = groupby(match_info_filter, itemgetter(4))
        assets = []

        for (key, matches) in groups:
            asset_title = key.split('/')[-1]
            print('\t', asset_title)
            q_ranges, r_ranges = __merge_group(list(matches))
            q_ranges, r_ranges = __sort_time(q_ranges, r_ranges)
            # q_ranges = __convert_to_minute(q_ranges)
            # r_ranges = __convert_to_minute(r_ranges)
            # print('\t q_ranges: ', q_ranges)
            # print('\t r_ranges: ', r_ranges)

            assets.append({
                'title': asset_title,
                'query_segments': list(q_ranges),
                'asset_segments': list(r_ranges)
            })

        r = {
            'clip_title': title,
            'assets': assets,
        }

        result.append(r)

    # exec_time = time.time() - s_time

    # result = {
    #     # 'config': conf,
    #     'count': len(source_files),
    #     'time': exec_time,
    #     'result': result
    # }

    # with open(result_file, 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(result, indent=2, ensure_ascii=False))


    # print('Match audio completed!')

    # print('Removing Log file')
    # subprocess.run(['rm', tmp_file])

    return result


def __isIoU(range_1, range_2):
    if range_1[0] > range_2[1] or range_2[0] > range_1[1]:
        return False
    return True

def __merge_match_valids(matches):
    q_ranges = []
    r_ranges = []

    for match in matches:
        query_time_end = match[1] + match[0]
        refer_time_end = match[3] + match[0]

        if len(q_ranges) <= 0:
            q_ranges.append([match[1], query_time_end])
            r_ranges.append([match[3], refer_time_end])
            continue

        flag = True
        for i, q_range in enumerate(q_ranges):
            r_range = r_ranges[i]

            __isIoU_query = __isIoU([match[1], query_time_end], q_range)
            __isIoU_refer = __isIoU([match[3], refer_time_end], r_range)

            if __isIoU_query:
                dist = abs((q_range[0] - match[1]) - (r_range[0] - match[3]))
                if __isIoU_refer:
                    flag = False
                    if dist < DELTA:
                        q_range[0] = min(q_range[0], match[1])
                        q_range[1] = max(q_range[1], query_time_end)
                        r_range[0] = min(r_range[0], match[3])
                        r_range[1] = max(r_range[1], refer_time_end)
                        continue
                    break
                
        if flag:
            q_ranges.append([match[1], query_time_end])
            r_ranges.append([match[3], refer_time_end])
            
    return q_ranges, r_ranges


def __merge_group(matches):
    match_valids = []
    for match in matches:
        if len(match_valids) <= 0:
            match_valids.append(match)
            continue

        flag = True
        for mv in match_valids:
            if abs(match[1] - mv[1]) <= DELTA_2 and abs(match[3] - mv[3]) > DELTA_2 or \
                (mv[1] + mv[0]) > match[1] > mv[1] and match[3] < mv[3] and abs(match[3] - mv[3]) > DELTA_2:
                flag = False
                break

        if not flag:
            continue

        match_valids.append(match)

    # print(match_valids)

    return __merge_match_valids(match_valids)



def __convert_to_minute(_list):
    return [[str(timedelta(seconds=math.floor(sec_range[0]))), str(timedelta(seconds=math.ceil(sec_range[1])))] for sec_range in _list]


def __sort_time(q_ranges, r_ranges):
    z = list(zip(q_ranges, r_ranges))
    z.sort(key = lambda e: e[0][0])
    q_ranges, r_ranges = zip(*z)
    return q_ranges, r_ranges

def __str_convert(time_ranges):
    return ';'.join(['-'.join(range) for range in time_ranges])
