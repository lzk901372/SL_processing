import os
import json
import msgpack
from pathlib import Path


def organize_mp_data(mp_data_path, output_file):
    '''
    Organize the mp_data to fit the pair video data, so that it could work well with the extract_listening_segments_threaded.py
    '''
    with open(mp_data_path, 'rb') as f:
        mp_data = msgpack.unpack(f, raw=False)
    
    organized_mp_data = {}
    for i in range(len(mp_data)):
        file_path = mp_data[i]["file_path"]
        file_name = Path(file_path).stem
        prefix = '_'.join(file_name.split('_')[:-1])
        identity = file_name.split('_')[-1]
        
        if prefix not in organized_mp_data:
            organized_mp_data[prefix] = {
                identity: {
                    "file_path": file_path,
                    "listening_segments": mp_data[i]["listening_segments"]
                }
            }
        else:
            organized_mp_data[prefix][identity] = {
                "file_path": file_path,
                "listening_segments": mp_data[i]["listening_segments"]
            }
    
    with open(output_file, 'wb') as f:
        msgpack.pack(organized_mp_data, f)


def optimize_msgpack(
    mp_file, 
    output_file,
    lazy_load=False,
    remove_duplicates=False,
    remove_empty_listening_segments=False,
):
    with open(mp_file, 'rb') as f:
        mp_data = msgpack.unpack(f, raw=False)
    if lazy_load:
        mp_data = {k: v for i, (k, v) in enumerate(mp_data.items()) if i < 100}
        # mp_data = mp_data[:100]
    if remove_duplicates or remove_empty_listening_segments:
        # for i in range(len(mp_data)):
        #     mp_data[i]["listening_segments"] = list(
        #         list(x) for x in set(tuple(x) for x in mp_data[i]["listening_segments"])
        #     )
        nonduplicated_mp_data = {}
        for prefix in mp_data.keys():
            identities = mp_data[prefix].keys()
            for identity in identities:
                if remove_empty_listening_segments and not mp_data[prefix][identity]["listening_segments"]:
                    continue
                if prefix not in nonduplicated_mp_data:
                    nonduplicated_mp_data[prefix] = {}
                nonduplicated_mp_data[prefix][identity] = {
                    "file_path": mp_data[prefix][identity]["file_path"],
                    "listening_segments": list(
                        list(x) for x in set(tuple(x) for x in mp_data[prefix][identity]["listening_segments"])
                    ) if remove_duplicates else mp_data[prefix][identity]["listening_segments"]
                }
        mp_data = nonduplicated_mp_data
    # if remove_empty_listening_segments:
    #     new_mp_data = []
    #     for i in range(len(mp_data)):
    #         if not mp_data[i]["listening_segments"]:
    #             continue
    #         new_mp_data.append(mp_data[i])
    #     mp_data = new_mp_data
    with open(output_file, 'wb') as f:
        msgpack.pack(mp_data, f)


def convert_mp_to_json(
    mp_file, 
    output_file,
    lazy_load=False,
    # remove_duplicates=False,
):
    with open(mp_file, 'rb') as f:
        mp_data = msgpack.unpack(f, raw=False)
    if lazy_load:
        mp_data = {k: v for i, (k, v) in enumerate(mp_data.items()) if i < 100}
        # mp_data = mp_data[:100]
    # if remove_duplicates:
    #     for i in range(len(mp_data)):
    #         mp_data[i]["listening_segments"] = list(
    #             list(x) for x in set(tuple(x) for x in mp_data[i]["listening_segments"])
    #         )
    with open(output_file, 'w') as f:
        json.dump(mp_data, f, indent=4)


if __name__ == "__main__":
    ori_mp_file = "/home/zliao/seamless_interaction/data/updated_diarization_timestamps.mp"
    organized_mp_file = "/home/zliao/seamless_interaction/data/updated_diarization_timestamps_organized.mp"
    output_msgpack_file = "/home/zliao/seamless_interaction/data/updated_diarization_timestamps_optimized.mp"
    output_json_file = "/home/zliao/seamless_interaction/data/updated_diarization_timestamps_optimized_lazy.json"
    
    # organize_mp_data(ori_mp_file, organized_mp_file)
    # optimize_msgpack(organized_mp_file, output_msgpack_file, lazy_load=False, remove_duplicates=True, remove_empty_listening_segments=True)
    convert_mp_to_json(output_msgpack_file, output_json_file, lazy_load=True)
