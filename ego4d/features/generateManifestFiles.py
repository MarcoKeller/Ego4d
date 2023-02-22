import pandas as pd
import os
import json
import argparse
import os.path
import ffmpeg

def getVideoMetaInformations(path):    
    vid = ffmpeg.probe(path)
    print(vid['streams'])

def appendManifestFiles(files, path, manifest, ego4d, path_extension):    
    for file in files:
        if ".mp4" not in file or ".avi" not in file:
            continue
        
        file_name = file.split(".")[0]
        metaInfromations = getVideoMetaInformations(os.path.join(path, file))
        
        ego4d["videos"].append({ "video_uid": file_name, "unique_identifier": None, "is_stereo" : False })
            
        display_width = 960
        display_heigh = 640
        num_frames = 60

        if path_extension == "":
            path_extension = None

        manifest = manifest.append({"video_uid":file_name, "unique_identifier": None, "path_extension":path_extension, "canonical_num_frames":num_frames, 
                                    "canonical_display_width":display_width, "canonical_display_height":display_heigh, "canonical_audio_start_sec":None, "canonical_audio_duration_sec":None}, ignore_index=True)
    return [manifest, ego4d]

def appendManifestFilesAMARV(files, manifest, ego4d, path_extension, num_frames, identifier, resolutions=[256], cameras=["Front","Back","Left","Right"], data_types=["RGB","Depth","Normal","InstanceSegmentation","SemanticSegmentation"]):    
    for file in files:
        if ".mp4" not in file or ".avi" not in file:
            continue
        
        file_name = file.split(".")[0]
        
        file_resolution = int(file_name.split("_")[3])
        camera = file_name.split("_")[1]
        data_type = file_name.split("_")[0]
        
        if file_resolution in resolutions and camera in cameras and data_type in data_types:
            ego4d["videos"].append({ "video_uid": file_name, "unique_identifier": identifier, "is_stereo" : False })
            
            display_width = 960
            if file_resolution == 256:
                display_width = 384

            manifest = manifest.append({"video_uid":file_name, "unique_identifier":identifier, "path_extension":path_extension, "canonical_num_frames":num_frames, 
                                        "canonical_display_width":display_width, "canonical_display_height":file_resolution, "canonical_audio_start_sec":None, "canonical_audio_duration_sec":None}, ignore_index=True)
    return [manifest, ego4d]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, dest="input", default=None, help='path to the input data directory')
    parser.add_argument('-o', '--output', type=str, dest="output", default=None, help='path to the output data directory')
    parser.add_argument('-g', '--gpus', type=str, dest="output", default=None, help='path to the output data directory')
    parser.add_argument('-d', '--dataset', type=str, dest="dataset_name", default="AMARV", help='path to the output data directory')

    args = parser.parse_args()
    
    # initialize manifest files
    manifest = pd.DataFrame(columns=["video_uid", "unique_identifier", "path_extension", "canonical_num_frames","canonical_display_width","canonical_display_height",
                            "canonical_audio_start_sec","canonical_audio_duration_sec"])
    ego4d = { "videos" : [] }
    
    output_files = []
    if not args.output == None:
        output_files = os.listdir(args.output)
    # walk over dirs and find Video dirs
    for (root, dirs, files) in os.walk(args.input):
        found = False
        for file in files:
            if ".mp4" in file or ".avi" in file:
                found = True
                break
        
        if found:
            path_extension = root.replace(args.input, "")
            print(path_extension)
            
            if args.dataset_name in ["AMARV"]:
                # extract identifier of the sequence
                num_frames = root.split(os.sep)[-2].split("_")[1]
                identifier = root.split(os.sep)[-2].split("_")[2]
            
                # append videos to the manifest files
                [manifest, ego4d] = appendManifestFilesAMARV(files, manifest, ego4d, path_extension, num_frames, identifier, data_types=["RGB"])
            else:
                [manifest, ego4d] = appendManifestFiles(files, root, manifest, ego4d, path_extension)

    
    # save manifest files to input path
    manifest.to_csv(os.path.join(args.input, "manifest.csv"),index=False)
    with open(os.path.join(args.input, "ego4d.json"), "w") as f:
        json.dump(ego4d , f, indent=2) 