# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from fractions import Fraction
from typing import Any, List

import av
import numpy as np
import torch
from ego4d.features.config import FeatureExtractConfig, get_transform, Video
from pytorchvideo.data import UniformClipSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.data.utils import thwc_to_cthw
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


def get_frames(container, t1, t2, buffer, max_buffer_size):
    # [t1, t2]
    ret = []

    tb = container.streams.video[0].time_base

    def is_in_range(frame):
        t = frame.pts * tb
        return t >= t1 and t < t2

    def exceeds_range(frame):
        return frame.pts * tb >= t2

    for frame in buffer:
        if is_in_range(frame):
            ret.append(frame)

    prev_pts = None
    tmp_buffer = []
    #for frame in sorted(frames, key=lambda x: x.pts):
    for frame in container.decode(video=0):
        if frame.pts is None:
            raise AssertionError("frame is None")
        if prev_pts is not None and frame.pts < prev_pts:
            #raise AssertionError("failed assumption pts in order: ")
            for i in range(len(buffer)-1, -1, -1):
                if buffer[i].pts > frame.pts:
                    tmp_buffer.insert(0, buffer[i])
                    del buffer[i]
                else:
                    break
            for tmp in tmp_buffer:
                for i, item in enumerate(ret):
                    if tmp.pts == item.pts:
                        del ret[i]
                        break
        if not isinstance(frame, av.VideoFrame):
            raise AssertionError("other packets not supported")

        prev_pts = frame.pts

        buffer.append(frame)
        if len(buffer) > max_buffer_size:
            del buffer[0]

        if is_in_range(frame):
            ret.append(frame)
        elif exceeds_range(frame):
            break

        if len(tmp_buffer) > 0:
            for item in tmp_buffer:
                prev_pts = item.pts
                buffer.append(item)
                if len(buffer) > max_buffer_size:
                    del buffer[0]

                if is_in_range(item):
                    ret.append(item)
                elif exceeds_range(item):
                    break
            tmp_buffer = []

    pts_in_ret = [frame.pts for frame in ret]
    if not (np.diff(pts_in_ret) > 0).all():
        raise AssertionError("not increasing sequence of frames")
    return ret


class EncodedVideoCached:
    def __init__(self, path, frame_buffer_size=16):
        self.path = path
        self.vid = EncodedVideo.from_path(path, decoder="pyav")
        self.vid._container.seek(0)

        self.frame_buffer_size = frame_buffer_size
        self.frame_buffer = []
        self.last_t = None

    def get_clip(self, t1, t2):
        if self.last_t is not None and t1 < self.last_t:
            raise AssertionError("cannot seek backward")

        vstream = self.vid._container.streams.video[0]
        vs = vstream.start_time * vstream.time_base
        frames = get_frames(
            self.vid._container,
            t1 + vs,
            t2 + vs,
            self.frame_buffer,
            self.frame_buffer_size,
        )
        self.last_t = t1
        return {
            "video": thwc_to_cthw(
                torch.stack(
                    [torch.from_numpy(frame.to_rgb().to_ndarray()) for frame in frames]
                )
            ).to(torch.float32),
            "audio": None,
        }

    @property
    def duration(self) -> float:
        vstream = self.vid._container.streams.video[0]
        return vstream.duration * vstream.time_base


class IndexableVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self, config: FeatureExtractConfig, videos: List[Video], sampler, transform
    ):
        assert (
            config.inference_config.include_audio
            ^ config.inference_config.include_video
        ), """
        cannot include audio and video at the same time
        """
        self.config = config
        self.clips = []
        self.sampler = sampler
        self.transform = transform

        if self.config.inference_config.include_video:

            self.encoded_videos = {}
            for v in videos:
                try:
                    self.encoded_videos[v.uid] = EncodedVideoCached(v.path)
                except:
                    print(f"coudln't encode video {v.path}")
        else:
            assert self.config.inference_config.include_audio
            self.encoded_videos = {
                v.uid: EncodedVideo.from_path(
                    v.path,
                    decode_audio=True,
                    decode_video=False,
                    perform_seek=True,
                    decoder="pyav",
                )
                for v in videos
            }

        for v in videos:
            self.clips.extend(
                list(get_all_clips(v, self.encoded_videos[v.uid].duration, sampler))
            )

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        video, clip = self.clips[idx]

        (
            clip_start,
            clip_end,
            clip_index,
            aug_index,
            is_last_clip,
        ) = clip

        encoded_video = self.encoded_videos[video.uid]
        datum = encoded_video.get_clip(clip_start, clip_end)
        v_frames = datum["video"]
        a_frames = datum["audio"]
        sample_dict = {
            "video_name": video.uid, # video.uid if video.unique_identifier == "" else f"{video.uid}_{video.unique_identifier}",
            "video_index": idx,
            "clip_index": clip_index,
            "aug_index": aug_index,
            "is_stereo": video.is_stereo,
            "clip_start_sec": float(clip_start),
            "clip_end_sec": float(clip_end),
        }
        if v_frames is not None:
            sample_dict["video"] = v_frames
        if a_frames is not None:
            sample_dict["audio"] = a_frames
            sample_dict["audio_sample_rate"] = encoded_video._container.streams.audio[
                0
            ].rate

        sample_dict = self.transform(sample_dict)
        return sample_dict


class CropIfStereo:
    def __init__(self):
        pass

    def __call__(self, x):
        if x["is_stereo"]:
            v = x["video"]
            assert len(v.shape) == 4
            x["video"] = v[:, :, :, 0 : v.shape[-1] // 2]

            # edge case where some videos are incorrectly
            # encoded from source and weren't corrected
            if v.shape[-1] < v.shape[-2]:
                x["video"] = torch.nn.functional.interpolate(
                    x["video"],
                    size=(x["video"].shape[-1], x["video"].shape[-2] // 2),
                    mode="bilinear",
                )
            # for debugging
            # torchvision.utils.save_image(x["video"].permute(1, 0, 2, 3)[0] / 255.0, fp="/tmp/test.jpg")  # noqa
        return x


def get_all_clips(video, video_length, sampler):
    last_clip_time = None
    annotation = {}
    n_clips = 0
    while True:
        clip = sampler(last_clip_time, video_length, annotation)
        last_clip_time = clip.clip_end_sec
        n_clips += 1

        yield (video, clip)

        if clip.is_last_clip:
            break


def create_dset(
    videos: List[Video], config: FeatureExtractConfig
) -> IndexableVideoDataset:
    assert isinstance(videos[0], Video)

    clip_sampler = UniformClipSampler(
        clip_duration=Fraction(
            config.inference_config.frame_window, config.inference_config.fps
        )
        if isinstance(config.inference_config.frame_window, int)
        else config.inference_config.frame_window,
        stride=Fraction(config.inference_config.stride, config.inference_config.fps)
        if isinstance(config.inference_config.stride, int)
        else config.inference_config.stride,
        backpad_last=True,
    )

    transforms_to_use = [
        CropIfStereo(),
        get_transform(config),
    ]
    if config.io.debug_mode:
        transforms_to_use = [
            CropIfStereo(),
            ApplyTransformToKey(key="video", transform=ShortSideScale(size=256)),
        ]
    try:
        return IndexableVideoDataset(
            config, videos, clip_sampler, Compose(transforms_to_use)
        )
    except:
        return None


def create_data_loader(dset, config: FeatureExtractConfig) -> DataLoader:
    if config.inference_config.batch_size == 0:
        raise AssertionError("not supported")

    if config.inference_config.num_workers == 0:  # for debugging
        return dset

    return DataLoader(
        dset,
        batch_size=config.inference_config.batch_size,
        num_workers=config.inference_config.num_workers,
        prefetch_factor=config.inference_config.prefetch_factor,
    )

def equalFrameRate(vid: EncodedVideoCached, frameRate: int) -> bool:
    if vid.duration.denominator >= 1000:
        if vid.duration.denominator // 1000 == frameRate:
            return True
        else:
            return False
    else:
        if vid.duration.denominator == frameRate:
            return True
        else:
            return False
def create_data_loader_or_dset(
    videos: List[Video], config: FeatureExtractConfig
) -> Any:
    dset = create_dset(videos, config)
    if dset == None:
        return None
    #for k, v in dset.encoded_videos.items():
    #    if not equalFrameRate(v, config.inference_config.fps):
    #        print(f"wrong frameRate: {v.duration}, config:{config.inference_config.fps}")
    #        return None
    return create_data_loader(dset=dset, config=config)
