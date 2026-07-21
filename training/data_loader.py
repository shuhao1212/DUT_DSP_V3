"""
data_loader.py — Google Speech Commands v2 数据加载器

数据集路径: D:/speech_data
12 类: down, go, left, no, off, on, right, stop, up, yes, _silence_, _unknown_
特征: 40 Mel bins × 101 frames, 完全匹配 DSP 端 Project3_ExtractLogMel()
"""

import os
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

# ── 常量 ──────────────────────────────────────────────────
SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000  # 1 秒
WIN_SIZE = 480        # 30ms @ 16kHz (DSP: PROJECT3_WIN_SIZE)
HOP_SIZE = 160        # 10ms @ 16kHz (DSP: PROJECT3_HOP_SIZE)
FFT_LEN = 512         # DSP: PROJECT3_FFT_LEN
FREQ_NUM = 257        # FFT_LEN/2 + 1
MELS_NUM = 40         # DSP: PROJECT3_MELS_NUM
NUM_FRAMES = 101      # DSP: PROJECT3_MODEL_FRAMES

COMMAND_WORDS = ['marvin', 'down', 'go', 'left', 'no', 'off', 'on',
                 'right', 'stop', 'up', 'yes']
LABELS = ['_silence_', '_unknown_'] + COMMAND_WORDS
LABEL_TO_ID = {lbl: i for i, lbl in enumerate(LABELS)}
N_CLASSES = len(LABELS)

DATA_ROOT = Path(r"D:/speech_data")


def _build_mel_filter_bank():
    """构建 Mel 滤波器组 (与 DSP Project3_InitTables 完全一致)."""
    import math
    mel_filter = np.zeros((FREQ_NUM, MELS_NUM), dtype=np.float32)

    m_min = 2595.0 * math.log10(1.0)
    m_max = 2595.0 * math.log10(1.0 + (SAMPLE_RATE / 2.0) / 700.0)

    m_pts = np.zeros(MELS_NUM + 2)
    f_pts = np.zeros(MELS_NUM + 2)
    for i in range(MELS_NUM + 2):
        m_pts[i] = m_min + (m_max - m_min) * i / (MELS_NUM + 1)
        f_pts[i] = 700.0 * (10.0 ** (m_pts[i] / 2595.0) - 1.0)

    f_diff = np.diff(f_pts)

    for i in range(FREQ_NUM):
        all_freq = i * (SAMPLE_RATE / 2.0) / (FREQ_NUM - 1)
        for j in range(MELS_NUM):
            slope_j = f_pts[j] - all_freq
            slope_j2 = f_pts[j + 2] - all_freq
            left = -slope_j / f_diff[j]
            right = slope_j2 / f_diff[j + 1]
            val = min(left, right)
            if val < 0:
                val = 0.0
            mel_filter[i, j] = val

    return mel_filter


# 全局 Mel 滤波器组 (惰性初始化)
_mel_filter = None
_window = None


def _get_mel_filter():
    global _mel_filter
    if _mel_filter is None:
        _mel_filter = _build_mel_filter_bank()
    return _mel_filter


def _get_window():
    global _window
    if _window is None:
        _window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(WIN_SIZE) / (WIN_SIZE - 1)))
        _window = _window.astype(np.float32)
    return _window


def extract_log_mel(waveform: np.ndarray) -> np.ndarray:
    """
    提取 Log-Mel 频谱, 与 DSP 端 Project3_ExtractLogMel() 完全一致.

    Args:
        waveform: float32, shape [16000], 范围 [-1, 1]

    Returns:
        logmel: float32, shape [40, 101]
    """
    mel_filter = _get_mel_filter()
    window = _get_window()

    # 镜像填充 (匹配 DSP PaddedModelSample)
    pad_len = WIN_SIZE // 2
    padded = np.zeros(len(waveform) + WIN_SIZE, dtype=np.float32)
    # 左侧镜像
    for i in range(pad_len):
        src = pad_len - i
        if src >= len(waveform):
            src = len(waveform) - 1
        padded[i] = waveform[src]
    # 中间直通
    padded[pad_len:pad_len + len(waveform)] = waveform
    # 右侧镜像
    for i in range(pad_len + len(waveform), len(padded)):
        tail = i - (pad_len + len(waveform))
        src = len(waveform) - 2 - tail
        if src < 0:
            src = 0
        padded[i] = waveform[src]

    logmel = np.zeros((MELS_NUM, NUM_FRAMES), dtype=np.float32)

    for frame in range(NUM_FRAMES):
        start = frame * HOP_SIZE
        frame_data = padded[start:start + WIN_SIZE] * window

        # FFT
        fft_in = np.zeros(FFT_LEN, dtype=np.float32)
        fft_in[:WIN_SIZE] = frame_data
        spec = np.fft.rfft(fft_in)  # → FFT_LEN/2 + 1 = 257
        power = np.abs(spec) ** 2

        # Mel 滤波
        for m in range(MELS_NUM):
            mel_energy = np.dot(power, mel_filter[:, m]) + 1e-6
            logmel[m, frame] = np.log(mel_energy)

    return logmel


def load_wav(filepath: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """加载 WAV 文件, 重采样到 target_sr, 归一化到 [-1, 1]."""
    data, sr = sf.read(str(filepath), dtype='float32')
    if data.ndim > 1:
        data = data[:, 0]  # 取单声道

    # 简单重采样 (数据集本就是 16kHz, 通常不需要)
    if sr != target_sr:
        import scipy.signal
        num_samples = int(len(data) * target_sr / sr)
        data = scipy.signal.resample(data, num_samples)

    # 标准化长度到 CLIP_SAMPLES
    if len(data) < CLIP_SAMPLES:
        data = np.pad(data, (0, CLIP_SAMPLES - len(data)))
    else:
        data = data[:CLIP_SAMPLES]

    # 归一化
    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak

    return data.astype(np.float32)


class SpeechCommandsDataset(Dataset):
    """Google Speech Commands v2 数据集, 12 类."""

    def __init__(self, split: str = 'training',
                 bg_noise_dir: Path = DATA_ROOT / "_background_noise_",
                 bg_noise_prob: float = 0.8,
                 time_shift: int = 1600,
                 augment: bool = True):
        """
        Args:
            split: 'training', 'validation', 'testing'
            bg_noise_dir: 背景噪声目录
            bg_noise_prob: 混合背景噪声的概率
            time_shift: 时间偏移范围 (samples)
            augment: 是否做数据增强
        """
        self.split = split
        self.bg_noise_dir = bg_noise_dir
        self.bg_noise_prob = bg_noise_prob
        self.time_shift = time_shift
        self.augment = augment and (split == 'training')

        # 加载 split 列表
        self.samples = []  # list of (filepath, label_id)

        # 读取官方 split 文件
        val_set = set()
        test_set = set()
        val_list = DATA_ROOT / "validation_list.txt"
        test_list = DATA_ROOT / "testing_list.txt"

        if val_list.exists():
            with open(val_list) as f:
                for line in f:
                    val_set.add(line.strip())
        if test_list.exists():
            with open(test_list) as f:
                for line in f:
                    test_set.add(line.strip())

        # 收集命令词样本
        for word in COMMAND_WORDS:
            word_dir = DATA_ROOT / word
            if not word_dir.exists():
                continue
            for wav_file in sorted(word_dir.glob("*.wav")):
                rel_path = f"{word}/{wav_file.name}"
                if split == 'validation' and rel_path in val_set:
                    self.samples.append((wav_file, LABEL_TO_ID[word]))
                elif split == 'testing' and rel_path in test_set:
                    self.samples.append((wav_file, LABEL_TO_ID[word]))
                elif split == 'training' and rel_path not in val_set and rel_path not in test_set:
                    self.samples.append((wav_file, LABEL_TO_ID[word]))

        # _unknown_ 类: 从其他词中采样
        other_words = []
        for d in DATA_ROOT.iterdir():
            if d.is_dir() and d.name not in COMMAND_WORDS and not d.name.startswith('_'):
                other_words.append(d.name)

        unknown_samples = []
        for word in other_words:
            word_dir = DATA_ROOT / word
            if not word_dir.exists():
                continue
            for wav_file in word_dir.glob("*.wav"):
                rel_path = f"{word}/{wav_file.name}"
                if split == 'validation' and rel_path in val_set:
                    unknown_samples.append(wav_file)
                elif split == 'testing' and rel_path in test_set:
                    unknown_samples.append(wav_file)
                elif split == 'training' and rel_path not in val_set and rel_path not in test_set:
                    unknown_samples.append(wav_file)

        # _unknown_ 数量 = 命令词平均数量
        cmd_count = len(self.samples)
        unknown_needed = cmd_count // len(COMMAND_WORDS)
        if unknown_samples:
            rng = np.random.RandomState(42)
            chosen = rng.choice(unknown_samples, min(unknown_needed, len(unknown_samples)), replace=False)
            for f in chosen:
                self.samples.append((f, LABEL_TO_ID['_unknown_']))

        # 加载背景噪声
        self.bg_noises = []
        if bg_noise_dir.exists():
            for noise_file in bg_noise_dir.glob("*.wav"):
                data, sr = sf.read(str(noise_file), dtype='float32')
                if data.ndim > 1:
                    data = data[:, 0]
                if sr != SAMPLE_RATE:
                    import scipy.signal
                    num_samples = int(len(data) * SAMPLE_RATE / sr)
                    data = scipy.signal.resample(data, num_samples)
                self.bg_noises.append(data.astype(np.float32))

        print(f"[{split}] {len(self.samples)} samples "
              f"(commands: {cmd_count}, unknown: {len(self.samples) - cmd_count})")

    def __len__(self):
        return len(self.samples)

    def _time_shift(self, waveform: np.ndarray) -> np.ndarray:
        shift = random.randint(-self.time_shift, self.time_shift)
        if shift > 0:
            waveform = np.pad(waveform, (shift, 0))[:CLIP_SAMPLES]
        elif shift < 0:
            waveform = np.pad(waveform, (0, -shift))[-CLIP_SAMPLES:]
        return waveform

    def _add_background_noise(self, waveform: np.ndarray) -> np.ndarray:
        if not self.bg_noises or random.random() > self.bg_noise_prob:
            return waveform

        noise = random.choice(self.bg_noises)
        # 随机裁剪噪声到相同长度
        if len(noise) > CLIP_SAMPLES:
            start = random.randint(0, len(noise) - CLIP_SAMPLES)
            noise = noise[start:start + CLIP_SAMPLES]
        else:
            noise = np.pad(noise, (0, CLIP_SAMPLES - len(noise)))

        # 随机 SNR 5~15 dB
        snr_db = random.uniform(5.0, 15.0)
        signal_power = np.mean(waveform ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power > 0:
            scale = np.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
            noise = noise * scale

        return waveform + noise

    def _volume_perturbation(self, waveform: np.ndarray) -> np.ndarray:
        factor = 10 ** (random.uniform(-3.0, 3.0) / 20.0)  # ±3 dB
        return waveform * factor

    def __getitem__(self, index):
        filepath, label = self.samples[index]

        # _silence_ 类在训练时合成
        if label == LABEL_TO_ID['_silence_']:
            waveform = np.random.randn(CLIP_SAMPLES).astype(np.float32) * 0.001
        else:
            waveform = load_wav(filepath)

            if self.augment:
                waveform = self._time_shift(waveform)
                waveform = self._volume_perturbation(waveform)
                waveform = self._add_background_noise(waveform)

        # 提取 Log-Mel
        logmel = extract_log_mel(waveform)  # [40, 101]

        # 添加 channel 维度
        logmel = logmel[np.newaxis, :, :]  # [1, 40, 101]

        return torch.from_numpy(logmel), torch.tensor(label, dtype=torch.long)


def create_dataloaders(batch_size: int = 64,
                       num_workers: int = 0,
                       augment: bool = True):
    """创建训练/验证/测试 DataLoader."""
    train_dataset = SpeechCommandsDataset(split='training', augment=augment)
    val_dataset = SpeechCommandsDataset(split='validation', augment=False)
    test_dataset = SpeechCommandsDataset(split='testing', augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader


def precompute_all_mel(cache_dir: Path = None):
    """
    预计算所有 WAV 文件的 Mel 频谱并缓存.
    首次运行必需, 后续运行跳过.
    """
    if cache_dir is None:
        cache_dir = DATA_ROOT.parent / "mel_cache"

    # 读取 split 文件
    val_set, test_set = set(), set()
    val_list = DATA_ROOT / "validation_list.txt"
    test_list = DATA_ROOT / "testing_list.txt"
    if val_list.exists():
        with open(val_list) as f:
            for line in f:
                val_set.add(line.strip())
    if test_list.exists():
        with open(test_list) as f:
            for line in f:
                test_set.add(line.strip())

    # 遍历所有 WAV 文件
    all_words = [d.name for d in DATA_ROOT.iterdir()
                 if d.is_dir() and not d.name.startswith('_')]

    total = 0
    for word in all_words:
        word_dir = DATA_ROOT / word
        if not word_dir.exists():
            continue
        for wav_file in sorted(word_dir.glob("*.wav")):
            rel_path = f"{word}/{wav_file.name}"

            # 确定 split
            if rel_path in val_set:
                split = 'validation'
            elif rel_path in test_set:
                split = 'testing'
            else:
                split = 'training'

            cache_path = cache_dir / split / word / (wav_file.stem + '.npy')

            if cache_path.exists():
                continue

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            waveform = load_wav(wav_file)
            mel = extract_log_mel(waveform)
            np.save(str(cache_path), mel)
            total += 1

            if total % 1000 == 0:
                print(f"  Precomputed {total} Mel spectrograms ...")

    print(f"  Precomputation complete: {total} new files cached.")
    return total


class PrecomputedSpeechCommandsDataset(Dataset):
    """
    预计算 Mel 频谱的 Dataset, 大幅加速训练.

    首次运行将所有 WAV 文件转换为 Mel 频谱并缓存为 .npy 文件.
    后续运行直接从缓存加载 (比实时计算快 5-10x).
    """

    def __init__(self, split: str = 'training',
                 cache_dir: Path = None,
                 augment: bool = True,
                 spec_augment: bool = True):
        """
        Args:
            split: 'training', 'validation', 'testing'
            cache_dir: 缓存目录, 默认 DATA_ROOT/../mel_cache/
            augment: 是否做数据增强
            spec_augment: 是否做频谱增强 (SpecAugment)
        """
        self.split = split
        self.augment = augment and (split == 'training')
        self.spec_augment = spec_augment and self.augment

        if cache_dir is None:
            cache_dir = DATA_ROOT.parent / "mel_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 收集样本列表
        self.samples = []  # list of (cache_path, label_id)
        self._build_sample_list()
        print(f"[{split}] {len(self.samples)} samples (precomputed mode)")

    def _build_sample_list(self):
        """从缓存目录构建样本列表（比遍历原始数据目录快得多）."""
        split_cache_dir = self.cache_dir / self.split

        # 先收集命令词样本 (从缓存目录直接扫描 .npy 文件)
        cmd_samples = []
        unknown_candidates = []

        if split_cache_dir.exists():
            for word_dir in sorted(split_cache_dir.iterdir()):
                if not word_dir.is_dir():
                    continue
                word = word_dir.name
                npy_files = sorted(word_dir.glob("*.npy"))
                if word in COMMAND_WORDS:
                    for npy_file in npy_files:
                        cmd_samples.append((npy_file, LABEL_TO_ID[word]))
                elif not word.startswith('_'):
                    for npy_file in npy_files:
                        unknown_candidates.append(npy_file)

        self.samples = cmd_samples

        # _unknown_ 类: 从非命令词中随机选择
        if unknown_candidates and self.split == 'training':
            cmd_count = len(cmd_samples)
            unknown_needed = max(1, cmd_count // len(COMMAND_WORDS))
            rng = np.random.RandomState(42)
            chosen = rng.choice(unknown_candidates,
                               min(unknown_needed, len(unknown_candidates)),
                               replace=False)
            for cp in chosen:
                self.samples.append((cp, LABEL_TO_ID['_unknown_']))
        elif unknown_candidates:
            # validation/testing: 使用所有非命令词作为 unknown
            for cp in unknown_candidates:
                self.samples.append((cp, LABEL_TO_ID['_unknown_']))

    def _in_split(self, rel_path, val_set, test_set):
        if self.split == 'validation':
            return rel_path in val_set
        elif self.split == 'testing':
            return rel_path in test_set
        else:  # training
            return rel_path not in val_set and rel_path not in test_set

    def _get_or_compute_mel(self, wav_file: Path, rel_path: str) -> Path:
        """获取或计算 Mel 频谱缓存文件."""
        # 缓存文件名: mel_cache/split/word/hash.npy
        cache_path = self.cache_dir / self.split / rel_path
        cache_path = cache_path.with_suffix('.npy')

        if cache_path.exists():
            return cache_path

        # 计算并缓存
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        waveform = load_wav(wav_file)
        mel = extract_log_mel(waveform)  # [40, 101]
        np.save(str(cache_path), mel)
        return cache_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cache_path, label = self.samples[index]

        # 加载预计算 Mel
        if label == LABEL_TO_ID['_silence_']:
            # _silence_ 在训练时生成低能量随机频谱
            mel = np.random.randn(MELS_NUM, NUM_FRAMES).astype(np.float32) * 0.01
        else:
            mel = np.load(str(cache_path)).astype(np.float32)  # [40, 101]

        # 频谱级数据增强
        if self.augment:
            mel = self._freq_mask(mel)
            mel = self._time_mask(mel)

        # 添加 channel 维度: [1, 40, 101]
        mel = mel[np.newaxis, :, :]
        return torch.from_numpy(mel), torch.tensor(label, dtype=torch.long)

    def _freq_mask(self, mel: np.ndarray, max_width: int = 10, num_masks: int = 1) -> np.ndarray:
        """频率维 masking (SpecAugment)."""
        if not self.spec_augment:
            return mel
        for _ in range(num_masks):
            if np.random.random() < 0.5:
                f = np.random.randint(0, max_width)
                f0 = np.random.randint(0, MELS_NUM - f)
                mel[f0:f0 + f, :] = mel.mean()
        return mel

    def _time_mask(self, mel: np.ndarray, max_width: int = 20, num_masks: int = 2) -> np.ndarray:
        """时间维 masking (SpecAugment)."""
        if not self.spec_augment:
            return mel
        for _ in range(num_masks):
            if np.random.random() < 0.5:
                t = np.random.randint(0, max_width)
                t0 = np.random.randint(0, NUM_FRAMES - t)
                mel[:, t0:t0 + t] = mel.mean()
        return mel


def create_precomputed_dataloaders(batch_size: int = 64,
                                   num_workers: int = 0,
                                   augment: bool = True,
                                   cache_dir: Path = None):
    """创建预计算数据集 DataLoader (首次运行自动预计算)."""
    if cache_dir is None:
        cache_dir = DATA_ROOT.parent / "mel_cache"

    # 首次运行: 预计算所有 Mel
    print("Checking Mel cache ...")
    precompute_all_mel(cache_dir)

    print("Building datasets from cache ...")
    train_dataset = PrecomputedSpeechCommandsDataset(
        split='training', augment=augment, cache_dir=cache_dir)
    val_dataset = PrecomputedSpeechCommandsDataset(
        split='validation', augment=False, cache_dir=cache_dir)
    test_dataset = PrecomputedSpeechCommandsDataset(
        split='testing', augment=False, cache_dir=cache_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing precomputed data loader ...")
    train_loader, val_loader, test_loader = create_precomputed_dataloaders(
        batch_size=8, augment=True)
    x, y = next(iter(train_loader))
    print(f"Batch shape: {x.shape}, labels: {y}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
