import shutil
import torchaudio
from pathlib import Path

src_root = Path(r"C:\Users\tom\Documents\GitHub\sound_separation\datasets\dcase2019\test")
dst_root = Path(r"C:\Users\tom\Documents\GitHub\sound_separation\datasets\dcase2019_16kHz\test")
dst_root.mkdir(exist_ok=True, parents=True)

for wav in src_root.rglob("*.wav"):
    x, sr = torchaudio.load(wav)
    if sr != 16000:
        x = torchaudio.functional.resample(x, sr, 16000)
    out_path = dst_root / wav.name
    torchaudio.save(out_path, x, 16000)

for csv_file in src_root.rglob("*.csv"):
    try:
        out_csv = dst_root / csv_file.relative_to(src_root)
        out_csv.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(csv_file, out_csv)  # copy2 keeps timestamps and metadata
    except Exception as e:
        print(f"!!! Could not copy {csv_file}: {e}")