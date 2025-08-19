# Dataset.py
# 변환 분리 + 동기화(트리플릿 변환)
# Dataset.py  (latent/cond 전용 미니멀 버전)
import re, torch, pathlib
from typing import Optional
from torch.utils.data import Dataset
from lru_cache import LRUCache

class DefectSynthesisDataset(Dataset):
    """
    기대 디렉토리 구조 (원본 .pt는 사용하지 않음):
      root/<class>/
        ├─ OK_lat/*.pt         # list[Tensor] of (4,64,64), fp16 권장
        ├─ Full_NG_lat/*.pt    # list[Tensor] of (4,64,64), fp16 권장  (target)
        └─ NG_cond/*.pt        # list[Tensor] of (3,512,512), [-1,1], fp16 권장
    """

    def __init__(
        self,
        root_dir: str | pathlib.Path,
        *,
        cache_in_ram: bool = False,           # RAM 여유 있을 때만 True
        lru_max_items: Optional[int] = 256,   # LRU 항목 수 제한
        lru_max_bytes: Optional[int] = None,  # LRU 바이트 제한(예: 4*1024**3)
        max_samples: Optional[int] = None,
    ):
        self.root_dir = pathlib.Path(root_dir)
        self.cache_in_ram = cache_in_ram
        self._file_cache = LRUCache(max_items=lru_max_items, max_bytes=lru_max_bytes) if cache_in_ram else None
        self.max_samples = max_samples

        self.class2idx: dict[str,int] = {}
        self.defect2idx: dict[str,int] = {}
        # (ok_lat_path, ng_cond_path, full_lat_path, img_idx, class_id, defect_id)
        self.samples: list[tuple[pathlib.Path, pathlib.Path, pathlib.Path, int, int, int]] = []

        # ── 스캔 ───────────────────────────────────────────
        classes = [p for p in sorted(self.root_dir.iterdir()) if p.is_dir() and p.name != "lost+found"]
        for cls_path in classes:
            ok_lat_dir   = cls_path / "OK_lat"
            full_lat_dir = cls_path / "Full_NG_lat"
            ng_cond_dir  = cls_path / "NG_cond"

            if not (ok_lat_dir.is_dir() and full_lat_dir.is_dir() and ng_cond_dir.is_dir()):
                print(f"[SKIP] {cls_path.name}: OK_lat/Full_NG_lat/NG_cond 중 누락")
                continue

            ok_map   = {p.name: p for p in ok_lat_dir.glob("*.pt")}
            full_map = {p.name: p for p in full_lat_dir.glob("*.pt")}
            ng_map   = {p.name: p for p in ng_cond_dir.glob("*.pt")}

            names = sorted(set(ok_map) & set(full_map) & set(ng_map))
            if not names:
                print(f"[WARN] {cls_path.name}: 파일 교집합 0개")
                continue

            cls_id = self.class2idx.setdefault(cls_path.name, len(self.class2idx))

            for name in names:
                try:
                    n_ok   = self._len_of_pt(ok_map[name])
                    n_full = self._len_of_pt(full_map[name])
                    n_ng   = self._len_of_pt(ng_map[name])
                except Exception as e:
                    print(f"[WARN] {cls_path.name}/{name}: 길이 확인 실패 → 스킵 ({e})")
                    continue

                if not (n_ok == n_full == n_ng):
                    print(f"[WARN] {cls_path.name}/{name}: 길이 불일치 → 스킵 (ok={n_ok}, full={n_full}, ng={n_ng})")
                    continue

                defect = self._extract_defect_type(name)
                dft_id = self.defect2idx.setdefault(defect, len(self.defect2idx))

                for img_idx in range(n_ok):
                    self.samples.append((ok_map[name], ng_map[name], full_map[name], img_idx, cls_id, dft_id))
                    if self.max_samples and len(self.samples) >= self.max_samples:
                        break
                if self.max_samples and len(self.samples) >= self.max_samples:
                    break

        print(f"[INFO] 샘플 {len(self.samples):,}개 | 클래스 {len(self.class2idx)} | 결함 {len(self.defect2idx)}")
        if self.cache_in_ram:
            print(f"[INFO] LRU: max_items={self._file_cache.max_items}  max_bytes={self._file_cache.max_bytes}")

    # ── 유틸 ───────────────────────────────────────────
    def _load_pt(self, path: pathlib.Path):
        def to_cpu(storage, _): return storage.cpu()
        if not self.cache_in_ram:
            return torch.load(path, map_location=to_cpu)

        if path in self._file_cache:
            return self._file_cache.get(path)

        data = torch.load(path, map_location=to_cpu)
        if isinstance(data, list):
            data = [t.cpu() for t in data]
        elif torch.is_tensor(data):
            data = data.cpu()
        self._file_cache.put(path, data)
        return data

    def _len_of_pt(self, path: pathlib.Path) -> int:
        data = self._load_pt(path)
        if isinstance(data, list):
            return len(data)
        raise ValueError(f"{path} is not list[Tensor]")

    @staticmethod
    def _extract_defect_type(filename: str) -> str:
        # 예: scratch_0001.pt → "scratch"
        return re.split(r"[_\-]", pathlib.Path(filename).stem)[0] or "unknown"

    # ── 필수 메서드 ───────────────────────────────────
    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx: int):
        ok_p, ng_p, full_p, img_idx, cls_id, dft_id = self.samples[idx]
        ok_lat  = self._load_pt(ok_p)[img_idx]     # (4,64,64)
        ng_lat  = self._load_pt(full_p)[img_idx]   # (4,64,64)  ← target
        cond    = self._load_pt(ng_p)[img_idx]     # (3,512,512) in [-1,1]

        sample = {
            "ok_lat":   ok_lat,
            "ng_lat":   ng_lat,
            "cond":     cond,
            "class_id":  torch.tensor(cls_id, dtype=torch.long),
            "defect_id": torch.tensor(dft_id, dtype=torch.long),
        }
        # 캐시 데이터 오염 방지
        return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sample.items()}

    # ── 디버그 ────────────────────────────────────────
    def cache_stats(self):
        if not self.cache_in_ram:
            return {"enabled": False}
        c = self._file_cache
        return {
            "enabled": True,
            "items": len(c._store),
            "hits": c.hits,
            "misses": c.misses,
            "max_items": c.max_items,
            "max_bytes": c.max_bytes,
            "cur_bytes": getattr(c, "cur_bytes", None),
        }
