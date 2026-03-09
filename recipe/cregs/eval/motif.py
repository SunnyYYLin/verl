import pandas as pd
import subprocess
import os
import tempfile
import glob
import logging
from pathlib import Path
from typing import Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
NUM_PROCESSES = max(1, (os.cpu_count() or 1) // 4)

class MotifScorer:
    def __init__(self, motif_file: Optional[str|Path] = None):
        if motif_file is None:
            motif_file = self._find_default_motif_file()
            logger.info(f"Using auto-discovered motif file: {motif_file}")
            
        self.motif_file = Path(motif_file).expanduser()
        if not self.motif_file.exists():
            raise FileNotFoundError(f"Motif file not found: {self.motif_file}")
        
    def _find_default_motif_file(self) -> Path:
        """Find the default known.motifs file in the HOMER installation directory."""
        # Try to find homer2/homer in the current path
        homer_bin = subprocess.check_output(['which', 'homer2'], text=True).strip()
        if not homer_bin:
            homer_bin = subprocess.check_output(['which', 'homer'], text=True).strip()
        
        if not homer_bin:
            raise RuntimeError("HOMER not found in PATH. Cannot auto-discover motif file.")
        
        # Possible locations relative to bin
        # Path(homer_bin).resolve().parent.parent / 'share/homer/data/knownTFs/known.motifs'
        possible_paths = [
            Path(homer_bin).resolve().parents[1] / 'data/knownTFs/known.motifs',
            Path(homer_bin).resolve().parents[1] / 'share/homer/data/knownTFs/known.motifs',
        ]
        
        for p in possible_paths:
            if p.exists():
                return p
                
        raise FileNotFoundError("Could not find default known.motifs in HOMER directory.")

    def _parse_homer_output_text(self, text: str) -> pd.DataFrame:
        """Parse HOMER "find" output text and return DataFrame with columns `seq_idx` and `motif_score`."""
        output_data = []
        for line in (text or "").strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 6:
                seq_id = parts[0]
                try:
                    seq_id = int(seq_id)
                except (ValueError, TypeError):
                    pass
                try:
                    score = float(parts[5])
                except (ValueError, TypeError):
                    continue
                output_data.append({"seq_idx": seq_id, "motif_score": score})
        return pd.DataFrame(output_data)

    def _parse_homer_output_file(self, path: str|Path) -> pd.DataFrame:
        with open(path, "r") as fh:
            return self._parse_homer_output_text(fh.read())

    def score_sequences(self, sequences: pd.Series):
        """
        Score a series of sequences using HOMER.
        Returns a series of maximum scores for each sequence.
        """
        if sequences.empty:
            return pd.Series(dtype=float)

        tmp_dir = tempfile.mkdtemp()
        tmp_dir_path = Path(tmp_dir)
        fasta_path = tmp_dir_path / "seqs.fasta"
        scores_path = tmp_dir_path / "scores.txt"

        with open(fasta_path, "w") as f_in:
            for idx, seq in sequences.items():
                f_in.write(f">{idx}\n{seq}\n")

        cmd = [
                "homer2", "find", 
                "-i", str(fasta_path), 
                "-m", str(self.motif_file), 
                "-mscore", 
                "-o", str(scores_path),
                "-p", str(NUM_PROCESSES)
            ]
        
        try:
            logger.debug(f"正在运行 HOMER (homer2 find) 评分: {' '.join(cmd)}")
            # Set cwd to tmp_dir so .group and .seq files are created there and cleaned up automatically
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=tmp_dir)
            
            scores_df = pd.DataFrame()
            if scores_path.exists():
                scores_df = self._parse_homer_output_file(scores_path)
            elif result.stdout and result.stdout.strip():
                scores_df = self._parse_homer_output_text(result.stdout)
            
            if scores_df.empty:
                logger.warning("警告：没有检测到任何匹配的 Motif。")
                return pd.Series(float("nan"), index=sequences.index)

            # Max score per sequence
            max_scores = scores_df.groupby("seq_idx")["motif_score"].max()
            
            # Reindex to match input sequences
            return max_scores.reindex(sequences.index)

        except subprocess.CalledProcessError as e:
            logger.error(f"HOMER 运行出错: {e.stderr}")
            return pd.Series(float("nan"), index=sequences.index)

if __name__ == "__main__":
    from tap import Tap
    
    class Args(Tap):
        motif_file: Optional[Path] = None
        input_parquet: Path
        seq_key: str = 'responses'
    args = Args().parse_args()

    scorer = MotifScorer(args.motif_file)
    df = pd.read_parquet(args.input_parquet)
    
    seqs = df[args.seq_key]
    if seqs.dtype == list:
        logger.info(f"Input sequences are lists:\n{seqs[0][:10]}, exploding...")
        seqs = seqs.explode().reset_index(drop=True)

    scores = scorer.score_sequences(seqs)

    logger.info(f"Motif scoring completed. Scores:\n {scores}")
    scores.to_csv(args.input_parquet.with_suffix(".mscore.csv"), index_label="index", header=["mscore"])
    logger.info(f"Motif scores stats: \n{scores.describe()}")