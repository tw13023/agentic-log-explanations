"""
Log Normalizer for RAG retrieval.

Normalizes dynamic parameters (IP, UUID, HEX, paths, numbers) to placeholders
so that retrieval focuses on behavioral patterns, not specific values.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class NormalizationResult:
    """Result of log normalization."""
    normalized_text: str
    param_stats: Dict[str, int] = field(default_factory=dict)
    original_length: int = 0
    normalized_length: int = 0
    
    @property
    def compression_ratio(self) -> float:
        """How much the text was compressed by normalization."""
        if self.original_length == 0:
            return 0.0
        return 1 - (self.normalized_length / self.original_length)


class LogNormalizer:
    """
    Normalizes log messages by replacing dynamic parameters with placeholders.
    
    This is crucial for RAG retrieval - we want to match behavioral patterns,
    not specific IP addresses or timestamps.
    """
    
    # Regex patterns for various dynamic elements
    PATTERNS = [
        # IPv4 addresses (e.g., 192.168.1.1)
        (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IPV4>'),
        
        # IPv6 addresses (simplified)
        (r'\b[0-9a-fA-F:]{7,39}\b', '<IPV6>'),
        
        # MAC addresses
        (r'\b([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b', '<MAC>'),
        
        # UUIDs (e.g., 550e8400-e29b-41d4-a716-446655440000)
        (r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', '<UUID>'),
        
        # Hex values (8+ chars, common in logs)
        (r'\b0x[0-9a-fA-F]+\b', '<HEX>'),
        (r'\b[0-9a-fA-F]{8,}\b', '<HEX>'),
        
        # HDFS Block IDs (e.g., blk_-1608999687919862906)
        (r'\bblk_-?\d+\b', '<BLOCK_ID>'),
        
        # File paths (Unix-style)
        (r'\/[\w\-\.\/]+', '<PATH>'),
        
        # URLs
        (r'https?:\/\/[^\s]+', '<URL>'),
        
        # Email addresses
        (r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '<EMAIL>'),
        
        # Timestamps (various formats)
        (r'\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b', '<TIMESTAMP>'),
        (r'\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b', '<TIME>'),
        (r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\b', '<DATE>'),
        
        # Memory addresses (e.g., 0x7fff5fbff8c0)
        (r'\b0x[0-9a-fA-F]{6,16}\b', '<MEMADDR>'),
        
        # Port numbers (after colon, 1-65535)
        (r':(\d{1,5})\b', ':<PORT>'),
        
        # Process/Thread IDs
        (r'\bpid[=:\s]*\d+\b', 'pid=<PID>'),
        (r'\btid[=:\s]*\d+\b', 'tid=<TID>'),
        (r'\bthread[=:\s]*\d+\b', 'thread=<TID>'),
        
        # Generic large numbers (likely IDs)
        (r'\b\d{6,}\b', '<NUM>'),
        
        # Smaller numbers (keep some context)
        (r'\b\d{2,5}\b', '<NUM>'),
    ]
    
    def __init__(self, custom_patterns: Optional[List[Tuple[str, str]]] = None):
        """
        Initialize normalizer with optional custom patterns.
        
        Args:
            custom_patterns: List of (regex_pattern, replacement) tuples
        """
        self.patterns = self.PATTERNS.copy()
        if custom_patterns:
            # Custom patterns take priority (prepended)
            self.patterns = custom_patterns + self.patterns
        
        # Compile patterns for efficiency
        self.compiled_patterns = [
            (re.compile(pattern), replacement)
            for pattern, replacement in self.patterns
        ]
    
    def normalize_line(self, line: str) -> Tuple[str, Dict[str, int]]:
        """
        Normalize a single log line.
        
        Returns:
            Tuple of (normalized_line, param_counts)
        """
        param_counts: Dict[str, int] = {}
        normalized = line
        
        for pattern, replacement in self.compiled_patterns:
            matches = pattern.findall(normalized)
            if matches:
                # Track what we're replacing
                placeholder = replacement.strip('<>')
                param_counts[placeholder] = param_counts.get(placeholder, 0) + len(matches)
                normalized = pattern.sub(replacement, normalized)
        
        return normalized, param_counts
    
    def normalize_lines(self, lines: List[str], join: bool = True) -> NormalizationResult:
        """
        Normalize multiple log lines.
        
        Args:
            lines: List of log lines
            join: If True, join lines with newline; if False, keep as list
            
        Returns:
            NormalizationResult with normalized text and statistics
        """
        normalized_lines = []
        total_params: Dict[str, int] = {}
        
        for line in lines:
            norm_line, params = self.normalize_line(line)
            normalized_lines.append(norm_line)
            
            # Aggregate param counts
            for key, count in params.items():
                total_params[key] = total_params.get(key, 0) + count
        
        original_text = "\n".join(lines)
        
        if join:
            normalized_text = "\n".join(normalized_lines)
        else:
            normalized_text = normalized_lines  # type: ignore
        
        return NormalizationResult(
            normalized_text=normalized_text,
            param_stats=total_params,
            original_length=len(original_text),
            normalized_length=len(normalized_text) if isinstance(normalized_text, str) else sum(len(l) for l in normalized_text)
        )
    
    def normalize_session(self, session) -> NormalizationResult:
        """
        Normalize a Session object's lines.
        
        Args:
            session: Session object with .lines attribute
            
        Returns:
            NormalizationResult
        """
        return self.normalize_lines(session.lines)


class BGLNormalizer(LogNormalizer):
    """
    BGL-specific normalizer with patterns tuned for BlueGene/L logs.
    """
    
    BGL_PATTERNS = [
        # BGL-specific node identifiers (e.g., R00-M0-N0-C:J00-U00)
        (r'\bR\d{2}-M\d-N\d-C:J\d{2}-U\d{2}\b', '<NODE>'),
        (r'\bR\d{2}-M\d-N\d(?:-C)?(?:-J\d{2})?(?:-U\d{2})?\b', '<NODE>'),
        
        # BGL core/processor IDs
        (r'\bcore\.\d+\b', 'core.<CORE>'),
        
        # BGL-specific hex identifiers
        (r'\b[0-9a-fA-F]{8}\b', '<HEX8>'),
        
        # DDR errors, memory locations
        (r'\bDDR\(\d+,\d+,\d+\)', 'DDR(<MEMLOC>)'),
        
        # Torus coordinates
        (r'\(\d+,\d+,\d+\)', '(<COORD>)'),
    ]
    
    def __init__(self):
        super().__init__(custom_patterns=self.BGL_PATTERNS)


class HDFSNormalizer(LogNormalizer):
    """
    HDFS-specific normalizer with patterns tuned for Hadoop logs.
    """
    
    HDFS_PATTERNS = [
        # HDFS Block IDs (already in base, but prioritize)
        (r'\bblk_-?\d+\b', '<BLOCK>'),
        
        # DataNode identifiers
        (r'\b\d+\.\d+\.\d+\.\d+:\d+\b', '<DATANODE>'),
        
        # Hadoop-specific paths
        (r'/user/[\w/]+', '<HDFS_PATH>'),
        (r'/tmp/[\w/]+', '<TMP_PATH>'),
        
        # Replication info
        (r'replicas=\d+', 'replicas=<NUM>'),
        (r'size=\d+', 'size=<SIZE>'),
    ]
    
    def __init__(self):
        super().__init__(custom_patterns=self.HDFS_PATTERNS)


def get_normalizer(dataset: str) -> LogNormalizer:
    """Factory function to get dataset-specific normalizer."""
    if dataset.upper() == "BGL":
        return BGLNormalizer()
    elif dataset.upper() == "HDFS":
        return HDFSNormalizer()
    else:
        return LogNormalizer()


# Quick test
if __name__ == "__main__":
    # Test samples
    bgl_samples = [
        "1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected",
        "- 1117838573 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.53.002423 R02-M1-N0-C:J12-U11 RAS APP FATAL ciod: Error reading message prefix on CioStream socket to 172.16.96.116:33850",
    ]
    
    hdfs_samples = [
        "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
        "081109 203518 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.19.102:50010 is added to blk_-1608999687919862906 size 67108864",
    ]
    
    print("="*60)
    print("BGL Normalization Test")
    print("="*60)
    
    bgl_norm = BGLNormalizer()
    for sample in bgl_samples:
        result = bgl_norm.normalize_lines([sample])
        print(f"\nOriginal: {sample[:80]}...")
        print(f"Normalized: {result.normalized_text[:80]}...")
        print(f"Params: {result.param_stats}")
        print(f"Compression: {result.compression_ratio:.1%}")
    
    print("\n" + "="*60)
    print("HDFS Normalization Test")
    print("="*60)
    
    hdfs_norm = HDFSNormalizer()
    for sample in hdfs_samples:
        result = hdfs_norm.normalize_lines([sample])
        print(f"\nOriginal: {sample[:80]}...")
        print(f"Normalized: {result.normalized_text[:80]}...")
        print(f"Params: {result.param_stats}")
        print(f"Compression: {result.compression_ratio:.1%}")
