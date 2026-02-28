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

    def structural_summary(self, session) -> Optional[str]:
        """
        Build an optional structural summary for a session.

        Subclasses override this to inject dataset-specific structural
        annotations that help BM25 differentiate anomaly from normal sessions.

        Returns:
            A structural annotation string, or None if not applicable.
        """
        return None

    def normalize_signature(self, name: str) -> str:
        """
        Normalize a signature name to canonical form.

        Subclasses override this to fix LLM-generated signature names
        (e.g. mapping wrong component prefixes to correct ones).

        Returns:
            Normalized signature name.
        """
        return name


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

    _VALID_COMPONENTS = {"KERNEL", "APP", "MMCS", "LINKCARD"}

    # Maps error types to their correct BGL component
    _COMPONENT_MAP = {
        # CIOD-related → APP
        "CIOD_STREAM_ERROR": "APP",
        "CIOD_ERROR": "APP",
        "CIOD_SOCKET_ERROR": "APP",
        "CIOD_NODE_MAP_ERROR": "APP",
        "CIOD_SIGNAL_RECEIVED": "APP",
        "CIOD_MESSAGE_ERROR": "APP",
        "CIOD_PROGRAM_IMAGE_ERROR": "APP",
        "LOGIN_CHDIR_FAILED": "APP",
        "EXEC_FORMAT_ERROR": "APP",
        "DEVICE_RESOURCE_BUSY": "APP",
        # VPD / link-card → LINKCARD
        "NODE_CARD_VPD_CHECK": "LINKCARD",
        "NODE_CARD_STATUS_ERROR": "LINKCARD",
        "HARDWARE_WARNING": "LINKCARD",
        "DISCOVERY_ERROR": "LINKCARD",
        "MIDPLANE_SWITCH_ERROR": "LINKCARD",
        "MONITOR_FAILURE": "LINKCARD",
    }

    # Consolidate duplicate error types → canonical form
    _ERROR_TYPE_CANONICAL = {
        # TLB variations
        "DATA_TLB_ERROR_INTERRUPT": "DATA_TLB_ERROR",
        "TLB_ERROR": "DATA_TLB_ERROR",
        # CIOD variations
        "CIOD_SOCKET_ERROR": "CIOD_STREAM_ERROR",
        "CIOD_ERROR": "CIOD_STREAM_ERROR",
        "CIOD_UNEXPECTED_EOF": "CIOD_STREAM_ERROR",
        "CIOD_SIGNAL_ERROR": "CIOD_STREAM_ERROR",
        "CIOD_SIGNAL_15": "CIOD_STREAM_ERROR",
        "UNEXPECTED_EOF": "CIOD_STREAM_ERROR",
        # Machine check
        "MACHINE_CHECK_INTERRUPT": "MACHINE_CHECK",
        "MACHINE_CHECK_STATUS_REGISTER": "MACHINE_CHECK",
        "MACHINE_CHECK_ENABLE": "MACHINE_CHECK",
        "MACHINE_CHECK_ENABLE_0": "MACHINE_CHECK",
        # Login
        "LOGIN_CHDIR_FAILURE": "LOGIN_CHDIR_FAILED",
        "LOGIN_CHDIR_ERROR": "LOGIN_CHDIR_FAILED",
        "CIOD_LOGIN_CHDIR_FAILED": "LOGIN_CHDIR_FAILED",
        "CIOD_LOGIN_ERROR": "LOGIN_CHDIR_FAILED",
        "CIOD_LOGIN_FAILED": "LOGIN_CHDIR_FAILED",
        "CIOD_LOGIN_FAILURE": "LOGIN_CHDIR_FAILED",
        # Floating point
        "FLOATING_POINT_INSTR_ENABLED": "FLOATING_POINT_ERROR",
        "FLOATING_PT_EX_MODE_0_ENABLE": "FLOATING_POINT_ERROR",
        "FLOATING_POINT_ALIGNMENT_EXCEPTIONS": "FLOATING_POINT_ERROR",
        # DDR
        "DDR_ERRORS": "DDR_ERROR",
        "DDR_ERROR_APP_FATAL_CIOD_ERROR": "DDR_ERROR",
        # VPD
        "NODE_CARD_VPD_CHECK_FAILURE": "NODE_CARD_VPD_CHECK",
        "NODE_CARD_VPD_CHECK_ERROR": "NODE_CARD_VPD_CHECK",
        "NODE_CARD_VPD_CHECK_FAILED": "NODE_CARD_VPD_CHECK",
        "NODE_VPD_CHECK_FAILURE": "NODE_CARD_VPD_CHECK",
        "VPD_CHECK_FAILURE": "NODE_CARD_VPD_CHECK",
        "VPD_MISMATCH": "NODE_CARD_VPD_CHECK",
        # L3 / EDRAM
        "L3_MAJOR_INTERNAL_ERROR": "L3_INTERNAL_ERROR",
        "L3_EDRAM_ERROR": "EDRAM_ERROR",
        # Alignment
        "INTEGER_ALIGNMENT_EXCEPTION": "INTEGER_ALIGNMENT_ERROR",
        "INTEGER_ALIGNMENT_EXCEPTIONS": "INTEGER_ALIGNMENT_ERROR",
        # Parity errors → single canonical
        "PARITY_ERROR_IN_READ_QUEUE_PLB": "PARITY_ERROR",
        "PARITY_ERROR_IN_READ_QUEUE": "PARITY_ERROR",
        "INSTRUCTION_CACHE_PARITY_ERROR": "PARITY_ERROR",
        "D_CACHE_SEARCH_PARITY_ERROR": "PARITY_ERROR",
        "L2_DCACHE_UNIT_DATA_PARITY_ERROR": "PARITY_ERROR",
        # Interrupt _ENABLE suffixes
        "EXTERNAL_INPUT_INTERRUPT_ENABLE": "EXTERNAL_INPUT_INTERRUPT",
        "CRITICAL_INPUT_INTERRUPT_ENABLE": "CRITICAL_INPUT_INTERRUPT",
        "CRITICAL_INPUT_INTERRUPT_ENABLE_0": "CRITICAL_INPUT_INTERRUPT",
        # Missing fields
        "MISSING_INVALID_FIELDS": "MISSING_OR_INVALID_FIELDS",
        # IDO / libido proxy
        "IDOPROXY_COMMUNICATION_FAILURE": "IDO_PROXY_COMMUNICATION_FAILURE",
        "LIB_IDO_ERROR_1019_SOCKET_CLOSED": "IDO_PROXY_COMMUNICATION_FAILURE",
        "LIBIDO_ERROR": "IDO_PROXY_COMMUNICATION_FAILURE",
        # Torus / retransmission
        "RETRANSMISSION_ERROR": "TORUS_RECEIVER_ERROR",
        "TORUS_NON_RECOVERABLE_ERROR": "TORUS_RECEIVER_ERROR",
        # Tree network
        "TREE_RECEIVER_ERROR": "TREE_NETWORK_PACKET_ERROR",
        "SENDING_PACKET_ON_TREE_NETWORK": "TREE_NETWORK_PACKET_ERROR",
        # RTS
        "RTS": "RTS_INTERNAL_ERROR",
        "RTS_TREE_LINK_TRAINING_FAILED": "RTS_INTERNAL_ERROR",
        "RTS_TREE_TORUS_LINK_TRAINING_FAILED": "RTS_INTERNAL_ERROR",
        "RTS_TERMINATED": "KERNEL_TERMINATED",
        "KERNEL_PANIC": "KERNEL_TERMINATED",
        # Node card status
        "NODE_CARD_STATUS": "NODE_CARD_STATUS_ERROR",
        "NODE_CARD_NOT_FULLY_FUNCTIONAL": "NODE_CARD_STATUS_ERROR",
        "NODE_CARD_POWER_MODULE_NOT_ACCESSIBLE": "NODE_CARD_STATUS_ERROR",
        # Hardware severity variants → LINKCARD warning
        "HARDWARE_SEVERE": "HARDWARE_WARNING",
        # Register dumps
        "GENERAL_PURPOSE_REGISTERS": "REGISTER_DUMP",
        "GENERATING_CORE": "REGISTER_DUMP",
        # CE/ECC symbol errors
        "CE_SYM_10": "CE_SYM_ERROR",
        # Icache prefetch
        "ICACHE_PREFETCH_THRESHOLD_0": "ICACHE_PREFETCH_ERROR",
        "ICACHE_PREFETCH_THRESHOLD_ERROR": "ICACHE_PREFETCH_ERROR",
        # Verbose literal → canonical
        "LINK_SEVERED": "LOAD_MESSAGE_ERROR",
        "RECEIVING_PACKET": "TREE_NETWORK_PACKET_ERROR",
        "ERROR_RECEIVING_PACKET": "TREE_NETWORK_PACKET_ERROR",
        "CREATING_NODE_MAP": "NODE_MAP_ERROR",
        "ERROR_CREATING_NODE_MAP": "NODE_MAP_ERROR",
        "NODE_MAP_CREATION_ERROR": "NODE_MAP_ERROR",
        "TERMINATION": "KERNEL_TERMINATED",
        "CHECK_INITIAL_GLOBAL_INTERRUPT_VALUES": "EXTERNAL_INPUT_INTERRUPT",
    }

    # Error type names where the FATAL_ prefix is part of the canonical name
    _SEVERITY_PREFIX_SAFE = {"FATAL_ERROR", "FATAL_MESSAGE"}

    # Severity-only error types that should map to a default
    _SEVERITY_ONLY = {"FATAL": "FATAL_ERROR", "INFO": "INFO"}

    def __init__(self):
        super().__init__(custom_patterns=self.BGL_PATTERNS)

    def normalize_signature(self, name: str) -> str:
        """
        Normalize a BGL signature name to canonical form.

        Strips severity labels (INFO, WARN, ERROR, FATAL, WARNING, SEVERE)
        from any position, handles RAS_ prefixes, collapses verbose
        LLM-generated names, and maps to correct components.

        Handles all LLM output patterns::

            KERNEL__FATAL__DATA_TLB_ERROR          → KERNEL__DATA_TLB_ERROR
            KERNEL__FATAL_data_TLB_ERROR           → KERNEL__DATA_TLB_ERROR
            KERNEL__FATAL__data TLB error interrupt → KERNEL__DATA_TLB_ERROR
            RAS_APP_FATAL__CIOD_STREAM_ERROR       → APP__CIOD_STREAM_ERROR
            KERNEL__FATAL__kernel terminated for reason 1001
                                                   → KERNEL__KERNEL_TERMINATED
        """
        upper = name.upper()

        # Handle names without __ that match known patterns
        if "__" not in name:
            cleaned = re.sub(r'[^A-Z0-9_]', '_', upper).strip('_')
            if 'RTS_PANIC' in cleaned:
                return "KERNEL__KERNEL_TERMINATED"
            return upper

        severity_labels = {"INFO", "WARN", "ERROR", "FATAL", "WARNING", "SEVERE"}

        # ── 1. Split on __ and strip pure severity segments ──
        segments = [s.upper() for s in name.split("__")]
        non_sev = [s for s in segments if s not in severity_labels]

        if len(non_sev) < 2:
            # All error-type segments were severity labels; keep the last one
            # e.g. KERNEL__FATAL → prefix=KERNEL, error_type=FATAL
            segments = [segments[0]] + [segments[-1]] if len(segments) >= 2 else segments
        else:
            segments = non_sev

        if len(segments) < 2:
            return upper

        # ── 2. Clean component prefix ──
        prefix = segments[0]

        # Strip RAS_ prefix  (RAS_APP_FATAL → APP_FATAL → APP)
        if prefix.startswith("RAS_"):
            prefix = prefix[4:]

        # Strip severity suffix from component  (APP_FATAL → APP)
        for suffix in ("_FATAL", "_INFO", "_WARN", "_ERROR", "_WARNING", "_SEVERE"):
            if prefix.endswith(suffix):
                prefix = prefix[: -len(suffix)]
                break

        # ── 3. Rejoin error type segments ──
        error_type = "_".join(segments[1:]) if len(segments) > 2 else segments[1]

        # Normalize whitespace and special characters
        error_type = error_type.replace(" ", "_")
        error_type = re.sub(r'[^A-Z0-9_]', '_', error_type)
        error_type = re.sub(r'_+', '_', error_type)
        error_type = error_type.strip('_')

        # ── 4. Verbose pattern collapsing (before severity-prefix strip) ──
        if ('KERNEL_TERMINATED' in error_type
                or 'TERMINATED_FOR_REASON' in error_type):
            error_type = "KERNEL_TERMINATED"
        elif 'RTS_PANIC' in error_type:
            error_type = "KERNEL_TERMINATED"
        elif ('CIOSTREAM' in error_type
              or ('CIOD' in error_type
                  and 'ERROR_READING_MESSAGE' in error_type)
              or ('CIOD' in error_type
                  and 'FAILED_TO_READ_MESSAGE_PREFIX' in error_type)):
            error_type = "CIOD_STREAM_ERROR"
        elif ('RECEIVING_PACKET' in error_type
              and 'TREE_NETWORK' in error_type):
            error_type = "NETWORK_RECEIVE_ERROR"
        elif ('MOUNT' in error_type
              and ('UNABLE' in error_type or 'FILESYSTEM' in error_type)):
            error_type = "LUSTRE_MOUNT_FAILED"
        elif ('BAD_MESSAGE_HEADER' in error_type
              and error_type != 'BAD_MESSAGE_HEADER'):
            error_type = "BAD_MESSAGE_HEADER"
        elif error_type == "APP_FATAL":
            error_type = "FATAL_ERROR"
        # Coordinate exceeds dimension (any axis)
        elif 'COORDINATE_EXCEEDS' in error_type:
            error_type = "COORDINATE_EXCEEDS_DIMENSION"
        # Midplane switch controller verbose names
        elif 'MIDPLANESWITCHCONTROLLER' in error_type:
            error_type = "MIDPLANE_SWITCH_ERROR"
        # CIOD program image / loading errors (extremely verbose paths)
        elif ('CIOD' in error_type
              and ('PROGRAM_IMAGE' in error_type
                   or 'ERROR_LOADING' in error_type)):
            error_type = "CIOD_PROGRAM_IMAGE_ERROR"
        # LR/CR/XER/CTR register dumps
        elif ('LR_' in error_type and 'CR_' in error_type
              and 'XER_' in error_type):
            error_type = "REGISTER_DUMP"
        # Torus sender retransmission verbose names
        elif ('TORUS_SENDER' in error_type
              and 'RETRANSMISSION' in error_type):
            error_type = "TORUS_RECEIVER_ERROR"
        # Duplicate canonical rank mapping (verbose node map error)
        elif 'DUPLICATE_CANONICAL_RANK' in error_type:
            error_type = "NODE_MAP_ERROR"
            prefix = "APP"
        # PrepareForService / nodecard shutdown
        elif 'PREPAREFORSERVICE' in error_type:
            error_type = "NODE_CARD_STATUS_ERROR"
            prefix = "LINKCARD"

        # ── 4b. Non-standard component prefix → fold into error type ──
        _VALID = self._VALID_COMPONENTS
        if prefix not in _VALID:
            # e.g. HARDWARE__WARNING → error_type = HARDWARE_WARNING
            error_type = f"{prefix}_{error_type}"
            prefix = "KERNEL"

        # ── 5. Strip severity PREFIX from error type ──
        #   FATAL_DATA_TLB_ERROR → DATA_TLB_ERROR
        #   but keep FATAL_ERROR, FATAL_MESSAGE as-is
        if error_type not in self._SEVERITY_PREFIX_SAFE:
            for sev in ("FATAL_", "INFO_", "WARN_", "WARNING_",
                        "ERROR_", "SEVERE_"):
                if error_type.startswith(sev):
                    error_type = error_type[len(sev):]
                    break

        # ── 6. Canonical error type ──
        # Handle severity-only error types (e.g. KERNEL__FATAL → KERNEL__FATAL_ERROR)
        if error_type in self._SEVERITY_ONLY:
            error_type = self._SEVERITY_ONLY[error_type]
        error_type = self._ERROR_TYPE_CANONICAL.get(error_type, error_type)

        # ── 7. Determine correct component ──
        component = self._COMPONENT_MAP.get(error_type, None)
        if component is None:
            component = prefix if prefix in self._VALID_COMPONENTS else "KERNEL"

        return f"{component}__{error_type}"


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

    # Maps RAS_*-style prefixes to correct HDFS component names
    _COMPONENT_MAP = {
        "BLOCK_WRITE_FAILURE": "DATANODE",
        "BLOCK_RECEIVING_FAILURE": "DATANODE",
        "BLOCK_RECEIVING_ERROR": "DATANODE",
        "BLOCK_RECEIVE_FAILURE": "DATANODE",
        "BLOCK_RECEIVER_ERROR": "DATANODE",
        "BLOCK_RECEIVE_INTERRUPT": "DATANODE",
        "BLOCK_READ_FAILURE": "DATANODE",
        "BLOCK_SERVING_FAILURE": "DATANODE",
        "BLOCK_TRANSFER_FAILURE": "DATANODE",
        "BLOCK_TRANSFER_TIMEOUT": "DATANODE",
        "BLOCK_DELETE_FAILURE": "DATANODE",
        "BLOCK_DELETION_FAILURE": "DATANODE",
        "RETRANSMIT_REQUEST": "DATANODE",
        "DATA_STORAGE_INTERRUPT": "FSDATASET",
        "REPLICATION_INCOMPLETE": "NAMENODE",
        "REDUNDANT_ADDSTOREDBLOCK_REQUEST": "NAMENODE",
        "REDUNDANT_ADDSTOREDBLOCK": "NAMENODE",
        "PENDING_REPLICATION_MONITOR_TIMED_OUT": "NAMENODE",
        "PENDING_REPLICATION_TIMEOUT": "NAMENODE",
        # Structural anomaly tags
        "INCOMPLETE_PIPELINE": "NAMENODE",
        "MISSING_ACKNOWLEDGMENT": "DATANODE",
        "EXCESS_REPLICATION": "NAMENODE",
    }

    # Consolidate duplicate error types -> canonical form
    _ERROR_TYPE_CANONICAL = {
        "BLOCK_RECEIVING_FAILURE": "BLOCK_RECEIVE_FAILURE",
        "BLOCK_RECEIVING_ERROR": "BLOCK_RECEIVE_FAILURE",
        "BLOCK_RECEIVER_ERROR": "BLOCK_RECEIVE_FAILURE",
        "BLOCK_RECEIVE_INTERRUPT": "BLOCK_RECEIVE_FAILURE",
        "BLOCK_DELETE_FAILURE": "BLOCK_DELETION_FAILURE",
        "BLOCK_DELETE_FAILED": "BLOCK_DELETION_FAILURE",
        "BLOCK_DELETION_FAILED": "BLOCK_DELETION_FAILURE",
        "REDUNDANT_ADDSTOREDBLOCK_REQUEST": "REDUNDANT_STORED_BLOCK",
        "REDUNDANT_ADDSTOREDBLOCK": "REDUNDANT_STORED_BLOCK",
        "PENDING_REPLICATION_MONITOR_TIMED_OUT": "PENDING_REPLICATION_TIMEOUT",
        "BLOCK_REPLICATION_FAILURE": "BLOCK_REPLICATION_FAILED",
        "BLOCK_REPLICATION_CONFLICT": "BLOCK_REPLICATION_FAILED",
        "BLOCK_REPLICATION_OVERLAP": "BLOCK_REPLICATION_FAILED",
        "BLOCK_REPLICAS_MISSING": "REPLICATION_INCOMPLETE",
        "WRITE_BLOCK_FAILURE": "BLOCK_WRITE_FAILURE",
        "WRITE_PIPELINE_FAILED": "WRITE_PIPELINE_FAILURE",
        "BLOCK_REPLICATION_SUCCESSFUL": "BLOCK_DUPLICATION",
        "RECEIVING_BLOCK_DUPLICATION": "BLOCK_DUPLICATION",
        "BLOCK_READ_FAILURE": "BLOCK_SERVING_FAILURE",
        "RECEIVING_EMPTY_PACKET": "BLOCK_RECEIVE_FAILURE",
        "RECEIVING_BLOCK_FAILURE": "BLOCK_RECEIVE_FAILURE",
        "BLOCK_RECEIVING_DUPLICATE": "BLOCK_DUPLICATION",
        "WRITE_PIPELINE_EXCEPTION": "WRITE_PIPELINE_FAILURE",
        "IO_EXCEPTION": "WRITE_PIPELINE_FAILURE",
        "REDUNDANT_ADDSTOREDBLOCK_REQUEST_RECEIVED": "REDUNDANT_STORED_BLOCK",
        "BLOCK_RECEIVING_FAILED": "BLOCK_RECEIVE_FAILURE",
        "REDUNDANT_ADD_STORED_BLOCK_REQUEST": "REDUNDANT_STORED_BLOCK",
        # Structural anomaly tags (from structural_summary)
        "INCOMPLETE_PIPELINE": "INCOMPLETE_PIPELINE",
        "MISSING_ACKNOWLEDGMENT": "MISSING_ACKNOWLEDGMENT",
        "EXCESS_REPLICATION": "EXCESS_REPLICATION",
    }

    def __init__(self):
        super().__init__(custom_patterns=self.HDFS_PATTERNS)

    def structural_summary(self, session) -> Optional[str]:
        """
        Build a structural summary for HDFS sessions.

        HDFS anomalies are structurally distinct (same vocabulary, different
        sequence patterns). This injects discriminative tokens that BM25 can
        use to differentiate anomaly from normal sessions.

        Returns:
            Structural annotation string, e.g.:
            "STRUCTURAL: receives=4 received=2 allocate=1 responder=3
             INCOMPLETE_PIPELINE EXCESS_REPLICATION"
        """
        text = "\n".join(session.lines).lower()

        # Count key HDFS operations
        receives = len(re.findall(r'receiving block', text))
        received = len(re.findall(r'received block', text))
        allocate = len(re.findall(r'allocateblock|namesystem\.allocateblock', text))
        addstoredblock = len(re.findall(r'addstoredblock', text))
        responder = len(re.findall(r'packetresponder', text))
        exceptions = len(re.findall(r'exception|error|failed', text))
        writeblock = len(re.findall(r'writeblock', text))
        delete = len(re.findall(r'delete block|invalidate', text))

        # Build counts line
        parts = [
            "STRUCTURAL:",
            f"receives={receives}",
            f"received={received}",
            f"allocate={allocate}",
            f"addstoredblock={addstoredblock}",
            f"responder={responder}",
            f"exceptions={exceptions}",
        ]

        # Add discriminative tags based on structural anomalies
        tags = []
        if receives > 0 and received < receives:
            tags.append("INCOMPLETE_PIPELINE")
        if addstoredblock > 3:
            tags.append("EXCESS_REPLICATION")
        if exceptions > 0:
            tags.append("HAS_EXCEPTION")
        if writeblock > 0 and exceptions > 0:
            tags.append("WRITE_FAILURE")
        if delete > 0:
            tags.append("BLOCK_DELETION")
        if receives > 0 and responder == 0:
            tags.append("MISSING_ACKNOWLEDGMENT")
        if addstoredblock > 0 and allocate == 0:
            tags.append("ORPHAN_BLOCK")

        if tags:
            parts.extend(tags)
        else:
            parts.append("NORMAL_FLOW")

        return " ".join(parts)

    def normalize_signature(self, name: str) -> str:
        """
        Normalize an HDFS signature name to use correct component prefixes.

        Strips severity labels (INFO, WARN, ERROR, FATAL) from any position,
        maps to correct HDFS components, and consolidates duplicate error types
        to canonical forms.

        Handles all LLM output patterns:
          - DATANODE_INFO__BLOCK_WRITE_FAILURE  (severity as prefix suffix)
          - DATANODE__INFO__BLOCK_WRITE_FAILURE  (severity as middle segment)
          - DATANODE__BLOCK_WRITE_FAILURE        (already clean)

        Args:
            name: Raw LLM-generated signature name

        Returns:
            Normalized name like "DATANODE__BLOCK_WRITE_FAILURE"
        """
        if "__" not in name:
            return name

        # Split into all segments and strip severity from any position
        severity_labels = {"INFO", "WARN", "ERROR", "FATAL"}
        segments = [s.upper() for s in name.split("__")]
        segments = [s for s in segments if s not in severity_labels]

        if len(segments) < 2:
            # Degenerate — only one segment left after stripping
            return name

        # First segment is the component prefix (may have _INFO etc. as suffix)
        prefix_upper = segments[0]
        for suffix in ("_INFO", "_WARN", "_ERROR", "_FATAL"):
            if prefix_upper.endswith(suffix):
                prefix_upper = prefix_upper[: -len(suffix)]
                break

        # Remaining segments rejoin as the error type
        error_upper = "_".join(segments[1:]) if len(segments) > 2 else segments[1]

        # Normalize spaces to underscores (LLM sometimes uses spaces)
        error_upper = error_upper.replace(" ", "_")

        # --- Determine correct HDFS component ---
        component = self._COMPONENT_MAP.get(error_upper, None)

        if component is None:
            valid_prefixes = ("DATANODE", "NAMENODE", "FSDATASET", "BLOCKSCANNER")
            if prefix_upper in valid_prefixes:
                component = prefix_upper
            else:
                component = "DATANODE"  # safe fallback

        # --- Canonical error type ---
        canonical_error = self._ERROR_TYPE_CANONICAL.get(error_upper, error_upper)

        return f"{component}__{canonical_error}"


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
