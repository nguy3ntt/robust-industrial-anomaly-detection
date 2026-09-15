"""Dataset governance, manifest construction, and leakage auditing."""

from robust_deepfake_detection.data.dfdc import (
    DfdcAuditReport,
    DfdcBuildResult,
    DfdcConfig,
    DfdcConfigurationError,
    DfdcVideoRecord,
)
from robust_deepfake_detection.data.dfdc import (
    build_manifest as build_dfdc_manifest,
)
from robust_deepfake_detection.data.dfdc import (
    load_config as load_dfdc_config,
)
from robust_deepfake_detection.data.dfdc import (
    load_provider_metadata as load_dfdc_provider_metadata,
)
from robust_deepfake_detection.data.dfdc import (
    load_verified_manifest as load_verified_dfdc_manifest,
)
from robust_deepfake_detection.data.dfdc import (
    write_build_outputs as write_dfdc_build_outputs,
)
from robust_deepfake_detection.data.faceforensics import (
    AuditIssue,
    AuditReport,
    BuildResult,
    FaceForensicsConfig,
    ManifestVerificationError,
    VideoRecord,
    audit_records,
    build_manifest,
    load_config,
    load_identity_map,
    load_official_splits,
    load_verified_manifest,
    write_build_outputs,
)

__all__ = [
    "AuditIssue",
    "AuditReport",
    "BuildResult",
    "DfdcAuditReport",
    "DfdcBuildResult",
    "DfdcConfig",
    "DfdcConfigurationError",
    "DfdcVideoRecord",
    "FaceForensicsConfig",
    "ManifestVerificationError",
    "VideoRecord",
    "audit_records",
    "build_dfdc_manifest",
    "build_manifest",
    "load_config",
    "load_dfdc_config",
    "load_dfdc_provider_metadata",
    "load_identity_map",
    "load_official_splits",
    "load_verified_dfdc_manifest",
    "load_verified_manifest",
    "write_build_outputs",
    "write_dfdc_build_outputs",
]
