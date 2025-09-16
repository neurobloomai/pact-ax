"""
PACT-AX: Agent Collaboration Layer
Context Security Audit

Comprehensive audit system that provides observability, compliance reporting,
and forensic analysis for all security operations while learning from patterns
to improve future security decisions.
"""

from typing import Dict, Any, Optional, List, Tuple, Set, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timezone, timedelta
import json
import hashlib
import uuid
import threading
import sqlite3
import gzip
import base64
from collections import defaultdict, deque
from pathlib import Path

# Import from sibling modules
from ..context_share.schemas import (
    ContextType, TrustLevel, AgentIdentity, ContextPacket, 
    CollaborationOutcome
)
from ..context_share.encryption import EncryptionLevel
from .manager import SecurityEvent, SecurityEventType, ThreatLevel


class AuditLevel(Enum):
    """Audit detail levels"""
    MINIMAL = 1      # Only critical events
    STANDARD = 2     # Normal security events
    DETAILED = 3     # All security operations
    FORENSIC = 4     # Maximum detail for investigations
    DEBUG = 5        # Development and debugging


class ComplianceStandard(Enum):
    """Supported compliance standards"""
    HIPAA = "HIPAA"
    GDPR = "GDPR"
    SOX = "SOX"
    PCI_DSS = "PCI_DSS"
    ISO27001 = "ISO27001"
    NIST = "NIST"
    CUSTOM = "CUSTOM"


class AuditEventSeverity(Enum):
    """Severity levels for audit events"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class AuditEvent:
    """Comprehensive audit event with full context"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: str = "security_audit"
    severity: AuditEventSeverity = AuditEventSeverity.MEDIUM
    
    # Agent and context information
    agent_from: Optional[str] = None
    agent_to: Optional[str] = None
    context_type: Optional[ContextType] = None
    context_packet_id: Optional[str] = None
    
    # Security details
    security_decision: Dict[str, Any] = field(default_factory=dict)
    policy_applied: List[str] = field(default_factory=list)
    encryption_level: Optional[EncryptionLevel] = None
    trust_level: Optional[float] = None
    threat_assessment: Optional[ThreatLevel] = None
    
    # Outcome and impact
    outcome: Optional[str] = None
    error_details: Optional[str] = None
    impact_score: float = 0.0
    
    # Compliance and regulatory
    compliance_requirements: List[ComplianceStandard] = field(default_factory=list)
    regulatory_notes: Dict[str, Any] = field(default_factory=dict)
    
    # Technical details
    payload_hash: Optional[str] = None
    lineage_trace: List[str] = field(default_factory=list)
    system_context: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary for storage"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "severity": self.severity.value,
            "agent_from": self.agent_from,
            "agent_to": self.agent_to,
            "context_type": self.context_type.value if self.context_type else None,
            "context_packet_id": self.context_packet_id,
            "security_decision": self.security_decision,
            "policy_applied": self.policy_applied,
            "encryption_level": self.encryption_level.value if self.encryption_level else None,
            "trust_level": self.trust_level,
            "threat_assessment": self.threat_assessment.value if self.threat_assessment else None,
            "outcome": self.outcome,
            "error_details": self.error_details,
            "impact_score": self.impact_score,
            "compliance_requirements": [c.value for c in self.compliance_requirements],
            "regulatory_notes": self.regulatory_notes,
            "payload_hash": self.payload_hash,
            "lineage_trace": self.lineage_trace,
            "system_context": self.system_context,
            "tags": list(self.tags),
            "correlation_id": self.correlation_id,
            "session_id": self.session_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create audit event from dictionary"""
        event = cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(timezone.utc),
            event_type=data.get("event_type", "security_audit"),
            severity=AuditEventSeverity(data.get("severity", AuditEventSeverity.MEDIUM.value)),
            agent_from=data.get("agent_from"),
            agent_to=data.get("agent_to"),
            context_packet_id=data.get("context_packet_id"),
            security_decision=data.get("security_decision", {}),
            policy_applied=data.get("policy_applied", []),
            trust_level=data.get("trust_level"),
            outcome=data.get("outcome"),
            error_details=data.get("error_details"),
            impact_score=data.get("impact_score", 0.0),
            regulatory_notes=data.get("regulatory_notes", {}),
            payload_hash=data.get("payload_hash"),
            lineage_trace=data.get("lineage_trace", []),
            system_context=data.get("system_context", {}),
            correlation_id=data.get("correlation_id"),
            session_id=data.get("session_id")
        )
        
        if data.get("context_type"):
            event.context_type = ContextType(data["context_type"])
        if data.get("encryption_level"):
            event.encryption_level = EncryptionLevel(data["encryption_level"])
        if data.get("threat_assessment"):
            event.threat_assessment = ThreatLevel(data["threat_assessment"])
        if data.get("compliance_requirements"):
            event.compliance_requirements = [ComplianceStandard(c) for c in data["compliance_requirements"]]
        if data.get("tags"):
            event.tags = set(data["tags"])
        
        return event


@dataclass
class ComplianceReport:
    """Compliance report for regulatory requirements"""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    compliance_standard: ComplianceStandard = ComplianceStandard.CUSTOM
    report_period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc) - timedelta(days=30))
    report_period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Compliance metrics
    total_events: int = 0
    compliant_events: int = 0
    non_compliant_events: int = 0
    compliance_percentage: float = 0.0
    
    # Detailed findings
    violations: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    
    # Supporting evidence
    audit_events: List[str] = field(default_factory=list)  # Event IDs
    policy_coverage: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_compliance_percentage(self):
        """Calculate compliance percentage"""
        if self.total_events > 0:
            self.compliance_percentage = (self.compliant_events / self.total_events) * 100.0
        else:
            self.compliance_percentage = 100.0


class SecurityAuditManager:
    """
    Comprehensive audit manager for PACT-AX security operations.
    Provides observability, compliance reporting, and forensic analysis capabilities.
    """
    
    def __init__(self, 
                 agent_identity: AgentIdentity,
                 audit_level: AuditLevel = AuditLevel.STANDARD,
                 storage_path: Optional[Path] = None):
        
        self.agent_identity = agent_identity
        self.audit_level = audit_level
        self.storage_path = storage_path or Path("./audit_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory buffers for recent events
        self.recent_events: deque = deque(maxlen=10000)
        self.event_correlations: Dict[str, List[str]] = defaultdict(list)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Persistent storage
        self.db_path = self.storage_path / "audit.db"
        self._initialize_database()
        
        # Analytics and patterns
        self.pattern_analyzer = AuditPatternAnalyzer()
        self.compliance_checker = ComplianceChecker()
        
        # Threading for async operations
        self._lock = threading.RLock()
        
        # Metrics tracking
        self.audit_metrics = {
            "total_events": 0,
            "events_by_type": defaultdict(int),
            "events_by_severity": defaultdict(int),
            "compliance_violations": defaultdict(int),
            "forensic_investigations": 0,
            "storage_size_bytes": 0
        }
    
    def record_security_event(self, security_event: SecurityEvent, 
                            additional_context: Optional[Dict[str, Any]] = None) -> AuditEvent:
        """Record a security event in the audit trail with full context"""
        
        with self._lock:
            # Convert security event to audit event
            audit_event = self._create_audit_event_from_security_event(security_event, additional_context)
            
            # Determine if event should be audited based on audit level
            if not self._should_audit_event(audit_event):
                return audit_event
            
            # Enrich event with additional context
            self._enrich_audit_event(audit_event)
            
            # Store event
            self._store_audit_event(audit_event)
            
            # Update metrics
            self._update_audit_metrics(audit_event)
            
            # Check for compliance violations
            self._check_compliance_violations(audit_event)
            
            # Update correlations and patterns
            self._update_event_correlations(audit_event)
            self.pattern_analyzer.analyze_event(audit_event)
            
            return audit_event
    
    def record_context_operation(self, 
                                operation_type: str,
                                context_packet: ContextPacket,
                                security_decision: Dict[str, Any],
                                outcome: str,
                                additional_context: Optional[Dict[str, Any]] = None) -> AuditEvent:
        """Record a complete context operation with all security details"""
        
        with self._lock:
            # Create comprehensive audit event
            audit_event = AuditEvent(
                event_type=f"context_operation_{operation_type}",
                severity=self._determine_event_severity(operation_type, outcome),
                agent_from=context_packet.from_agent.agent_id,
                agent_to=context_packet.to_agent,
                context_type=context_packet.context_type,
                context_packet_id=context_packet.metadata.packet_id,
                security_decision=security_decision,
                policy_applied=security_decision.get("policies_applied", []),
                encryption_level=EncryptionLevel(context_packet.metadata.encryption_level) if context_packet.metadata.encryption_level != "none" else None,
                trust_level=security_decision.get("trust_level"),
                threat_assessment=security_decision.get("threat_assessment"),
                outcome=outcome,
                payload_hash=self._hash_payload(context_packet.payload),
                lineage_trace=context_packet.metadata.lineage.copy(),
                system_context=additional_context or {}
            )
            
            # Set compliance requirements based on context
            audit_event.compliance_requirements = self._determine_compliance_requirements(context_packet, security_decision)
            
            # Add tags for searchability
            audit_event.tags.update([
                f"operation_{operation_type}",
                f"context_{context_packet.context_type.value}",
                f"encryption_{context_packet.metadata.encryption_level}",
                f"outcome_{outcome}"
            ])
            
            return self.record_security_event(self._audit_event_to_security_event(audit_event), additional_context)
    
    def generate_compliance_report(self, 
                                 compliance_standard: ComplianceStandard,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> ComplianceReport:
        """Generate comprehensive compliance report for specified standard"""
        
        if not start_date:
            start_date = datetime.now(timezone.utc) - timedelta(days=30)
        if not end_date:
            end_date = datetime.now(timezone.utc)
        
        # Query relevant events from database
        events = self._query_events_for_compliance(compliance_standard, start_date, end_date)
        
        # Generate report
        report = ComplianceReport(
            compliance_standard=compliance_standard,
            report_period_start=start_date,
            report_period_end=end_date
        )
        
        # Analyze compliance
        report.total_events = len(events)
        violations = []
        
        for event in events:
            is_compliant = self.compliance_checker.check_event_compliance(event, compliance_standard)
            
            if is_compliant:
                report.compliant_events += 1
            else:
                report.non_compliant_events += 1
                violations.append({
                    "event_id": event.event_id,
                    "violation_type": self.compliance_checker.get_violation_type(event, compliance_standard),
                    "severity": event.severity.value,
                    "timestamp": event.timestamp.isoformat(),
                    "details": event.error_details or "Compliance violation detected"
                })
        
        report.violations = violations
        report.calculate_compliance_percentage()
        
        # Generate recommendations
        report.recommendations = self.compliance_checker.generate_recommendations(compliance_standard, violations)
        
        # Risk assessment
        report.risk_assessment = self._assess_compliance_risk(violations, events)
        
        # Policy coverage analysis
        report.policy_coverage = self._analyze_policy_coverage(events, compliance_standard)
        
        # Store report
        self._store_compliance_report(report)
        
        return report
    
    def start_forensic_investigation(self, 
                                   investigation_name: str,
                                   trigger_event_id: str,
                                   scope_hours: int = 24) -> str:
        """Start forensic investigation around specific event"""
        
        investigation_id = str(uuid.uuid4())
        
        # Find trigger event
        trigger_event = self._find_event_by_id(trigger_event_id)
        if not trigger_event:
            raise ValueError(f"Trigger event {trigger_event_id} not found")
        
        # Define investigation scope
        scope_start = trigger_event.timestamp - timedelta(hours=scope_hours//2)
        scope_end = trigger_event.timestamp + timedelta(hours=scope_hours//2)
        
        # Gather related events
        related_events = self._query_events_for_investigation(trigger_event, scope_start, scope_end)
        
        # Create investigation record
        investigation = {
            "investigation_id": investigation_id,
            "name": investigation_name,
            "trigger_event_id": trigger_event_id,
            "scope_start": scope_start.isoformat(),
            "scope_end": scope_end.isoformat(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
            "related_events": len(related_events),
            "findings": [],
            "timeline": self._create_investigation_timeline(related_events),
            "patterns": self.pattern_analyzer.analyze_investigation_patterns(related_events)
        }
        
        # Store investigation
        self._store_forensic_investigation(investigation)
        self.audit_metrics["forensic_investigations"] += 1
        
        return investigation_id
    
    def get_audit_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights about audit patterns and system behavior"""
        
        with self._lock:
            # Recent activity analysis
            recent_events = list(self.recent_events)[-1000:] if self.recent_events else []
            
            # Pattern analysis
            patterns = self.pattern_analyzer.get_pattern_insights()
            
            # Compliance status
            compliance_status = {}
            for standard in ComplianceStandard:
                if standard != ComplianceStandard.CUSTOM:
                    recent_violations = len([
                        e for e in recent_events 
                        if standard in getattr(e, 'compliance_requirements', [])
                        and getattr(e, 'outcome', '') == 'failure'
                    ])
                    compliance_status[standard.value] = {
                        "recent_violations": recent_violations,
                        "risk_level": "high" if recent_violations > 10 else "medium" if recent_violations > 3 else "low"
                    }
            
            # System health metrics
            total_events = len(recent_events)
            failed_events = len([e for e in recent_events if getattr(e, 'outcome', '') == 'failure'])
            success_rate = ((total_events - failed_events) / total_events * 100) if total_events > 0 else 100.0
            
            return {
                "audit_metrics": dict(self.audit_metrics),
                "recent_activity": {
                    "total_events": total_events,
                    "failed_events": failed_events,
                    "success_rate": success_rate,
                    "events_last_hour": len([
                        e for e in recent_events 
                        if (datetime.now(timezone.utc) - getattr(e, 'timestamp', datetime.min.replace(tzinfo=timezone.utc))).total_seconds() < 3600
                    ])
                },
                "patterns": patterns,
                "compliance_status": compliance_status,
                "top_event_types": dict(list(self.audit_metrics["events_by_type"].items())[:10]),
                "severity_distribution": dict(self.audit_metrics["events_by_severity"]),
                "storage_info": {
                    "database_size_mb": self.audit_metrics["storage_size_bytes"] / (1024 * 1024),
                    "events_in_memory": len(self.recent_events),
                    "active_correlations": len(self.event_correlations)
                }
            }
    
    def export_audit_data(self, 
                         export_format: str = "json",
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         include_sensitive: bool = False) -> str:
        """Export audit data in specified format"""
        
        if not start_date:
            start_date = datetime.now(timezone.utc) - timedelta(days=30)
        if not end_date:
            end_date = datetime.now(timezone.utc)
        
        events = self._query_events_by_date_range(start_date, end_date)
        
        # Convert to export format
        if export_format.lower() == "json":
            export_data = {
                "export_metadata": {
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "agent_id": self.agent_identity.agent_id,
                    "total_events": len(events),
                    "date_range": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat()
                    }
                },
                "events": [self._sanitize_event_for_export(event, include_sensitive) for event in events]
            }
            
            return json.dumps(export_data, indent=2, default=str)
        
        elif export_format.lower() == "csv":
            # CSV format for spreadsheet analysis
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Headers
            headers = [
                "timestamp", "event_id", "event_type", "severity", "agent_from", "agent_to",
                "context_type", "encryption_level", "trust_level", "outcome", "impact_score"
            ]
            writer.writerow(headers)
            
            # Data rows
            for event in events:
                row = [
                    event.timestamp.isoformat(),
                    event.event_id,
                    event.event_type,
                    event.severity.value,
                    event.agent_from or "",
                    event.agent_to or "",
                    event.context_type.value if event.context_type else "",
                    event.encryption_level.value if event.encryption_level else "",
                    event.trust_level or "",
                    event.outcome or "",
                    event.impact_score
                ]
                writer.writerow(row)
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    # Private helper methods
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent audit storage"""
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Create audit events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_data TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    agent_from TEXT,
                    agent_to TEXT,
                    context_type TEXT,
                    outcome TEXT,
                    correlation_id TEXT,
                    session_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indices for fast queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_from ON audit_events(agent_from)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_correlation_id ON audit_events(correlation_id)')
            
            # Create compliance reports table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS compliance_reports (
                    report_id TEXT PRIMARY KEY,
                    compliance_standard TEXT NOT NULL,
                    report_data TEXT NOT NULL,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create forensic investigations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS forensic_investigations (
                    investigation_id TEXT PRIMARY KEY,
                    investigation_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            
        finally:
            conn.close()
    
    def _create_audit_event_from_security_event(self, security_event: SecurityEvent, additional_context: Optional[Dict[str, Any]]) -> AuditEvent:
        """Convert security event to audit event"""
        
        audit_event = AuditEvent(
            event_type=f"security_{security_event.event_type.value}",
            severity=self._map_security_severity_to_audit_severity(security_event.threat_level),
            agent_from=security_event.agent_from,
            agent_to=security_event.agent_to,
            context_type=security_event.context_type,
            trust_level=security_event.trust_level,
            threat_assessment=security_event.threat_level,
            outcome=security_event.outcome,
            impact_score=security_event.impact_score,
            system_context=additional_context or {}
        )
        
        # Add details from security event
        if security_event.details:
            audit_event.security_decision = security_event.details
        
        return audit_event
    
    def _should_audit_event(self, audit_event: AuditEvent) -> bool:
        """Determine if event should be audited based on audit level"""
        
        if self.audit_level == AuditLevel.MINIMAL:
            return audit_event.severity in [AuditEventSeverity.CRITICAL, AuditEventSeverity.EMERGENCY]
        elif self.audit_level == AuditLevel.STANDARD:
            return audit_event.severity.value >= AuditEventSeverity.MEDIUM.value
        elif self.audit_level == AuditLevel.DETAILED:
            return audit_event.severity.value >= AuditEventSeverity.LOW.value
        else:  # FORENSIC or DEBUG
            return True
    
    def _enrich_audit_event(self, audit_event: AuditEvent):
        """Enrich audit event with additional context"""
        
        # Add system context
        audit_event.system_context.update({
            "audit_agent": self.agent_identity.agent_id,
            "audit_level": self.audit_level.value,
            "system_timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Generate correlation ID if not present
        if not audit_event.correlation_id:
            audit_event.correlation_id = audit_event.context_packet_id or str(uuid.uuid4())
        
        # Add to active session if applicable
        if audit_event.agent_from:
            session_key = f"{audit_event.agent_from}_{audit_event.agent_to or 'unknown'}"
            if session_key not in self.active_sessions:
                self.active_sessions[session_key] = {
                    "session_id": str(uuid.uuid4()),
                    "started_at": datetime.now(timezone.utc),
                    "event_count": 0
                }
            
            session = self.active_sessions[session_key]
            session["event_count"] += 1
            session["last_activity"] = datetime.now(timezone.utc)
            audit_event.session_id = session["session_id"]
    
    def _store_audit_event(self, audit_event: AuditEvent):
        """Store audit event in persistent storage"""
        
        # Add to recent events buffer
        self.recent_events.append(audit_event)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO audit_events 
                (event_id, event_data, timestamp, event_type, severity, agent_from, agent_to, 
                 context_type, outcome, correlation_id, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                audit_event.event_id,
                json.dumps(audit_event.to_dict()),
                audit_event.timestamp.isoformat(),
                audit_event.event_type,
                audit_event.severity.value,
                audit_event.agent_from,
                audit_event.agent_to,
                audit_event.context_type.value if audit_event.context_type else None,
                audit_event.outcome,
                audit_event.correlation_id,
                audit_event.session_id
            ))
            conn.commit()
        finally:
            conn.close()
    
    def _update_audit_metrics(self, audit_event: AuditEvent):
        """Update audit metrics"""
        
        self.audit_metrics["total_events"] += 1
        self.audit_metrics["events_by_type"][audit_event.event_type] += 1
        self.audit_metrics["events_by_severity"][audit_event.severity.value] += 1
        
        # Update storage size estimate
        event_size = len(json.dumps(audit_event.to_dict()).encode('utf-8'))
        self.audit_metrics["storage_size_bytes"] += event_size
    
    def _check_compliance_violations(self, audit_event: AuditEvent):
        """Check for compliance violations"""
        
        for standard in audit_event.compliance_requirements:
            is_compliant = self.compliance_checker.check_event_compliance(audit_event, standard)
            if not is_compliant:
                self.audit_metrics["compliance_violations"][standard.value] += 1
    
    def _update_event_correlations(self, audit_event: AuditEvent):
        """Update event correlation tracking"""
        
        if audit_event.correlation_id:
            self.event_correlations[audit_event.correlation_id].append(audit_event.event_id)
        
        # Clean up old correlations periodically
        if len(self.event_correlations) > 10000:
            # Remove oldest correlations
            oldest_correlations = list(self.event_correlations.keys())[:1000]
            for correlation_id in oldest_correlations:
                del self.event_correlations[correlation_id]
    
    def _determine_event_severity(self, operation_type: str, outcome: str) -> AuditEventSeverity:
        """Determine severity based on operation type and outcome"""
        
        if outcome == "failure":
            if operation_type in ["decrypt", "verify"]:
                return AuditEventSeverity.HIGH
            else:
                return AuditEventSeverity.MEDIUM
        elif outcome == "success":
            return AuditEventSeverity.LOW
        else:
            return AuditEventSeverity.MEDIUM
    
    def _hash_payload(self, payload: Dict[str, Any]) -> str:
        """Create hash of payload for audit purposes"""
        
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode('utf-8')).hexdigest()
    
    def _determine_compliance_requirements(self, context_packet: ContextPacket, security_decision: Dict[str, Any]) -> List[ComplianceStandard]:
        """Determine compliance requirements for context"""
        
        requirements = []
        
        # Check payload for compliance indicators
        payload_str = json.dumps(context_packet.payload).lower()
        
        if any(keyword in payload_str for keyword in ["medical", "health", "patient", "hipaa"]):
            requirements.append(ComplianceStandard.HIPAA)
        
        if any(keyword in payload_str for keyword in ["personal", "gdpr", "privacy", "eu"]):
            requirements.append(ComplianceStandard.GDPR)
        
        if any(keyword in payload_str for keyword in ["financial", "sox", "audit", "accounting"]):
            requirements.append(ComplianceStandard.SOX)
        
        if any(keyword in payload_str for keyword in ["payment", "credit", "card", "pci"]):
            requirements.append(ComplianceStandard.PCI_DSS)
        
        return requirements
    
    def _audit_event_to_security_event(self, audit_event: AuditEvent) -> SecurityEvent:
        """Convert audit event back to security event (simplified)"""
        
        return SecurityEvent(
            event_type=SecurityEventType.CONTEXT_SECURED,
            agent_from=audit_event.agent_from,
            agent_to=audit_event.agent_to,
            context_type=audit_event.context_type,
            trust_level=audit_event.trust_level,
            encryption_level=audit_event.encryption_level,
            threat_level=audit_event.threat_assessment or ThreatLevel.LOW,
            details=audit_event.security_decision,
            outcome=audit_event.outcome,
            impact_score=audit_event.impact_score
        )
    
    def _map_security_severity_to_audit_severity(self, threat_level: ThreatLevel) -> AuditEventSeverity:
        """Map security threat level to audit severity"""
        
        mapping = {
            ThreatLevel.MINIMAL: AuditEventSeverity.LOW,
            ThreatLevel.LOW: AuditEventSeverity.LOW,
            ThreatLevel.MODERATE: AuditEventSeverity.MEDIUM,
            ThreatLevel.HIGH: AuditEventSeverity.HIGH,
            ThreatLevel.CRITICAL: AuditEventSeverity.CRITICAL
        }
        
        return mapping.get(threat_level, AuditEventSeverity.MEDIUM)
    
    def _query_events_for_compliance(self, standard: ComplianceStandard, start_date: datetime, end_date: datetime) -> List[AuditEvent]:
        """Query events relevant to compliance standard"""
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT event_data FROM audit_events 
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
            ''', (start_date.isoformat(), end_date.isoformat()))
            
            rows = cursor.fetchall()
            events = []
            
            for row in rows:
                event_data = json.loads(row[0])
                event = AuditEvent.from_dict(event_data)
                
                # Check if event is relevant to compliance standard
                if standard in event.compliance_requirements:
                    events.append(event)
            
            return events
            
        finally:
            conn.close()
    
    def _find_event_by_id(self, event_id: str) -> Optional[AuditEvent]:
        """Find specific event by ID"""
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT event_data FROM audit_events WHERE event_id = ?', (event_id,))
            row = cursor.fetchone()
            
            if row:
                event_data = json.loads(row[0])
                return AuditEvent.from_dict(event_data)
            
            return None
            
        finally:
            conn.close()
    
    def _query_events_for_investigation(self, trigger_event: AuditEvent, start_time: datetime, end_time: datetime) -> List[AuditEvent]:
        """Query events for forensic investigation"""
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT event_data FROM audit_events 
                WHERE timestamp >= ? AND timestamp <= ?
                AND (agent_from = ? OR agent_to = ? OR correlation_id = ?)
                ORDER BY timestamp ASC
            ''', (
                start_time.isoformat(),
                end_time.isoformat(),
                trigger_event.agent_from,
                trigger_event.agent_to,
                trigger_event.correlation_id
            ))
            
            rows = cursor.fetchall()
            events = []
            
            for row in rows:
                event_data = json.loads(row[0])
                events.append(AuditEvent.from_dict(event_data))
            
            return events
            
        finally:
            conn.close()
    
    def _query_events_by_date_range(self, start_date: datetime, end_date: datetime) -> List[AuditEvent]:
        """Query events by date range"""
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT event_data FROM audit_events 
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 10000
            ''', (start_date.isoformat(), end_date.isoformat()))
            
            rows = cursor.fetchall()
            events = []
            
            for row in rows:
                event_data = json.loads(row[0])
                events.append(AuditEvent.from_dict(event_data))
            
            return events
            
        finally:
            conn.close()
    
    def _assess_compliance_risk(self, violations: List[Dict[str, Any]], events: List[AuditEvent]) -> Dict[str, Any]:
        """Assess compliance risk based on violations"""
        
        if not violations:
            return {"risk_level": "low", "score": 0.1, "factors": []}
        
        # Calculate risk factors
        high_severity_violations = len([v for v in violations if v.get("severity", 1) >= 4])
        recent_violations = len([
            v for v in violations 
            if datetime.fromisoformat(v["timestamp"]) > datetime.now(timezone.utc) - timedelta(days=7)
        ])
        
        risk_score = (len(violations) * 0.1) + (high_severity_violations * 0.3) + (recent_violations * 0.2)
        risk_score = min(1.0, risk_score)
        
        risk_level = "low"
        if risk_score > 0.7:
            risk_level = "high"
        elif risk_score > 0.4:
            risk_level = "medium"
        
        return {
            "risk_level": risk_level,
            "score": risk_score,
            "factors": {
                "total_violations": len(violations),
                "high_severity_violations": high_severity_violations,
                "recent_violations": recent_violations,
                "violation_rate": len(violations) / len(events) if events else 0
            }
        }
    
    def _analyze_policy_coverage(self, events: List[AuditEvent], standard: ComplianceStandard) -> Dict[str, Any]:
        """Analyze policy coverage for compliance standard"""
        
        covered_events = len([e for e in events if e.policy_applied])
        total_events = len(events)
        coverage_percentage = (covered_events / total_events * 100) if total_events > 0 else 0
        
        return {
            "coverage_percentage": coverage_percentage,
            "covered_events": covered_events,
            "total_events": total_events,
            "gaps": total_events - covered_events
        }
    
    def _store_compliance_report(self, report: ComplianceReport):
        """Store compliance report in database"""
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO compliance_reports (report_id, compliance_standard, report_data)
                VALUES (?, ?, ?)
            ''', (
                report.report_id,
                report.compliance_standard.value,
                json.dumps(report.__dict__, default=str)
            ))
            conn.commit()
        finally:
            conn.close()
    
    def _create_investigation_timeline(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Create timeline for forensic investigation"""
        
        timeline = []
        for event in sorted(events, key=lambda e: e.timestamp):
            timeline.append({
                "timestamp": event.timestamp.isoformat(),
                "event_id": event.event_id,
                "event_type": event.event_type,
                "severity": event.severity.value,
                "summary": f"{event.event_type}: {event.outcome or 'unknown outcome'}",
                "agents": [event.agent_from, event.agent_to]
            })
        
        return timeline
    
    def _store_forensic_investigation(self, investigation: Dict[str, Any]):
        """Store forensic investigation in database"""
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO forensic_investigations (investigation_id, investigation_data)
                VALUES (?, ?)
            ''', (
                investigation["investigation_id"],
                json.dumps(investigation, default=str)
            ))
            conn.commit()
        finally:
            conn.close()
    
    def _sanitize_event_for_export(self, event: AuditEvent, include_sensitive: bool) -> Dict[str, Any]:
        """Sanitize audit event for export"""
        
        event_dict = event.to_dict()
        
        if not include_sensitive:
            # Remove or mask sensitive fields
            if "payload_hash" in event_dict:
                event_dict["payload_hash"] = "***REDACTED***"
            
            if "security_decision" in event_dict:
                # Keep structure but remove sensitive values
                sanitized_decision = {}
                for key, value in event_dict["security_decision"].items():
                    if "password" in key.lower() or "secret" in key.lower():
                        sanitized_decision[key] = "***REDACTED***"
                    else:
                        sanitized_decision[key] = value
                event_dict["security_decision"] = sanitized_decision
        
        return event_dict


class AuditPatternAnalyzer:
    """Analyzes audit patterns to identify trends, anomalies, and insights"""
    
    def __init__(self):
        self.patterns = {
            "temporal": defaultdict(list),
            "agent_behavior": defaultdict(dict),
            "failure_patterns": [],
            "success_patterns": [],
            "anomalies": []
        }
    
    def analyze_event(self, audit_event: AuditEvent):
        """Analyze single audit event for patterns"""
        
        # Temporal patterns
        hour = audit_event.timestamp.hour
        self.patterns["temporal"][hour].append(audit_event.event_type)
        
        # Agent behavior patterns
        if audit_event.agent_from:
            agent_patterns = self.patterns["agent_behavior"][audit_event.agent_from]
            if audit_event.event_type not in agent_patterns:
                agent_patterns[audit_event.event_type] = {"count": 0, "outcomes": []}
            
            agent_patterns[audit_event.event_type]["count"] += 1
            if audit_event.outcome:
                agent_patterns[audit_event.event_type]["outcomes"].append(audit_event.outcome)
        
        # Failure/Success patterns
        if audit_event.outcome == "failure":
            self.patterns["failure_patterns"].append({
                "event_type": audit_event.event_type,
                "context_type": audit_event.context_type.value if audit_event.context_type else None,
                "trust_level": audit_event.trust_level,
                "timestamp": audit_event.timestamp
            })
        elif audit_event.outcome == "success":
            self.patterns["success_patterns"].append({
                "event_type": audit_event.event_type,
                "context_type": audit_event.context_type.value if audit_event.context_type else None,
                "trust_level": audit_event.trust_level,
                "timestamp": audit_event.timestamp
            })
    
    def analyze_investigation_patterns(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Analyze patterns specific to forensic investigation"""
        
        if not events:
            return {}
        
        # Sequence analysis
        event_sequence = [(e.timestamp, e.event_type, e.outcome) for e in events]
        event_sequence.sort()
        
        # Find unusual patterns
        failure_clusters = self._find_failure_clusters(events)
        agent_interactions = self._analyze_agent_interactions(events)
        
        return {
            "event_sequence": [{"time": t.isoformat(), "type": et, "outcome": o} for t, et, o in event_sequence],
            "failure_clusters": failure_clusters,
            "agent_interactions": agent_interactions,
            "timeline_anomalies": self._detect_timeline_anomalies(events)
        }
    
    def get_pattern_insights(self) -> Dict[str, Any]:
        """Get comprehensive pattern insights"""
        
        return {
            "temporal_patterns": self._analyze_temporal_patterns(),
            "agent_behavior_insights": self._analyze_agent_behavior(),
            "failure_analysis": self._analyze_failure_patterns(),
            "success_analysis": self._analyze_success_patterns(),
            "detected_anomalies": len(self.patterns["anomalies"])
        }
    
    def _find_failure_clusters(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Find clusters of failures in time"""
        
        failures = [e for e in events if e.outcome == "failure"]
        failures.sort(key=lambda e: e.timestamp)
        
        clusters = []
        current_cluster = []
        cluster_threshold = timedelta(minutes=30)
        
        for failure in failures:
            if not current_cluster or (failure.timestamp - current_cluster[-1].timestamp) <= cluster_threshold:
                current_cluster.append(failure)
            else:
                if len(current_cluster) >= 2:
                    clusters.append({
                        "start_time": current_cluster[0].timestamp.isoformat(),
                        "end_time": current_cluster[-1].timestamp.isoformat(),
                        "failure_count": len(current_cluster),
                        "event_types": list(set(f.event_type for f in current_cluster))
                    })
                current_cluster = [failure]
        
        if len(current_cluster) >= 2:
            clusters.append({
                "start_time": current_cluster[0].timestamp.isoformat(),
                "end_time": current_cluster[-1].timestamp.isoformat(),
                "failure_count": len(current_cluster),
                "event_types": list(set(f.event_type for f in current_cluster))
            })
        
        return clusters
    
    def _analyze_agent_interactions(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Analyze interactions between agents"""
        
        interactions = defaultdict(int)
        
        for event in events:
            if event.agent_from and event.agent_to:
                interaction_key = f"{event.agent_from}->{event.agent_to}"
                interactions[interaction_key] += 1
        
        return dict(interactions)
    
    def _detect_timeline_anomalies(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Detect anomalies in event timeline"""
        
        anomalies = []
        events.sort(key=lambda e: e.timestamp)
        
        for i in range(1, len(events)):
            time_gap = (events[i].timestamp - events[i-1].timestamp).total_seconds()
            
            if time_gap > 3600 and events[i-1].timestamp.hour in range(8, 18):
                anomalies.append({
                    "type": "unusual_time_gap",
                    "gap_seconds": time_gap,
                    "before_event": events[i-1].event_id,
                    "after_event": events[i].event_id
                })
        
        return anomalies
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in audit events"""
        
        peak_hours = {}
        for hour, events in self.patterns["temporal"].items():
            peak_hours[hour] = len(events)
        
        if peak_hours:
            busiest_hour = max(peak_hours.items(), key=lambda x: x[1])
            quietest_hour = min(peak_hours.items(), key=lambda x: x[1])
        else:
            busiest_hour = (0, 0)
            quietest_hour = (0, 0)
        
        return {
            "busiest_hour": {"hour": busiest_hour[0], "event_count": busiest_hour[1]},
            "quietest_hour": {"hour": quietest_hour[0], "event_count": quietest_hour[1]},
            "hourly_distribution": dict(peak_hours)
        }
    
    def _analyze_agent_behavior(self) -> Dict[str, Any]:
        """Analyze agent behavior patterns"""
        
        agent_insights = {}
        
        for agent_id, patterns in self.patterns["agent_behavior"].items():
            total_events = sum(p["count"] for p in patterns.values())
            failure_rate = 0
            
            for event_type, data in patterns.items():
                failures = data["outcomes"].count("failure")
                total_outcomes = len(data["outcomes"])
                if total_outcomes > 0:
                    failure_rate += (failures / total_outcomes) / len(patterns)
            
            agent_insights[agent_id] = {
                "total_events": total_events,
                "event_types": len(patterns),
                "failure_rate": failure_rate,
                "most_common_event": max(patterns.items(), key=lambda x: x[1]["count"])[0] if patterns else None
            }
        
        return agent_insights
    
    def _analyze_failure_patterns(self) -> Dict[str, Any]:
        """Analyze failure patterns"""
        
        if not self.patterns["failure_patterns"]:
            return {"total_failures": 0}
        
        failure_by_type = defaultdict(int)
        failure_by_context = defaultdict(int)
        
        for failure in self.patterns["failure_patterns"]:
            failure_by_type[failure["event_type"]] += 1
            if failure["context_type"]:
                failure_by_context[failure["context_type"]] += 1
        
        return {
            "total_failures": len(self.patterns["failure_patterns"]),
            "failures_by_type": dict(failure_by_type),
            "failures_by_context": dict(failure_by_context),
            "most_common_failure_type": max(failure_by_type.items(), key=lambda x: x[1])[0] if failure_by_type else None
        }
    
    def _analyze_success_patterns(self) -> Dict[str, Any]:
        """Analyze success patterns"""
        
        if not self.patterns["success_patterns"]:
            return {"total_successes": 0}
        
        success_by_trust_level = defaultdict(int)
        
        for success in self.patterns["success_patterns"]:
            if success["trust_level"]:
                trust_bucket = int(success["trust_level"] * 10) / 10
                success_by_trust_level[trust_bucket] += 1
        
        return {
            "total_successes": len(self.patterns["success_patterns"]),
            "success_by_trust_level": dict(success_by_trust_level),
            "optimal_trust_range": self._find_optimal_trust_range()
        }
    
    def _find_optimal_trust_range(self) -> Dict[str, float]:
        """Find optimal trust range based on success patterns"""
        
        trust_levels = [s["trust_level"] for s in self.patterns["success_patterns"] if s["trust_level"]]
        
        if trust_levels:
            return {
                "min": min(trust_levels),
                "max": max(trust_levels),
                "avg": sum(trust_levels) / len(trust_levels)
            }
        
        return {"min": 0.0, "max": 1.0, "avg": 0.5}


class ComplianceChecker:
    """Checks audit events against compliance requirements"""
    
    def __init__(self):
        self.compliance_rules = {
            ComplianceStandard.HIPAA: self._check_hipaa_compliance,
            ComplianceStandard.GDPR: self._check_gdpr_compliance,
            ComplianceStandard.SOX: self._check_sox_compliance,
            ComplianceStandard.PCI_DSS: self._check_pci_compliance
        }
    
    def check_event_compliance(self, event: AuditEvent, standard: ComplianceStandard) -> bool:
        """Check if event meets compliance requirements"""
        
        if standard in self.compliance_rules:
            return self.compliance_rules[standard](event)
        
        return True
    
    def get_violation_type(self, event: AuditEvent, standard: ComplianceStandard) -> str:
        """Get type of compliance violation"""
        
        if not self.check_event_compliance(event, standard):
            if standard == ComplianceStandard.HIPAA:
                return "PHI_PROTECTION_VIOLATION"
            elif standard == ComplianceStandard.GDPR:
                return "PERSONAL_DATA_VIOLATION"
            elif standard == ComplianceStandard.SOX:
                return "FINANCIAL_DATA_VIOLATION"
            elif standard == ComplianceStandard.PCI_DSS:
                return "PAYMENT_DATA_VIOLATION"
        
        return "UNKNOWN_VIOLATION"
    
    def generate_recommendations(self, standard: ComplianceStandard, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on compliance violations"""
        
        recommendations = []
        
        if standard == ComplianceStandard.HIPAA:
            recommendations.extend([
                "Ensure all PHI is encrypted with at least AES-256",
                "Implement comprehensive audit logging for all PHI access",
                "Review and strengthen access controls for medical data",
                "Establish regular security risk assessments"
            ])
        elif standard == ComplianceStandard.GDPR:
            recommendations.extend([
                "Implement data minimization principles",
                "Ensure explicit consent for personal data processing",
                "Review and update data retention policies",
                "Establish data subject rights procedures"
            ])
        elif standard == ComplianceStandard.SOX:
            recommendations.extend([
                "Strengthen financial data access controls",
                "Implement proper segregation of duties",
                "Review audit trail completeness",
                "Establish change management procedures"
            ])
        elif standard == ComplianceStandard.PCI_DSS:
            recommendations.extend([
                "Encrypt all payment card data",
                "Implement strong access control measures",
                "Regularly test security systems",
                "Maintain secure network infrastructure"
            ])
        
        return recommendations
    
    def _check_hipaa_compliance(self, event: AuditEvent) -> bool:
        """Check HIPAA compliance for event"""
        
        if event.encryption_level and event.encryption_level.value in ["none", "obfuscated"]:
            security_decision_str = json.dumps(event.security_decision).lower()
            if any(indicator in security_decision_str for indicator in ["medical", "health", "patient"]):
                return False
        
        if event.outcome == "failure" and not event.error_details:
            return False
        
        return True
    
    def _check_gdpr_compliance(self, event: AuditEvent) -> bool:
        """Check GDPR compliance for event"""
        
        if event.trust_level and event.trust_level < 0.6:
            security_decision_str = json.dumps(event.security_decision).lower()
            if any(indicator in security_decision_str for indicator in ["personal", "email", "address"]):
                if not event.encryption_level or event.encryption_level.value == "none":
                    return False
        
        return True
    
    def _check_sox_compliance(self, event: AuditEvent) -> bool:
        """Check SOX compliance for event"""
        
        if event.context_type in [ContextType.TASK_KNOWLEDGE, ContextType.SYSTEM_STATE]:
            security_decision_str = json.dumps(event.security_decision).lower()
            if any(indicator in security_decision_str for indicator in ["financial", "audit", "revenue"]):
                if not event.encryption_level or event.encryption_level.value in ["none", "obfuscated"]:
                    return False
                if not event.lineage_trace:
                    return False
        
        return True
    
    def _check_pci_compliance(self, event: AuditEvent) -> bool:
        """Check PCI DSS compliance for event"""
        
        security_decision_str = json.dumps(event.security_decision).lower()
        if any(indicator in security_decision_str for indicator in ["payment", "credit", "card"]):
            if not event.encryption_level or event.encryption_level.value in ["none", "obfuscated", "symmetric"]:
                return False
        
        return True


# Example usage and testing
if __name__ == "__main__":
    # Create audit manager
    agent = AgentIdentity(
        agent_id="audit-test-agent",
        agent_type="audit_agent", 
        version="1.0.0",
        capabilities=["audit_management", "compliance_reporting"]
    )
    
    audit_manager = SecurityAuditManager(agent, AuditLevel.DETAILED)
    
    # Create test context packet
    from ..context_share.schemas import ContextMetadata, Priority
    
    test_packet = ContextPacket(
        from_agent=agent,
        to_agent="target-agent",
        context_type=ContextType.EMOTIONAL_STATE,
        payload={"mood": "stressed", "medical_info": "patient_data"},
        metadata=ContextMetadata(),
        trust_required=TrustLevel.STRONG,
        priority=Priority.HIGH
    )
    
    # Test audit recording
    security_decision = {
        "encryption_level": "symmetric",
        "trust_level": 0.8,
        "policies_applied": ["hipaa_policy"],
        "threat_assessment": "low"
    }
    
    audit_event = audit_manager.record_context_operation(
        operation_type="encrypt",
        context_packet=test_packet,
        security_decision=security_decision,
        outcome="success"
    )
    
    print(f"Recorded audit event: {audit_event.event_id}")
    
    # Test compliance report
    compliance_report = audit_manager.generate_compliance_report(
        ComplianceStandard.HIPAA,
        datetime.now(timezone.utc) - timedelta(days=1),
        datetime.now(timezone.utc)
    )
    
    print(f"HIPAA compliance report: {compliance_report.compliance_percentage:.1f}% compliant")
    print(f"Violations: {len(compliance_report.violations)}")
    
    # Test forensic investigation
    investigation_id = audit_manager.start_forensic_investigation(
        "Test Investigation",
        audit_event.event_id,
        scope_hours=2
    )
    
    print(f"Started forensic investigation: {investigation_id}")
    
    # Test audit insights
    insights = audit_manager.get_audit_insights()
    print(f"Total events audited: {insights['audit_metrics']['total_events']}")
    print(f"Success rate: {insights['recent_activity']['success_rate']:.1f}%")
    
    # Test export
    export_data = audit_manager.export_audit_data("json", include_sensitive=False)
    print(f"Export data length: {len(export_data)} characters")
    
    print("Audit system testing completed successfully!")
