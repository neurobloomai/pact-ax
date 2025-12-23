"""
PACT-AX Epistemic State Transfer
Transfer knowledge WITH confidence, uncertainty, and boundaries
"""

from typing import Dict, Any, Optional
from datetime import datetime
from ..primitives.epistemic import EpistemicState, KnowledgeBoundary, BeliefUpdate


class EpistemicStateTransfer:
    """
    Transfers epistemic state between agents.
    Preserves humility through the transfer.
    """
    
    def __init__(self):
        self.transfer_log: List[Dict] = []
    
    def transfer(
        self,
        state: EpistemicState,
        from_agent_id: str,
        to_agent_id: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Transfer epistemic state with full fidelity.
        Receiving agent gets knowledge AND uncertainty.
        """
        transfer_package = {
            'knowledge': state.value,
            'confidence': state.confidence.name,
            'confidence_value': state.confidence.value,
            'uncertainty_reason': state.uncertainty_reason,
            'boundaries': self._serialize_boundary(state.boundary),
            'source': state.source,
            'metadata': {
                'from_agent': from_agent_id,
                'to_agent': to_agent_id,
                'transfer_time': datetime.now().isoformat(),
                'original_timestamp': state.timestamp.isoformat(),
                'state_id': state.id,
                'context': context or {}
            }
        }
        
        # Log transfer
        self.transfer_log.append(transfer_package)
        
        return transfer_package
    
    def receive(
        self,
        transfer_package: Dict[str, Any],
        receiving_agent_id: str
    ) -> EpistemicState:
        """
        Receiving agent reconstructs epistemic state.
        Can adjust confidence based on trust in source.
        """
        from ..primitives.epistemic import ConfidenceLevel
        
        # Reconstruct state
        confidence_level = ConfidenceLevel[transfer_package['confidence']]
        
        received_state = EpistemicState(
            value=transfer_package['knowledge'],
            confidence=confidence_level,
            uncertainty_reason=transfer_package['uncertainty_reason'],
            source=f"transferred_from_{transfer_package['metadata']['from_agent']}",
            timestamp=datetime.fromisoformat(transfer_package['metadata']['original_timestamp'])
        )
        
        return received_state
    
    def _serialize_boundary(self, boundary: Optional[KnowledgeBoundary]) -> Optional[Dict]:
        """Convert boundary to transferable format"""
        if not boundary:
            return None
        
        return {
            'domain': boundary.domain,
            'proficiency': boundary.proficiency.name,
            'known_limits': list(boundary.known_limits),
            'known_capabilities': list(boundary.known_capabilities),
            'last_updated': boundary.last_updated.isoformat()
        }
