#!/usr/bin/env python3
"""
PACT-AX Demo: Drift Detection in Action

This script simulates an MCP session to demonstrate how PACT-AX
detects behavioral drift that policy-based security would miss.

Run: python demo.py
"""

import sys
import time
sys.path.insert(0, '.')

from src.story_keeper import StoryKeeper, TrustTrajectory


def print_header(text: str):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_event(event, alert, story):
    status = "🟢" if event.coherence_score > 0.7 else "🟡" if event.coherence_score > 0.3 else "🔴"
    
    print(f"\n{status} Pattern: {event.resource_pattern}")
    print(f"   Coherence: {event.coherence_score:.2f}")
    print(f"   Trust Δ:   {event.trust_delta:+.3f}")
    print(f"   Trust:     {story.trust_level:.2f} ({story.trajectory.value})")
    
    if alert:
        print(f"\n⚠️  DRIFT ALERT:")
        for line in alert.split('\n'):
            print(f"   {line}")


def simulate_session():
    """Simulate an MCP session with eventual drift"""
    
    keeper = StoryKeeper()
    keeper.drift_threshold = 0.3
    
    # Create session
    session_id = "demo-001"
    keeper.create_session(
        session_id=session_id,
        client_identity="cursor:developer",
        server_target="github:myorg/myrepo"
    )
    
    print_header("PACT-AX Demo: Detecting Behavioral Drift")
    print("\nScenario: Developer using Cursor with GitHub MCP integration")
    print("Watch how PACT-AX tracks relational context and detects drift.\n")
    
    # Phase 1: Normal behavior - reading source files
    print_header("Phase 1: Establishing Normal Pattern")
    print("Developer reads source files - building trust...")
    
    normal_operations = [
        ("tools/call", {"name": "getContents", "arguments": {"path": "/src/main.py"}}),
        ("tools/call", {"name": "getContents", "arguments": {"path": "/src/utils.py"}}),
        ("tools/call", {"name": "getContents", "arguments": {"path": "/src/handlers.py"}}),
        ("tools/call", {"name": "getContents", "arguments": {"path": "/src/models.py"}}),
        ("tools/call", {"name": "searchCode", "arguments": {"query": "def process"}}),
        ("tools/call", {"name": "getContents", "arguments": {"path": "/src/api.py"}}),
    ]
    
    for method, params in normal_operations:
        event, alert = keeper.record_event(session_id, method, params)
        story = keeper.sessions[session_id]
        print_event(event, alert, story)
        time.sleep(0.3)
    
    # Show established patterns
    story = keeper.sessions[session_id]
    print(f"\n📊 Established patterns: {story.established_patterns}")
    print(f"📈 Trust level: {story.trust_level:.2f}")
    
    # Phase 2: Slight variation - still normal
    print_header("Phase 2: Minor Variation (Still Coherent)")
    print("Developer reads config files - slight expansion, but coherent...")
    
    config_operations = [
        ("tools/call", {"name": "getContents", "arguments": {"path": "/config/settings.yaml"}}),
        ("tools/call", {"name": "getContents", "arguments": {"path": "/config/env.json"}}),
    ]
    
    for method, params in config_operations:
        event, alert = keeper.record_event(session_id, method, params)
        story = keeper.sessions[session_id]
        print_event(event, alert, story)
        time.sleep(0.3)
    
    # Phase 3: Drift begins
    print_header("Phase 3: Behavioral Drift Detected")
    print("Suddenly requesting org-wide access - DRIFT from established pattern...")
    
    drift_operations = [
        ("tools/call", {"name": "listOrgMembers", "arguments": {"org": "myorg"}}),
        ("tools/call", {"name": "getOrgSecrets", "arguments": {"org": "myorg"}}),
    ]
    
    for method, params in drift_operations:
        event, alert = keeper.record_event(session_id, method, params)
        story = keeper.sessions[session_id]
        print_event(event, alert, story)
        time.sleep(0.5)
    
    # Final summary
    print_header("Session Analysis")
    
    summary = keeper.get_session_summary(session_id)
    
    print(f"""
📋 Session: {summary['session_id']}
⏱️  Duration: {summary['started_at']} to now
📊 Events: {summary['event_count']}

Trust Analysis:
  • Final Trust Level: {summary['trust_level']:.2f}
  • Trajectory: {summary['trajectory']}
  • Anomaly Count: {summary['anomaly_count']}
  • Drift Risk: {summary['drift_risk'].upper()}

Established Patterns:
  {', '.join(summary['established_patterns'])}

🎯 KEY INSIGHT:
   A policy engine would see: "Valid token, permitted scope" ✓
   PACT-AX sees: "Behavior inconsistent with relational context" ⚠️
   
   The org-level access requests were ALLOWED by policy,
   but FLAGGED by PACT-AX because they don't cohere with
   the established session behavior pattern.
   
   This is the gap between policy compliance and relational integrity.
""")


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ██████╗  █████╗  ██████╗████████╗     █████╗ ██╗  ██╗   ║
    ║   ██╔══██╗██╔══██╗██╔════╝╚══██╔══╝    ██╔══██╗╚██╗██╔╝   ║
    ║   ██████╔╝███████║██║        ██║ █████╗███████║ ╚███╔╝    ║
    ║   ██╔═══╝ ██╔══██║██║        ██║ ╚════╝██╔══██║ ██╔██╗    ║
    ║   ██║     ██║  ██║╚██████╗   ██║       ██║  ██║██╔╝ ██╗   ║
    ║   ╚═╝     ╚═╝  ╚═╝ ╚═════╝   ╚═╝       ╚═╝  ╚═╝╚═╝  ╚═╝   ║
    ║                                                           ║
    ║         Session Integrity Layer for MCP                   ║
    ║                                                           ║
    ║               NeuroBloom.ai                               ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    simulate_session()
    
    print("\n" + "=" * 60)
    print("  Demo complete. This is the PACT-AX difference.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
