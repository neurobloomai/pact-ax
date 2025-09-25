// PACT Protocol Engine - Core Implementation
// neurobloom.ai Agent Collaboration Layer

import { EventEmitter } from 'events';
import { Server as SocketIOServer } from 'socket.io';
import { createServer } from 'http';

// ============================================================================
// CORE TYPES & INTERFACES
// ============================================================================

export type StakeholderRole = 'technical' | 'policy' | 'defense' | 'ethics' | 'industry' | 'academia';

export interface Stakeholder {
  id: string;
  role: StakeholderRole;
  organizationId: string;
  displayName: string;
  trustLevel: number; // 0-1
  securityClearance: 'public' | 'internal' | 'confidential' | 'classified';
  connectionStatus: 'online' | 'offline' | 'away';
  lastActivity: Date;
}

export interface PACTMessage {
  id: string;
  fromStakeholder: string;
  toStakeholder?: string; // undefined means broadcast
  messageType: 'contribution' | 'translation' | 'consensus_update' | 'system';
  content: any;
  timestamp: Date;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  securityLevel: string;
}

export interface DecisionContext {
  id: string;
  title: string;
  description: string;
  requiredStakeholders: StakeholderRole[];
  deadline?: Date;
  securityLevel: string;
  status: 'initializing' | 'gathering_input' | 'building_consensus' | 'finalizing' | 'completed';
}

export interface StakeholderContribution {
  stakeholderId: string;
  role: StakeholderRole;
  content: {
    assessment: string;
    keyPoints: string[];
    riskLevel?: number;
    confidence: number;
    recommendation: string;
  };
  timestamp: Date;
  status: 'draft' | 'submitted' | 'acknowledged';
}

export interface ConsensusState {
  decisionId: string;
  overallAlignment: number; // 0-1
  stakeholderPositions: Map<string, StakeholderContribution>;
  convergenceAreas: string[];
  conflictPoints: string[];
  synthesizedView?: string;
  nextSteps: string[];
}

// ============================================================================
// PACT PROTOCOL ENGINE - MAIN CLASS
// ============================================================================

export class PACTProtocolEngine extends EventEmitter {
  private stakeholders: Map<string, Stakeholder> = new Map();
  private decisions: Map<string, DecisionContext> = new Map();
  private consensusStates: Map<string, ConsensusState> = new Map();
  private messageHistory: Map<string, PACTMessage[]> = new Map();
  public io: SocketIOServer;

  constructor(server?: any) {
    super();
    
    // Initialize Socket.IO server
    const httpServer = server || createServer();
    this.io = new SocketIOServer(httpServer, {
      cors: {
        origin: "*", // Configure appropriately for production
        methods: ["GET", "POST"]
      }
    });

    this.setupSocketHandlers();
    
    console.log('🚀 PACT Protocol Engine initialized');
  }

  // ============================================================================
  // STAKEHOLDER MANAGEMENT
  // ============================================================================

  registerStakeholder(stakeholderData: Omit<Stakeholder, 'connectionStatus' | 'lastActivity'>): string {
    const stakeholder: Stakeholder = {
      ...stakeholderData,
      connectionStatus: 'offline',
      lastActivity: new Date()
    };

    this.stakeholders.set(stakeholder.id, stakeholder);
    
    console.log(`✅ Stakeholder registered: ${stakeholder.displayName} (${stakeholder.role})`);
    
    // Emit stakeholder joined event
    this.emit('stakeholder:registered', stakeholder);
    
    return stakeholder.id;
  }

  connectStakeholder(stakeholderId: string, socketId: string): boolean {
    const stakeholder = this.stakeholders.get(stakeholderId);
    if (!stakeholder) return false;

    stakeholder.connectionStatus = 'online';
    stakeholder.lastActivity = new Date();
    
    // Associate socket with stakeholder
    this.io.to(socketId).emit('connection:established', {
      stakeholderId,
      role: stakeholder.role,
      availableDecisions: Array.from(this.decisions.values())
    });

    // Notify other stakeholders
    this.broadcastToDecisionParticipants(
      Array.from(this.decisions.keys())[0], // For demo, use first decision
      {
        type: 'stakeholder_connected',
        stakeholder: stakeholder.displayName,
        role: stakeholder.role
      }
    );

    console.log(`🔗 Stakeholder connected: ${stakeholder.displayName}`);
    
    this.emit('stakeholder:connected', stakeholder);
    return true;
  }

  // ============================================================================
  // DECISION CONTEXT MANAGEMENT
  // ============================================================================

  createDecision(decisionData: Omit<DecisionContext, 'status'>): string {
    const decision: DecisionContext = {
      ...decisionData,
      status: 'initializing'
    };

    this.decisions.set(decision.id, decision);
    
    // Initialize consensus state
    this.consensusStates.set(decision.id, {
      decisionId: decision.id,
      overallAlignment: 0,
      stakeholderPositions: new Map(),
      convergenceAreas: [],
      conflictPoints: [],
      nextSteps: ['Waiting for stakeholder contributions']
    });

    // Initialize message history
    this.messageHistory.set(decision.id, []);

    console.log(`📋 Decision created: ${decision.title}`);
    
    // Notify relevant stakeholders
    this.notifyRelevantStakeholders(decision);
    
    this.emit('decision:created', decision);
    return decision.id;
  }

  updateDecisionStatus(decisionId: string, status: DecisionContext['status']): boolean {
    const decision = this.decisions.get(decisionId);
    if (!decision) return false;

    decision.status = status;
    
    // Broadcast status update
    this.broadcastToDecisionParticipants(decisionId, {
      type: 'status_update',
      decisionId,
      newStatus: status
    });

    console.log(`📊 Decision ${decisionId} status updated to: ${status}`);
    
    this.emit('decision:status_updated', { decisionId, status });
    return true;
  }

  // ============================================================================
  // CONTRIBUTION PROCESSING
  // ============================================================================

  async submitContribution(
    decisionId: string, 
    stakeholderId: string, 
    contribution: Omit<StakeholderContribution, 'stakeholderId' | 'timestamp' | 'status'>
  ): Promise<boolean> {
    const decision = this.decisions.get(decisionId);
    const stakeholder = this.stakeholders.get(stakeholderId);
    
    if (!decision || !stakeholder) return false;

    const fullContribution: StakeholderContribution = {
      stakeholderId,
      ...contribution,
      timestamp: new Date(),
      status: 'submitted'
    };

    // Update consensus state
    const consensusState = this.consensusStates.get(decisionId)!;
    consensusState.stakeholderPositions.set(stakeholderId, fullContribution);

    // Process contribution through PACT protocols
    await this.processContribution(decisionId, fullContribution);

    console.log(`📝 Contribution submitted by ${stakeholder.displayName} for decision ${decisionId}`);
    
    this.emit('contribution:submitted', { decisionId, contribution: fullContribution });
    return true;
  }

  private async processContribution(decisionId: string, contribution: StakeholderContribution): Promise<void> {
    // Update consensus metrics
    await this.updateConsensusMetrics(decisionId);
    
    // Generate translations for other stakeholders
    await this.generateTranslations(decisionId, contribution);
    
    // Check if we can build consensus
    await this.attemptConsensusBuilding(decisionId);
    
    // Broadcast updates
    this.broadcastConsensusUpdate(decisionId);
  }

  // ============================================================================
  // TRANSLATION ENGINE
  // ============================================================================

  private async generateTranslations(decisionId: string, contribution: StakeholderContribution): Promise<void> {
    const decision = this.decisions.get(decisionId)!;
    const sourceStakeholder = this.stakeholders.get(contribution.stakeholderId)!;
    
    // For each other required stakeholder role, generate translation
    for (const targetRole of decision.requiredStakeholders) {
      if (targetRole === sourceStakeholder.role) continue;
      
      const translation = await this.translateContribution(
        contribution, 
        sourceStakeholder.role, 
        targetRole
      );
      
      // Send translation to stakeholders of target role
      this.sendTranslationToRole(decisionId, targetRole, translation);
    }
  }

  private async translateContribution(
    contribution: StakeholderContribution,
    fromRole: StakeholderRole,
    toRole: StakeholderRole
  ): Promise<any> {
    // Simulate AI-powered translation (replace with actual AI service)
    const translations = this.getTranslationTemplates();
    
    const translationKey = `${fromRole}_to_${toRole}`;
    const template = translations[translationKey] || translations.default;
    
    return {
      originalRole: fromRole,
      targetRole: toRole,
      originalContent: contribution.content.assessment,
      translatedContent: template.translate(contribution.content),
      keyPointsTranslated: contribution.content.keyPoints.map(point => 
        template.translatePoint(point)
      ),
      contextualNotes: template.getContextualNotes(contribution.content)
    };
  }

  private getTranslationTemplates() {
    return {
      'technical_to_policy': {
        translate: (content: any) => `Policy implications: ${content.assessment.replace(/technical/gi, 'regulatory')}`,
        translatePoint: (point: string) => `Regulatory consideration: ${point}`,
        getContextualNotes: (content: any) => [`Risk level ${content.riskLevel} requires policy review`]
      },
      'technical_to_defense': {
        translate: (content: any) => `Security assessment: ${content.assessment.replace(/model/gi, 'system')}`,
        translatePoint: (point: string) => `Threat consideration: ${point}`,
        getContextualNotes: (content: any) => [`Confidence level ${content.confidence} for operational planning`]
      },
      'policy_to_technical': {
        translate: (content: any) => `Technical requirements: ${content.assessment}`,
        translatePoint: (point: string) => `Implementation requirement: ${point}`,
        getContextualNotes: (content: any) => ['Technical feasibility assessment needed']
      },
      default: {
        translate: (content: any) => `Cross-domain perspective: ${content.assessment}`,
        translatePoint: (point: string) => `Shared consideration: ${point}`,
        getContextualNotes: (content: any) => ['Multi-stakeholder coordination required']
      }
    };
  }

  // ============================================================================
  // CONSENSUS BUILDING
  // ============================================================================

  private async updateConsensusMetrics(decisionId: string): Promise<void> {
    const consensusState = this.consensusStates.get(decisionId)!;
    const contributions = Array.from(consensusState.stakeholderPositions.values());
    
    if (contributions.length < 2) {
      consensusState.overallAlignment = 0;
      return;
    }

    // Calculate alignment score (simplified algorithm)
    let alignmentScore = 0;
    const totalPairs = contributions.length * (contributions.length - 1) / 2;
    
    for (let i = 0; i < contributions.length; i++) {
      for (let j = i + 1; j < contributions.length; j++) {
        alignmentScore += this.calculatePairwiseAlignment(contributions[i], contributions[j]);
      }
    }
    
    consensusState.overallAlignment = alignmentScore / totalPairs;
    
    // Update convergence and conflict areas
    consensusState.convergenceAreas = this.identifyConvergenceAreas(contributions);
    consensusState.conflictPoints = this.identifyConflictPoints(contributions);
  }

  private calculatePairwiseAlignment(contrib1: StakeholderContribution, contrib2: StakeholderContribution): number {
    // Simplified alignment calculation
    // In production, this would use more sophisticated NLP/AI
    
    const rec1 = contrib1.content.recommendation.toLowerCase();
    const rec2 = contrib2.content.recommendation.toLowerCase();
    
    // Check for similar recommendations
    if (rec1.includes('deploy') && rec2.includes('deploy')) return 0.8;
    if (rec1.includes('caution') && rec2.includes('caution')) return 0.8;
    if (rec1.includes('restrict') && rec2.includes('restrict')) return 0.8;
    if (rec1.includes('test') && rec2.includes('test')) return 0.7;
    
    return 0.3; // Default low alignment
  }

  private identifyConvergenceAreas(contributions: StakeholderContribution[]): string[] {
    const commonThemes: string[] = [];
    
    // Look for common keywords in assessments
    const allAssessments = contributions.map(c => c.content.assessment.toLowerCase());
    const keywords = ['safety', 'risk', 'testing', 'monitoring', 'gradual', 'caution'];
    
    keywords.forEach(keyword => {
      const count = allAssessments.filter(assessment => assessment.includes(keyword)).length;
      if (count >= Math.ceil(contributions.length * 0.6)) {
        commonThemes.push(`Shared concern about ${keyword}`);
      }
    });
    
    return commonThemes;
  }

  private identifyConflictPoints(contributions: StakeholderContribution[]): string[] {
    const conflicts: string[] = [];
    
    const deployCount = contributions.filter(c => 
      c.content.recommendation.toLowerCase().includes('deploy')
    ).length;
    
    const restrictCount = contributions.filter(c => 
      c.content.recommendation.toLowerCase().includes('restrict')
    ).length;
    
    if (deployCount > 0 && restrictCount > 0) {
      conflicts.push('Disagreement on deployment timeline');
    }
    
    return conflicts;
  }

  private async attemptConsensusBuilding(decisionId: string): Promise<void> {
    const consensusState = this.consensusStates.get(decisionId)!;
    const decision = this.decisions.get(decisionId)!;
    
    if (consensusState.overallAlignment > 0.7 && 
        consensusState.stakeholderPositions.size >= decision.requiredStakeholders.length) {
      
      // Generate synthesized view
      consensusState.synthesizedView = await this.generateSynthesizedView(consensusState);
      consensusState.nextSteps = ['Review synthesized recommendation', 'Prepare implementation plan'];
      
      // Update decision status
      this.updateDecisionStatus(decisionId, 'finalizing');
    }
  }

  private async generateSynthesizedView(consensusState: ConsensusState): Promise<string> {
    const contributions = Array.from(consensusState.stakeholderPositions.values());
    
    // Simplified synthesis (would use AI in production)
    const commonRecommendations = contributions
      .map(c => c.content.recommendation)
      .join(', ');
      
    return `Based on multi-stakeholder analysis: ${consensusState.convergenceAreas.join(', ')}. ` +
           `Recommended approach balances stakeholder concerns: ${commonRecommendations}. ` +
           `Confidence level: ${Math.round(consensusState.overallAlignment * 100)}%`;
  }

  // ============================================================================
  // COMMUNICATION & BROADCASTING
  // ============================================================================

  private broadcastToDecisionParticipants(decisionId: string, message: any): void {
    const decision = this.decisions.get(decisionId);
    if (!decision) return;

    // Find all connected stakeholders for this decision
    const relevantStakeholders = Array.from(this.stakeholders.values())
      .filter(s => decision.requiredStakeholders.includes(s.role) && s.connectionStatus === 'online');

    // Broadcast via Socket.IO
    relevantStakeholders.forEach(stakeholder => {
      this.io.emit(`decision:${decisionId}:update`, {
        ...message,
        timestamp: new Date(),
        forStakeholder: stakeholder.id
      });
    });
  }

  private sendTranslationToRole(decisionId: string, targetRole: StakeholderRole, translation: any): void {
    const targetStakeholders = Array.from(this.stakeholders.values())
      .filter(s => s.role === targetRole && s.connectionStatus === 'online');

    targetStakeholders.forEach(stakeholder => {
      this.io.emit(`translation:${stakeholder.id}`, {
        decisionId,
        translation,
        timestamp: new Date()
      });
    });
  }

  private broadcastConsensusUpdate(decisionId: string): void {
    const consensusState = this.consensusStates.get(decisionId);
    if (!consensusState) return;

    this.broadcastToDecisionParticipants(decisionId, {
      type: 'consensus_update',
      consensusState: {
        alignment: Math.round(consensusState.overallAlignment * 100),
        convergenceAreas: consensusState.convergenceAreas,
        conflictPoints: consensusState.conflictPoints,
        synthesizedView: consensusState.synthesizedView,
        nextSteps: consensusState.nextSteps
      }
    });
  }

  private notifyRelevantStakeholders(decision: DecisionContext): void {
    decision.requiredStakeholders.forEach(role => {
      const stakeholders = Array.from(this.stakeholders.values())
        .filter(s => s.role === role);

      stakeholders.forEach(stakeholder => {
        if (stakeholder.connectionStatus === 'online') {
          this.io.emit(`decision:new:${stakeholder.id}`, {
            decision,
            yourRole: role,
            timestamp: new Date()
          });
        }
      });
    });
  }

  // ============================================================================
  // SOCKET.IO HANDLERS
  // ============================================================================

  private setupSocketHandlers(): void {
    this.io.on('connection', (socket) => {
      console.log(`🔌 New socket connection: ${socket.id}`);

      socket.on('stakeholder:connect', (data) => {
        this.connectStakeholder(data.stakeholderId, socket.id);
      });

      socket.on('contribution:submit', async (data) => {
        await this.submitContribution(
          data.decisionId,
          data.stakeholderId,
          data.contribution
        );
      });

      socket.on('decision:join', (data) => {
        socket.join(`decision:${data.decisionId}`);
        
        // Send current state
        const consensusState = this.consensusStates.get(data.decisionId);
        if (consensusState) {
          socket.emit('decision:current_state', consensusState);
        }
      });

      socket.on('disconnect', () => {
        console.log(`🔌 Socket disconnected: ${socket.id}`);
        // Update stakeholder status to offline
        // (would need to track socket-to-stakeholder mapping)
      });
    });
  }

  // ============================================================================
  // PUBLIC API METHODS
  // ============================================================================

  getDecisionState(decisionId: string): ConsensusState | null {
    return this.consensusStates.get(decisionId) || null;
  }

  getAllStakeholders(): Stakeholder[] {
    return Array.from(this.stakeholders.values());
  }

  getAllDecisions(): DecisionContext[] {
    return Array.from(this.decisions.values());
  }

  getStakeholdersByRole(role: StakeholderRole): Stakeholder[] {
    return Array.from(this.stakeholders.values()).filter(s => s.role === role);
  }

  // Start the engine
  start(port: number = 3001): void {
    this.io.listen(port);
    console.log(`🚀 PACT Protocol Engine listening on port ${port}`);
  }
}

// ============================================================================
// DEMO SETUP HELPER
// ============================================================================

export function createDemoScenario(): { engine: PACTProtocolEngine, decisionId: string } {
  const engine = new PACTProtocolEngine();
  
  // Register demo stakeholders
  const stakeholders = [
    {
      id: 'tech-001',
      role: 'technical' as StakeholderRole,
      organizationId: 'openai-safety',
      displayName: 'Dr. Sarah Chen (Technical Lead)',
      trustLevel: 0.9,
      securityClearance: 'internal' as const
    },
    {
      id: 'policy-001', 
      role: 'policy' as StakeholderRole,
      organizationId: 'nist-ai',
      displayName: 'Michael Rodriguez (Policy Advisor)',
      trustLevel: 0.85,
      securityClearance: 'confidential' as const
    },
    {
      id: 'defense-001',
      role: 'defense' as StakeholderRole, 
      organizationId: 'dod-ai',
      displayName: 'Colonel Jennifer Smith (Defense Analyst)',
      trustLevel: 0.95,
      securityClearance: 'classified' as const
    },
    {
      id: 'ethics-001',
      role: 'ethics' as StakeholderRole,
      organizationId: 'ai-ethics-board',
      displayName: 'Prof. David Kim (Ethics Board)',
      trustLevel: 0.8,
      securityClearance: 'public' as const
    }
  ];

  stakeholders.forEach(s => engine.registerStakeholder(s));

  // Create demo decision
  const decisionId = engine.createDecision({
    id: 'gpt5-deployment-2024',
    title: 'GPT-5 Deployment Decision',
    description: 'Should GPT-5 be deployed with current safety measures or require additional testing?',
    requiredStakeholders: ['technical', 'policy', 'defense', 'ethics'],
    deadline: new Date(Date.now() + 24 * 60 * 60 * 1000), // 24 hours
    securityLevel: 'confidential'
  });

  console.log('🎭 Demo scenario created with GPT-5 deployment decision');
  
  return { engine, decisionId };
}
