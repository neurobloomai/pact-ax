// Demo Server for neurobloom.ai PACT Protocol Engine
// src/demo/server.ts

import express from 'express';
import { createServer } from 'http';
import path from 'path';
import { PACTProtocolEngine, createDemoScenario } from '../engine/PACTProtocolEngine';

const app = express();
const server = createServer(app);
const PORT = process.env.PORT || 3001;

// ============================================================================
// DEMO SETUP
// ============================================================================

console.log('🚀 Starting neurobloom.ai PACT Demo Server...');

// Initialize PACT Protocol Engine with demo scenario
const { engine, decisionId } = createDemoScenario();

// Start the PACT engine with our HTTP server
engine.io.attach(server);

// ============================================================================
// STATIC FILE SERVING
// ============================================================================

// Serve demo frontend files
app.use('/demo', express.static(path.join(__dirname, '../frontend')));
app.use('/assets', express.static(path.join(__dirname, '../frontend/assets')));

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// CORS for development
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  next();
});

// ============================================================================
// API ROUTES
// ============================================================================

// Demo status endpoint
app.get('/api/status', (req, res) => {
  res.json({
    status: 'running',
    engine: 'PACT Protocol Engine v1.0',
    demo: 'AI Safety Decision-Making',
    stakeholders: engine.getAllStakeholders().length,
    decisions: engine.getAllDecisions().length,
    uptime: process.uptime(),
    timestamp: new Date().toISOString()
  });
});

// Get all stakeholders
app.get('/api/stakeholders', (req, res) => {
  const stakeholders = engine.getAllStakeholders();
  res.json({
    count: stakeholders.length,
    stakeholders: stakeholders.map(s => ({
      id: s.id,
      role: s.role,
      displayName: s.displayName,
      organizationId: s.organizationId,
      trustLevel: s.trustLevel,
      connectionStatus: s.connectionStatus,
      securityClearance: s.securityClearance
    }))
  });
});

// Get all decisions
app.get('/api/decisions', (req, res) => {
  const decisions = engine.getAllDecisions();
  res.json({
    count: decisions.length,
    decisions: decisions
  });
});

// Get specific decision state
app.get('/api/decisions/:decisionId/state', (req, res) => {
  const decisionState = engine.getDecisionState(req.params.decisionId);
  
  if (!decisionState) {
    return res.status(404).json({ error: 'Decision not found' });
  }
  
  res.json({
    decisionId: req.params.decisionId,
    state: {
      overallAlignment: Math.round(decisionState.overallAlignment * 100),
      stakeholderCount: decisionState.stakeholderPositions.size,
      convergenceAreas: decisionState.convergenceAreas,
      conflictPoints: decisionState.conflictPoints,
      synthesizedView: decisionState.synthesizedView,
      nextSteps: decisionState.nextSteps
    }
  });
});

// Submit contribution (for API testing)
app.post('/api/decisions/:decisionId/contribute', async (req, res) => {
  const { decisionId } = req.params;
  const { stakeholderId, contribution } = req.body;
  
  const success = await engine.submitContribution(decisionId, stakeholderId, contribution);
  
  if (success) {
    res.json({ 
      success: true, 
      message: 'Contribution submitted successfully',
      decisionId,
      stakeholderId 
    });
  } else {
    res.status(400).json({ 
      success: false, 
      error: 'Failed to submit contribution' 
    });
  }
});

// Demo scenario reset
app.post('/api/demo/reset', (req, res) => {
  // Create fresh demo scenario
  const { engine: newEngine, decisionId: newDecisionId } = createDemoScenario();
  
  res.json({
    success: true,
    message: 'Demo scenario reset',
    newDecisionId,
    stakeholders: newEngine.getAllStakeholders().length
  });
});

// ============================================================================
// DEMO ROUTES
// ============================================================================

// Main demo interface
app.get('/', (req, res) => {
  res.redirect('/demo');
});

app.get('/demo', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/index.html'));
});

// Stakeholder-specific demo interfaces
app.get('/demo/technical', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/stakeholder.html'), (err) => {
    if (err) {
      res.send(`
        <html>
          <head><title>Technical Stakeholder - neurobloom.ai Demo</title></head>
          <body>
            <h1>Technical Stakeholder Interface</h1>
            <p>Role: Technical Lead</p>
            <p>Focus: Risk Assessment & Capability Analysis</p>
            <p><a href="/demo">← Back to main demo</a></p>
          </body>
        </html>
      `);
    }
  });
});

app.get('/demo/policy', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/stakeholder.html'), (err) => {
    if (err) {
      res.send(`
        <html>
          <head><title>Policy Stakeholder - neurobloom.ai Demo</title></head>
          <body>
            <h1>Policy Advisor Interface</h1>
            <p>Role: Policy & Regulatory</p>
            <p>Focus: Compliance & Regulatory Impact</p>
            <p><a href="/demo">← Back to main demo</a></p>
          </body>
        </html>
      `);
    }
  });
});

app.get('/demo/defense', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/stakeholder.html'), (err) => {
    if (err) {
      res.send(`
        <html>
          <head><title>Defense Stakeholder - neurobloom.ai Demo</title></head>
          <body>
            <h1>Defense Analyst Interface</h1>
            <p>Role: Defense & Security</p>
            <p>Focus: Threat Modeling & Operational Security</p>
            <p><a href="/demo">← Back to main demo</a></p>
          </body>
        </html>
      `);
    }
  });
});

app.get('/demo/ethics', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/stakeholder.html'), (err) => {
    if (err) {
      res.send(`
        <html>
          <head><title>Ethics Stakeholder - neurobloom.ai Demo</title></head>
          <body>
            <h1>Ethics Board Interface</h1>
            <p>Role: Ethics & Moral Analysis</p>
            <p>Focus: Societal Impact & Moral Implications</p>
            <p><a href="/demo">← Back to main demo</a></p>
          </body>
        </html>
      `);
    }
  });
});

// ============================================================================
// DEMO INFO & MONITORING
// ============================================================================

app.get('/demo/info', (req, res) => {
  const stakeholders = engine.getAllStakeholders();
  const decisions = engine.getAllDecisions();
  const currentDecisionState = engine.getDecisionState(decisionId);
  
  res.json({
    demoInfo: {
      title: 'neurobloom.ai PACT Protocol Demo',
      scenario: 'AI Safety Decision-Making',
      description: 'Multi-stakeholder collaboration for GPT-5 deployment decision'
    },
    stakeholders: stakeholders.map(s => ({
      role: s.role,
      name: s.displayName,
      organization: s.organizationId,
      status: s.connectionStatus,
      trustLevel: Math.round(s.trustLevel * 100) + '%'
    })),
    currentDecision: decisions[0],
    consensusState: currentDecisionState ? {
      alignment: Math.round(currentDecisionState.overallAlignment * 100) + '%',
      contributions: currentDecisionState.stakeholderPositions.size,
      convergenceAreas: currentDecisionState.convergenceAreas,
      conflicts: currentDecisionState.conflictPoints
    } : null,
    endpoints: {
      demo: '/demo',
      api: '/api/status',
      stakeholders: '/api/stakeholders',
      decisions: '/api/decisions'
    }
  });
});

// ============================================================================
// ENGINE EVENT MONITORING
// ============================================================================

// Monitor engine events for demo purposes
engine.on('stakeholder:connected', (stakeholder) => {
  console.log(`✅ Stakeholder connected: ${stakeholder.displayName} (${stakeholder.role})`);
});

engine.on('contribution:submitted', ({ decisionId, contribution }) => {
  const stakeholder = engine.getAllStakeholders().find(s => s.id === contribution.stakeholderId);
  console.log(`📝 New contribution from ${stakeholder?.displayName} for decision ${decisionId}`);
});

engine.on('decision:status_updated', ({ decisionId, status }) => {
  console.log(`📊 Decision ${decisionId} status updated to: ${status}`);
});

// ============================================================================
// ERROR HANDLING
// ============================================================================

// Handle 404s
app.use((req, res) => {
  res.status(404).json({
    error: 'Not found',
    availableEndpoints: [
      '/demo - Main demo interface',
      '/api/status - Server status',
      '/api/stakeholders - List all stakeholders',
      '/api/decisions - List all decisions',
      '/demo/info - Demo information'
    ]
  });
});

// Handle server errors
app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error('Server error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
  });
});

// ============================================================================
// SERVER START
// ============================================================================

server.listen(PORT, () => {
  console.log(`
🎭 ===============================================
   neurobloom.ai PACT Protocol Demo Server
   ===============================================
   
   🌐 Demo Interface:     http://localhost:${PORT}/demo
   📊 Server Status:      http://localhost:${PORT}/api/status
   👥 Stakeholders:       http://localhost:${PORT}/api/stakeholders
   📋 Decisions:          http://localhost:${PORT}/api/decisions
   ℹ️  Demo Info:          http://localhost:${PORT}/demo/info
   
   🔗 Stakeholder Interfaces:
   • Technical:           http://localhost:${PORT}/demo/technical
   • Policy:              http://localhost:${PORT}/demo/policy  
   • Defense:             http://localhost:${PORT}/demo/defense
   • Ethics:              http://localhost:${PORT}/demo/ethics
   
   ===============================================
   🚀 Ready for Krystle's team demo!
   ===============================================
  `);
  
  console.log('📋 Current demo scenario:', {
    decisionId,
    title: 'GPT-5 Deployment Decision',
    stakeholders: engine.getAllStakeholders().length,
    requiredRoles: ['technical', 'policy', 'defense', 'ethics']
  });
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('🛑 Shutting down demo server...');
  server.close(() => {
    console.log('✅ Demo server shut down gracefully');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down demo server...');
  server.close(() => {
    console.log('✅ Demo server shut down gracefully');
    process.exit(0);
  });
});

export { app, server, engine };
