import React from 'react';
import './index.css';

function App() {
  return (
    <>
      <div className="bg-glow"></div>
      
      {/* Navigation */}
      <nav style={{ position: 'sticky', top: 0, zIndex: 50, background: 'rgba(10, 10, 12, 0.8)', backdropFilter: 'blur(12px)', borderBottom: '1px solid var(--border-light)' }}>
        <div className="container flex items-center justify-between" style={{ padding: '1rem 2rem' }}>
          <div style={{ fontSize: '1.5rem', fontWeight: 800, fontFamily: 'Outfit', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            🧠 mem<span className="gradient-text">.ai</span>
          </div>
          <div className="flex gap-4 items-center">
            <a href="https://github.com/SakuDaku05/mem.ai" className="btn btn-secondary" style={{ padding: '0.5rem 1rem', fontSize: '0.9rem' }}>GitHub</a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="container mt-24 mb-8">
        <div className="text-center" style={{ maxWidth: '900px', margin: '0 auto' }}>
          <div style={{ display: 'inline-block', padding: '0.25rem 1rem', borderRadius: '99px', background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.2)', color: 'var(--accent-1)', fontSize: '0.85rem', fontWeight: 600, letterSpacing: '0.05em', marginBottom: '2rem' }}>
            Next-Gen Agentic Memory
          </div>
          <h1 style={{ fontSize: 'clamp(3.5rem, 6vw, 5rem)', fontWeight: 800, lineHeight: 1.1, marginBottom: '1.5rem', letterSpacing: '-0.03em' }}>
            Stop your AI from <br />
            <span className="gradient-text">forgetting everything.</span>
          </h1>
          <p style={{ fontSize: '1.25rem', color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: '3rem', maxWidth: '700px', margin: '0 auto 3rem' }}>
            A triple-layered architecture combining dense vector retrieval, causal event graphs, and PAMI injection to give LLMs flawless long-term recall.
          </p>
          <div className="flex justify-center gap-4">
            <a href="https://github.com/SakuDaku05/mem.ai" className="btn btn-primary">Start Building</a>
            <a href="#comparison" className="btn btn-secondary">Compare Solutions</a>
          </div>
        </div>

        {/* Terminal Code Snippet */}
        <div className="terminal mt-16" style={{ maxWidth: '750px', margin: '4rem auto 0' }}>
          <div className="terminal-header">
            <div className="dot r"></div>
            <div className="dot y"></div>
            <div className="dot g"></div>
          </div>
          <div className="terminal-body">
            <span className="comment"># 1. Install the standalone engine</span><br/>
            <span className="cmd">$ pip install memai</span><br/>
            <span className="cmd">$ memai serve</span><br/>
            <span className="output">INFO: Started memai engine on port 8000</span><br/><br/>
            
            <span className="comment"># 2. Drop it into your existing workflow</span><br/>
            <span className="cmd">>>> from memai.connectors.openai import wrap_openai</span><br/>
            <span className="cmd">>>> from openai import OpenAI</span><br/>
            <span className="cmd">>>> client = wrap_openai(OpenAI())</span><br/><br/>
            <span className="comment"># memai intercepts the call, injects memory, and saves the output invisibly.</span><br/>
            <span className="cmd">>>> client.chat.completions.create(..., user="agent_123")</span>
          </div>
        </div>
      </main>

      {/* Feature Architecture */}
      <section className="container mt-24">
        <div className="text-center mb-8">
          <h2 style={{ fontSize: '3rem', fontWeight: 800, marginBottom: '1rem' }}>The Tri-Engine Architecture</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '1.2rem', maxWidth: '600px', margin: '0 auto' }}>Standard vector databases hallucinate. mem.ai dynamically prunes and structures data.</p>
        </div>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem', marginTop: '4rem' }}>
          <div className="glass-card">
            <div style={{ fontSize: '3rem', marginBottom: '1.5rem' }}>📚</div>
            <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: '#fff' }}>Semantic Vector Core</h3>
            <p style={{ color: 'var(--text-muted)', lineHeight: 1.6 }}>Embedded ChromaDB handles millions of facts with dense vector retrieval and strict multi-tenant isolation via agent_id scoping.</p>
          </div>
          <div className="glass-card">
            <div style={{ fontSize: '3rem', marginBottom: '1.5rem' }}>🕸️</div>
            <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: '#fff' }}>Causal Event Graph</h3>
            <p style={{ color: 'var(--text-muted)', lineHeight: 1.6 }}>Kuzu-powered graph DB tracks temporal reasoning. It maps exactly how conversational events precede and cause one another.</p>
          </div>
          <div className="glass-card">
            <div style={{ fontSize: '3rem', marginBottom: '1.5rem' }}>🎯</div>
            <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: '#fff' }}>PAMI Context Injection</h3>
            <p style={{ color: 'var(--text-muted)', lineHeight: 1.6 }}>Position-Aware Memory Injection forces high-utility facts to the boundaries of the LLM window to guarantee structural recall.</p>
          </div>
        </div>
      </section>

      {/* Comparison Table */}
      <section id="comparison" className="container mt-24 mb-24">
        <div className="text-center mb-8">
          <h2 style={{ fontSize: '3rem', fontWeight: 800, marginBottom: '1rem' }}>Why mem.ai wins.</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '1.2rem' }}>How we compare against the industry standard memory wrappers.</p>
        </div>

        <div className="comp-table-wrapper">
          <table className="comp-table">
            <thead>
              <tr>
                <th style={{ width: '25%' }}>Feature</th>
                <th className="highlight-col" style={{ width: '25%', fontSize: '1.3rem' }}>🧠 mem.ai</th>
                <th style={{ width: '25%' }}>Mem0</th>
                <th style={{ width: '25%' }}>Supermemory / Standard RAG</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Data Architecture</td>
                <td className="highlight-col" style={{ color: 'white', fontWeight: 600 }}>Vector + Graph (Hybrid)</td>
                <td style={{ color: 'var(--text-muted)' }}>Vector Only</td>
                <td style={{ color: 'var(--text-muted)' }}>Vector Only</td>
              </tr>
              <tr>
                <td>Context Optimization</td>
                <td className="highlight-col" style={{ color: 'white', fontWeight: 600 }}>PAMI Injection (Boundary loaded)</td>
                <td style={{ color: 'var(--text-muted)' }}>Standard concatenation</td>
                <td style={{ color: 'var(--text-muted)' }}>Standard concatenation</td>
              </tr>
              <tr>
                <td>Auto-Pruning (R1-R4)</td>
                <td className="highlight-col" style={{ color: '#10b981', fontWeight: 600 }}>✓ Built-in Staleness Detection</td>
                <td style={{ color: 'var(--text-muted)' }}>Manual deletion</td>
                <td style={{ color: 'var(--text-muted)' }}>Manual deletion</td>
              </tr>
              <tr>
                <td>Causal Event Tracking</td>
                <td className="highlight-col" style={{ color: '#10b981', fontWeight: 600 }}>✓ Native (Kuzu)</td>
                <td style={{ color: '#ef4444' }}>✕ None</td>
                <td style={{ color: '#ef4444' }}>✕ None</td>
              </tr>
              <tr>
                <td>Academic Benchmarks</td>
                <td className="highlight-col" style={{ color: 'white', fontWeight: 600 }}>BEAM / LoCoMo Proven</td>
                <td style={{ color: 'var(--text-muted)' }}>Internal metrics</td>
                <td style={{ color: 'var(--text-muted)' }}>Unbenchmarked</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <footer className="container mt-24 mb-8" style={{ borderTop: '1px solid var(--border-light)', paddingTop: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        <p>© 2026 SakuDaku05. Open Source under the MIT License.</p>
      </footer>
    </>
  );
}

export default App;
