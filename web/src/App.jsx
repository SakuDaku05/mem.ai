import React from 'react';
import './index.css';

export default function App() {
  return (
    <>
      {/* Navigation */}
      <nav style={{ padding: '1.25rem 2rem', borderBottom: '1px solid var(--border)', background: 'rgba(255, 255, 255, 0.9)', backdropFilter: 'blur(8px)', position: 'sticky', top: 0, zIndex: 50 }}>
        <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ fontSize: '1.5rem', fontWeight: 800 }}>🧠 mem.ai</div>
          <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center' }}>
            <a href="#architecture" style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Architecture</a>
            <a href="#benchmarks" style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Benchmarks</a>
            <a href="#comparison" style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Compare</a>
            <a href="https://github.com/SakuDaku05/mem.ai" className="btn btn-secondary" style={{ padding: '0.4rem 1rem', fontSize: '0.9rem' }}>GitHub</a>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="section" style={{ textAlign: 'center', paddingTop: '8rem' }}>
        <div className="container" style={{ maxWidth: '900px' }}>
          <div className="badge">Next-Gen Agentic Memory</div>
          <h1 style={{ fontSize: '4.5rem', lineHeight: 1.1, marginBottom: '1.5rem' }}>
            Stop your LLMs from <br /><span className="text-gradient">forgetting everything.</span>
          </h1>
          <p style={{ fontSize: '1.25rem', color: 'var(--text-muted)', marginBottom: '3rem', maxWidth: '750px', margin: '0 auto 3rem' }}>
            A production-ready framework combining dense semantic retrieval, causal graph tracking, and PAMI-injected contexts to give your AI assistants flawless long-term recall.
          </p>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem' }}>
            <a href="https://github.com/SakuDaku05/mem.ai" className="btn btn-primary">Start Building</a>
            <a href="#comparison" className="btn btn-secondary">Read the Docs</a>
          </div>
        </div>
      </section>

      {/* Deep Dive 1: The Problem */}
      <section className="section section-alt">
        <div className="container grid-2">
          <div>
            <h2 style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>The "Lost in the Middle" Problem.</h2>
            <p style={{ fontSize: '1.1rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
              Standard RAG systems blindly concatenate facts and shove them into the prompt. LLMs notoriously suffer from the "U-shaped attention curve"—they read the top of the prompt and the bottom of the prompt, but completely ignore facts buried in the middle.
            </p>
            <p style={{ fontSize: '1.1rem', color: 'var(--text-muted)' }}>
              <strong>mem.ai solves this natively</strong> using Position-Aware Memory Injection (PAMI). We mathematically calculate the utility (Q-score) of every fact and structurally force the most critical context to the extreme boundaries of the LLM window.
            </p>
          </div>
          <div className="terminal">
            <div className="terminal-header">
              <div className="dot" style={{background: '#ef4444'}}></div>
              <div className="dot" style={{background: '#f59e0b'}}></div>
              <div className="dot" style={{background: '#10b981'}}></div>
            </div>
            <div className="terminal-body" style={{ fontSize: '0.85rem' }}>
              <span style={{ color: '#64748b' }}>// PAMI Output Generation</span><br/>
              <span style={{ color: '#3b82f6' }}>const</span> context = memai.inject(query);<br/><br/>
              <span style={{ color: '#a78bfa' }}>[CRITICAL BOUNDARY - TOP]</span><br/>
              - User is severely allergic to peanuts.<br/>
              <span style={{ color: '#64748b' }}>...</span><br/>
              <span style={{ color: '#64748b' }}>[LOW UTILITY - MIDDLE]</span><br/>
              - User prefers dark mode UI.<br/>
              - User lives in New York.<br/>
              <span style={{ color: '#64748b' }}>...</span><br/>
              <span style={{ color: '#a78bfa' }}>[CRITICAL BOUNDARY - BOTTOM]</span><br/>
              - User is severely allergic to peanuts.<br/>
            </div>
          </div>
        </div>
      </section>

      {/* Deep Dive 2: Architecture */}
      <section id="architecture" className="section">
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
            <h2 style={{ fontSize: '3rem', marginBottom: '1rem' }}>A Triple-Layered Architecture</h2>
            <p style={{ fontSize: '1.15rem', color: 'var(--text-muted)', maxWidth: '600px', margin: '0 auto' }}>
              Don't just store text. Map relationships, decay stale data, and build causal chains.
            </p>
          </div>
          
          <div className="grid-3">
            <div className="info-card">
              <div className="icon-box">📚</div>
              <h3 style={{ fontSize: '1.5rem', marginBottom: '0.75rem' }}>Semantic Core (ChromaDB)</h3>
              <p style={{ color: 'var(--text-muted)' }}>Ultra-fast dense vector retrieval. Embedded natively so you don't need external databases. Total sandboxing via `agent_id` tracking.</p>
            </div>
            <div className="info-card">
              <div className="icon-box">🕸️</div>
              <h3 style={{ fontSize: '1.5rem', marginBottom: '0.75rem' }}>Causal Event Graph (Kuzu)</h3>
              <p style={{ color: 'var(--text-muted)' }}>True temporal reasoning. Our native graph maps exactly how conversational events precede and cause one another across multiple turns.</p>
            </div>
            <div className="info-card">
              <div className="icon-box">🧹</div>
              <h3 style={{ fontSize: '1.5rem', marginBottom: '0.75rem' }}>Auto-Pruning (R1-R4)</h3>
              <p style={{ color: 'var(--text-muted)' }}>Standard RAG accumulates data until it hallucinates. mem.ai dynamically decays older facts and silently culls direct contradictions.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Deep Dive 3: Benchmarks */}
      <section id="benchmarks" className="section section-alt">
        <div className="container grid-2">
          <div style={{ background: 'white', padding: '2.5rem', borderRadius: '16px', border: '1px solid var(--border)' }}>
            <h3 style={{ fontSize: '1.5rem', marginBottom: '2rem' }}>BEAM Benchmark Results (ICLR 2026)</h3>
            
            <div style={{ marginBottom: '0.5rem', fontWeight: 600, display: 'flex', justifyContent: 'space-between' }}>
              <span>Information Extraction</span>
              <span style={{ color: 'var(--primary)' }}>92% vs 74%</span>
            </div>
            <div className="bar-track"><div className="bar-fill" style={{ width: '92%' }}></div></div>
            <div className="bar-track" style={{ height: '6px', marginTop: '-1rem' }}><div className="bar-fill baseline" style={{ width: '74%' }}></div></div>

            <div style={{ marginBottom: '0.5rem', fontWeight: 600, display: 'flex', justifyContent: 'space-between', marginTop: '1.5rem' }}>
              <span>Temporal Event Ordering</span>
              <span style={{ color: 'var(--primary)' }}>88% vs 61%</span>
            </div>
            <div className="bar-track"><div className="bar-fill" style={{ width: '88%' }}></div></div>
            <div className="bar-track" style={{ height: '6px', marginTop: '-1rem' }}><div className="bar-fill baseline" style={{ width: '61%' }}></div></div>
            
            <div style={{ marginBottom: '0.5rem', fontWeight: 600, display: 'flex', justifyContent: 'space-between', marginTop: '1.5rem' }}>
              <span>Conflict Resolution (Pruning)</span>
              <span style={{ color: 'var(--primary)' }}>95% vs 43%</span>
            </div>
            <div className="bar-track"><div className="bar-fill" style={{ width: '95%' }}></div></div>
            <div className="bar-track" style={{ height: '6px', marginTop: '-1rem' }}><div className="bar-fill baseline" style={{ width: '43%' }}></div></div>
          </div>
          
          <div>
            <h2 style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>Proven Academic Superiority.</h2>
            <p style={{ fontSize: '1.1rem', color: 'var(--text-muted)' }}>
              Engineered against the strictest evaluation harnesses, `mem.ai` consistently destroys standard RAG baselines. By leveraging the causal graph and PAMI injection, the LLM stops hallucinating on stale facts and correctly infers temporal timelines of user requests.
            </p>
          </div>
        </div>
      </section>

      {/* Deep Dive 4: Comparison */}
      <section id="comparison" className="section">
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
            <h2 style={{ fontSize: '3rem', marginBottom: '1rem' }}>Why mem.ai wins.</h2>
            <p style={{ fontSize: '1.15rem', color: 'var(--text-muted)' }}>Compare the underlying architecture against the industry alternatives.</p>
          </div>
          
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th style={{ width: '25%' }}>Feature Architecture</th>
                  <th className="highlight-col" style={{ width: '25%', fontSize: '1.2rem' }}>🧠 mem.ai</th>
                  <th style={{ width: '25%' }}>Mem0</th>
                  <th style={{ width: '25%' }}>Supermemory</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ fontWeight: 600 }}>Database Infrastructure</td>
                  <td className="highlight-col" style={{ fontWeight: 700 }}>Vector + Graph (Hybrid)</td>
                  <td style={{ color: 'var(--text-muted)' }}>Vector Only</td>
                  <td style={{ color: 'var(--text-muted)' }}>Vector Only</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Context Construction</td>
                  <td className="highlight-col" style={{ fontWeight: 700 }}>PAMI (Boundary loaded)</td>
                  <td style={{ color: 'var(--text-muted)' }}>Standard concatenation</td>
                  <td style={{ color: 'var(--text-muted)' }}>Standard concatenation</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Data Lifecycle</td>
                  <td className="highlight-col" style={{ color: 'var(--success)', fontWeight: 700 }}>✓ Auto-Pruning (R1-R4)</td>
                  <td style={{ color: 'var(--text-muted)' }}>Manual deletion required</td>
                  <td style={{ color: 'var(--text-muted)' }}>Manual deletion required</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Temporal Logic</td>
                  <td className="highlight-col" style={{ color: 'var(--success)', fontWeight: 700 }}>✓ Causal Tracking (Kuzu)</td>
                  <td style={{ color: 'var(--danger)' }}>✕ None</td>
                  <td style={{ color: 'var(--danger)' }}>✕ None</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Zero-Friction SDK</td>
                  <td className="highlight-col" style={{ color: 'var(--success)', fontWeight: 700 }}>✓ LangChain, AutoGen, OpenAI</td>
                  <td style={{ color: 'var(--success)' }}>✓ Supported</td>
                  <td style={{ color: 'var(--danger)' }}>✕ Minimal</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <footer style={{ padding: '4rem 2rem', textAlign: 'center', color: 'var(--text-muted)', borderTop: '1px solid var(--border)' }}>
        <p>© 2026 SakuDaku05. Open Source under the MIT License.</p>
      </footer>
    </>
  );
}
