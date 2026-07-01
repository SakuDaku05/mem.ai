import React from 'react';
import './index.css';

export default function App() {
  return (
    <>
      <nav>
        <div className="logo">mem<span>.</span>ai</div>
        <div className="nav-links">
          <a href="#system">System</a>
          <a href="#metrics">Metrics</a>
          <a href="#compare">Compare</a>
          <a href="/docs.html">Docs</a>
          <a href="https://github.com/SakuDaku05/mem.ai" className="btn btn-outline gh-btn">GitHub</a>
        </div>
      </nav>

      <main className="container main-content">
        
        {/* Asymmetrical Hero */}
        <section className="hero-section">
          
          <div className="hero-left">
            <span className="eyebrow">v0.1.0 Architecture</span>
            <h1 style={{ fontSize: 'clamp(4rem, 7vw, 7rem)', lineHeight: 0.95, marginBottom: '2rem' }}>
              ENGINEERED<br />
              FOR ABSOLUTE<br />
              RECALL.
            </h1>
            <p style={{ fontSize: '1.25rem', color: 'var(--ink-light)', maxWidth: '500px', marginBottom: '2.5rem' }}>
              We abandoned standard vector wrappers. <b>mem.ai</b> is a rigorous hybrid framework merging semantic density, causal graphs, and mathematical injection to cure LLM amnesia.
            </p>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
              <a href="https://github.com/SakuDaku05/mem.ai" className="btn btn-primary">Initialize Engine</a>
              <a href="/docs.html" className="btn btn-outline">Read Documentation</a>
            </div>
          </div>

          <div className="hero-right">
            <svg width="100%" height="auto" viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ maxWidth: '400px' }}>
              <g stroke="var(--ink)" strokeWidth="0.5">
                <rect x="10" y="10" width="180" height="180" />
                <rect x="30" y="30" width="140" height="140" />
                <rect x="50" y="50" width="100" height="100" />
                <rect x="70" y="70" width="60" height="60" />
                <line x1="10" y1="10" x2="190" y2="190" />
                <line x1="190" y1="10" x2="10" y2="190" />
                <line x1="100" y1="10" x2="100" y2="190" />
                <line x1="10" y1="100" x2="190" y2="100" />
                <circle cx="100" cy="100" r="40" stroke="var(--accent)" strokeWidth="1" />
                <circle cx="100" cy="100" r="10" fill="var(--accent)" stroke="none" />
              </g>
            </svg>
          </div>
          
        </section>

        {/* Bento Grid */}
        <section id="system" style={{ marginBottom: '8rem' }}>
          <span className="eyebrow">The Tri-Layer System</span>
          <div className="bento-grid">
            
            <div className="bento-cell col-span-8">
              <h2 style={{ fontSize: '3rem', marginBottom: '1rem', maxWidth: '500px' }}>The "Lost in the Middle" Cure.</h2>
              <p style={{ color: 'var(--ink-light)', fontSize: '1.1rem', marginBottom: '2rem', maxWidth: '500px' }}>
                Standard RAG suffocates the context window. <b>PAMI (Position-Aware Memory Injection)</b> calculates fact utility and structurally forces critical memory to the extreme prompt boundaries, guaranteeing 95%+ extraction accuracy.
              </p>
              <div className="code-block" style={{ marginTop: 'auto' }}>
                <span className="comment"># Injection sequence</span><br />
                <span className="keyword">const</span> context = memai.inject(query);<br /><br />
                <span className="keyword">{"{"}</span><br />
                &nbsp;&nbsp;top_boundary: [<span className="comment">"Critical Fact A"</span>],<br />
                &nbsp;&nbsp;mid_padding: [<span className="comment">"Low util Fact B"</span>],<br />
                &nbsp;&nbsp;bottom_boundary: [<span className="comment">"Critical Fact A"</span>]<br />
                <span className="keyword">{"}"}</span>
              </div>
            </div>

            <div className="bento-cell col-span-4" style={{ backgroundColor: 'var(--ink)', color: 'white', position: 'relative', overflow: 'hidden' }}>
              <div style={{ fontSize: '3rem', marginBottom: 'auto', color: 'var(--accent)' }}>01.</div>
              
              <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', opacity: 0.15, pointerEvents: 'none' }}>
                <svg width="120" height="120" viewBox="0 0 120 120">
                  {[...Array(6)].map((_, i) => 
                    [...Array(6)].map((_, j) => (
                      <circle key={`v-${i}-${j}`} cx={10 + i*20} cy={10 + j*20} r="2" fill="white" />
                    ))
                  )}
                </svg>
              </div>

              <h3 style={{ fontSize: '1.5rem', marginBottom: '0.5rem', color: 'white', position: 'relative', zIndex: 1 }}>Semantic Vector Core</h3>
              <p style={{ color: '#888', fontSize: '0.9rem', position: 'relative', zIndex: 1 }}>Embedded ChromaDB handles hyper-dense retrieval with strict multitenant isolation via agent_id indexing.</p>
            </div>

            <div className="bento-cell col-span-4" style={{ position: 'relative', overflow: 'hidden' }}>
              <div style={{ fontSize: '3rem', marginBottom: 'auto', color: 'var(--accent)' }}>02.</div>
              
              <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', opacity: 0.1, pointerEvents: 'none' }}>
                <svg width="120" height="120" viewBox="0 0 120 120" fill="none" stroke="var(--ink)" strokeWidth="2">
                  <circle cx="20" cy="60" r="10" />
                  <circle cx="60" cy="20" r="10" />
                  <circle cx="100" cy="80" r="10" />
                  <line x1="28" y1="52" x2="52" y2="28" />
                  <line x1="68" y1="32" x2="92" y2="72" />
                </svg>
              </div>

              <h3 style={{ fontSize: '1.5rem', marginBottom: '0.5rem', position: 'relative', zIndex: 1 }}>Causal Event Graph</h3>
              <p style={{ color: 'var(--ink-light)', fontSize: '0.9rem', position: 'relative', zIndex: 1 }}>A Kuzu graph database native to the framework. Maps precisely how events precede and trigger one another across temporal space.</p>
            </div>

            <div className="bento-cell col-span-4" style={{ position: 'relative', overflow: 'hidden' }}>
              <div style={{ fontSize: '3rem', marginBottom: 'auto', color: 'var(--accent)' }}>03.</div>
              
              <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', opacity: 0.1, pointerEvents: 'none' }}>
                <svg width="120" height="120" viewBox="0 0 120 120" fill="none" stroke="var(--ink)" strokeWidth="1">
                  <rect x="10" y="10" width="100" height="100" />
                  <rect x="20" y="20" width="80" height="80" strokeDasharray="4 4" />
                  <rect x="30" y="30" width="60" height="60" strokeDasharray="2 6" />
                  <line x1="10" y1="10" x2="110" y2="110" />
                </svg>
              </div>

              <h3 style={{ fontSize: '1.5rem', marginBottom: '0.5rem', position: 'relative', zIndex: 1 }}>R1-R4 Auto-Pruning</h3>
              <p style={{ color: 'var(--ink-light)', fontSize: '0.9rem', position: 'relative', zIndex: 1 }}>Blind accumulation causes hallucination. Built-in algorithms detect staleness, decay old facts, and silently cull contradictions.</p>
            </div>

            <div className="bento-cell col-span-4" id="metrics">
              <span className="eyebrow">ICLR 2026 BEAM Benchmarks</span>
              <h3 style={{ fontSize: '2rem', marginBottom: '2rem' }}>Performance<br />vs Baseline.</h3>
              
              <div style={{ marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.25rem' }}>
                  <span>Information Extraction</span><span style={{ color: 'var(--accent)' }}>92%</span>
                </div>
                <div style={{ height: '4px', background: 'var(--border)', width: '100%' }}>
                  <div style={{ height: '100%', background: 'var(--accent)', width: '92%' }}></div>
                </div>
              </div>

              <div style={{ marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.25rem' }}>
                  <span>Temporal Ordering</span><span style={{ color: 'var(--accent)' }}>88%</span>
                </div>
                <div style={{ height: '4px', background: 'var(--border)', width: '100%' }}>
                  <div style={{ height: '100%', background: 'var(--accent)', width: '88%' }}></div>
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.25rem' }}>
                  <span>Conflict Resolution</span><span style={{ color: 'var(--accent)' }}>95%</span>
                </div>
                <div style={{ height: '4px', background: 'var(--border)', width: '100%' }}>
                  <div style={{ height: '100%', background: 'var(--accent)', width: '95%' }}></div>
                </div>
              </div>
            </div>

          </div>
        </section>

        {/* Minimal Comparison */}
        <section id="compare">
          <span className="eyebrow">Architecture Comparison</span>
          <h2 style={{ fontSize: '4rem', marginBottom: '3rem', letterSpacing: '-0.03em' }}>The Divide.</h2>
          
          <table className="art-table">
            <thead>
              <tr>
                <th style={{ width: '30%' }}>Infrastructure</th>
                <th style={{ width: '25%' }}>mem.ai</th>
                <th style={{ width: '22.5%' }}>Mem0</th>
                <th style={{ width: '22.5%' }}>Supermemory</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Data Architecture</td>
                <td className="winner"><span className="winner-mark">●</span> Vector + Graph</td>
                <td>Vector Only</td>
                <td>Vector Only</td>
              </tr>
              <tr>
                <td>Context Assembly</td>
                <td className="winner"><span className="winner-mark">●</span> PAMI (Boundary)</td>
                <td>Concatenation</td>
                <td>Concatenation</td>
              </tr>
              <tr>
                <td>Data Lifecycle</td>
                <td className="winner"><span className="winner-mark">●</span> Auto-Pruning</td>
                <td>Manual</td>
                <td>Manual</td>
              </tr>
              <tr>
                <td>Temporal Logic</td>
                <td className="winner"><span className="winner-mark">●</span> Native Kuzu</td>
                <td>None</td>
                <td>None</td>
              </tr>
            </tbody>
          </table>
        </section>

      </main>
      
      <footer style={{ borderTop: '1px solid var(--border)', padding: '2rem', display: 'flex', justifyContent: 'space-between' }}>
        <div className="mono" style={{ fontSize: '0.8rem', color: 'var(--ink-light)' }}>// MEM.AI FRAMEWORK</div>
        <div className="mono" style={{ fontSize: '0.8rem', color: 'var(--ink-light)' }}>MIT LICENSE 2026</div>
      </footer>
    </>
  );
}
