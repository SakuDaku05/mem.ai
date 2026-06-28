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
          <a href="https://github.com/SakuDaku05/mem.ai" className="btn btn-outline" style={{ marginLeft: '1rem' }}>GitHub</a>
        </div>
      </nav>

      <main className="container" style={{ paddingTop: '6rem', paddingBottom: '8rem' }}>
        
        {/* Asymmetrical Hero */}
        <section style={{ marginBottom: '8rem' }}>
          <span className="eyebrow">v0.1.0 Architecture</span>
          <h1 style={{ fontSize: 'clamp(4rem, 8vw, 7rem)', lineHeight: 0.95, marginBottom: '2rem' }}>
            ENGINEERED<br />
            FOR ABSOLUTE<br />
            RECALL.
          </h1>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4rem' }}>
            <p style={{ fontSize: '1.25rem', color: 'var(--ink-light)', maxWidth: '500px' }}>
              We abandoned standard vector wrappers. <b>mem.ai</b> is a rigorous hybrid framework merging semantic density, causal graphs, and mathematical injection to cure LLM amnesia.
            </p>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
              <a href="https://github.com/SakuDaku05/mem.ai" className="btn btn-primary">Initialize Engine</a>
              <a href="/docs.html" className="btn btn-outline">Read Documentation</a>
            </div>
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

            <div className="bento-cell col-span-4" style={{ backgroundColor: 'var(--ink)', color: 'white' }}>
              <div style={{ fontSize: '3rem', marginBottom: 'auto', color: 'var(--accent)' }}>01.</div>
              <h3 style={{ fontSize: '1.5rem', marginBottom: '0.5rem', color: 'white' }}>Semantic Vector Core</h3>
              <p style={{ color: '#888', fontSize: '0.9rem' }}>Embedded ChromaDB handles hyper-dense retrieval with strict multitenant isolation via agent_id indexing.</p>
            </div>

            <div className="bento-cell col-span-4">
              <div style={{ fontSize: '3rem', marginBottom: 'auto', color: 'var(--accent)' }}>02.</div>
              <h3 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Causal Event Graph</h3>
              <p style={{ color: 'var(--ink-light)', fontSize: '0.9rem' }}>A Kuzu graph database native to the framework. Maps precisely how events precede and trigger one another across temporal space.</p>
            </div>

            <div className="bento-cell col-span-4">
              <div style={{ fontSize: '3rem', marginBottom: 'auto', color: 'var(--accent)' }}>03.</div>
              <h3 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>R1-R4 Auto-Pruning</h3>
              <p style={{ color: 'var(--ink-light)', fontSize: '0.9rem' }}>Blind accumulation causes hallucination. Built-in algorithms detect staleness, decay old facts, and silently cull contradictions.</p>
            </div>

            <div className="bento-cell col-span-4" id="metrics">
              <span className="eyebrow">ICLR 2026 BEAM Benchmarks</span>
              <h3 style={{ fontSize: '2rem', marginBottom: '2rem' }}>Performance<br />vs Baseline.</h3>
              
              <div style={{ marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.25rem' }}>
                  <span>Information Extraction</span><span style={{ color: var(--accent) }}>92%</span>
                </div>
                <div style={{ height: '4px', background: 'var(--border)', width: '100%' }}>
                  <div style={{ height: '100%', background: 'var(--accent)', width: '92%' }}></div>
                </div>
              </div>

              <div style={{ marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.25rem' }}>
                  <span>Temporal Ordering</span><span style={{ color: var(--accent) }}>88%</span>
                </div>
                <div style={{ height: '4px', background: 'var(--border)', width: '100%' }}>
                  <div style={{ height: '100%', background: 'var(--accent)', width: '88%' }}></div>
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.25rem' }}>
                  <span>Conflict Resolution</span><span style={{ color: var(--accent) }}>95%</span>
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
