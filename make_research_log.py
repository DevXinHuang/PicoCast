import re
with open('docs/index.html', 'r') as f:
    html = f.read()

# find </nav>
nav_end = html.find('</nav>') + len('</nav>')
head_and_nav = html[:nav_end]
head_and_nav = head_and_nav.replace('<title>PicoCAST K7UAZ Review</title>', '<title>PicoCAST Research Log</title>')

body = """
    </div>
  </div>

  <div class="page">
    <div class="section-header">
      <div>
        <h2>Research Log</h2>
        <p>A chronological record of research progress, visualizations, and findings.</p>
      </div>
    </div>

    <section class="section">
      <div class="panel lead-panel">
        <div class="panel-body">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <h3>Update: KEMX T004 P1 Candidate Map</h3>
            <span style="color:#637083; font-size:14px; font-weight:bold;">2026-06-19</span>
          </div>
          <p>
            This interactive map shows the radar candidate KEMX_T004_P1 overlaid with expected balloon position and KEMX radar reflectivity data.
          </p>
          <div style="margin: 16px 0;">
            <a class="button" href="maps/20260619_candidate_inspect_map.html" target="_blank">Open Map in New Tab</a>
            <a class="button secondary" href="https://github.com/DevXinHuang/PicoCast/blob/main/notebooks/candidate_inspect.ipynb" target="_blank">Download Jupyter Notebook on GitHub</a>
          </div>
          <div style="border: 1px solid #d7dce3; border-radius: 8px; overflow: hidden; height: 600px;">
            <iframe src="maps/20260619_candidate_inspect_map.html" width="100%" height="100%" frameborder="0"></iframe>
          </div>
        </div>
      </div>
    </section>
  </div>
</body>
</html>
"""

with open('docs/research_log.html', 'w') as f:
    f.write(head_and_nav + body)
