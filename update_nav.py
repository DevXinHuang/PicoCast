import re

with open('docs/index.html', 'r') as f:
    html = f.read()

# Let's find the nav and add our link
nav_pattern = r'(<div class="nav">.*?)(\s*<a href="#main-review">)'
# In docs/index.html around line 609:
# <div class="nav">
#   <a href="#main-review">Main review</a>

match = re.search(nav_pattern, html, flags=re.DOTALL)
if match:
    replacement = match.group(1) + '\n        <a href="research_log.html" style="color:#d29922;">Research log</a>' + match.group(2)
    new_html = html.replace(match.group(0), replacement)
    
    with open('docs/index.html', 'w') as f:
        f.write(new_html)
    print("Nav updated successfully.")
else:
    print("Could not find nav.")
