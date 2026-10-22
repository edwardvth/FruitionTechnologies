"""Scan reachable code/config history for credential-shaped strings, without printing values."""
import subprocess, re, json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
def git(*args): return subprocess.check_output(['git',*args],cwd=root)
commits=git('rev-list','HEAD').decode().splitlines(); seen=set(); hits=[]
patterns={'github_token':r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})', 'aws_access_key':r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b','private_key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', 'literal_secret':r'''(?i)(?:api_key|password|secret|token)\s*[:=]\s*["'](?!YOUR_|your_|<)[A-Za-z0-9_+/=-]{16,}["']'''}
for commit in commits:
 for entry in git('ls-tree','-r',commit).decode().splitlines():
  meta,path=entry.split('\t',1); oid=meta.split()[2]
  if oid in seen or not path.endswith(('.py','.yaml','.yml','.txt','.md','.sh','.service','.desktop','.json')) or '/data/' in path: continue
  seen.add(oid); text=git('cat-file','blob',oid).decode(errors='replace')
  for kind,pat in patterns.items():
   for match in re.finditer(pat,text): hits.append({'commit':commit,'path':path,'line':text[:match.start()].count('\n')+1,'pattern':kind})
out={'commits':len(commits),'unique_code_config_blobs':len(seen),'redacted_hits':hits,'limitations':'Regex only; excludes data/media/checkpoint blobs; not a guarantee of absence.'}
(root/'reports/audit-2026-10-22/security-evidence.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
