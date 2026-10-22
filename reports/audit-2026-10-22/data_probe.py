#!/usr/bin/env python3
"""Read-only stdlib audit of committed dataset; does not load model checkpoints."""
import collections, hashlib, itertools, json, math, pathlib, subprocess
ROOT=pathlib.Path(__file__).resolve().parents[2]
def git(*args): return subprocess.check_output(['git', '-C', str(ROOT), *args])
entries={}
for entry in git('ls-tree','-rz','HEAD').split(b'\0'):
    if entry:
        meta,path=entry.split(b'\t',1); mode,kind,oid=meta.decode().split()
        entries[path.decode()]=(kind,oid)
def read(path): return git('show','HEAD:'+path)
base='goose_detection/data/'
report={'commit':git('rev-parse','HEAD').decode().strip(),'method':'Committed git blobs; exact shared image blob identities, SHA-256 verified for cross-split groups; no pickle or model loading','coco':{},'yolo':{},'cross_split_duplicates':{}}
for split in ['train','val','test']:
    path=base+'processed/'+split+'_annotations.json'; data=json.loads(read(path)); imgs={i['id']:i for i in data['images']}; anns=collections.defaultdict(list)
    errors=collections.Counter(); whole=0; dup=0
    for a in data['annotations']:
        anns[a['image_id']].append(a)
        if a['image_id'] not in imgs: errors['orphan_annotation']+=1; continue
        i=imgs[a['image_id']]; x,y,w,h=a['bbox']
        if not all(math.isfinite(v) for v in [x,y,w,h]):errors['nonfinite']+=1
        if w<=0 or h<=0:errors['nonpositive_size']+=1
        if x<0 or y<0 or x+w>i['width']+1e-6 or y+h>i['height']+1e-6:errors['outside_image']+=1
        if [x,y,w,h]==[0,0,i['width'],i['height']]: whole+=1
    for aa in anns.values(): dup+=len(aa)-len({(a['category_id'],tuple(a['bbox'])) for a in aa})
    report['coco'][split]={'images':len(imgs),'annotations':len(data['annotations']),'negative_images':sum(not anns[i] for i in imgs),'whole_frame_boxes':whole,'duplicate_boxes':dup,'errors':dict(errors),'images_missing_processed_source':sum(base+'processed/images/'+pathlib.PurePosixPath(i['file_name']).name not in entries for i in imgs.values())}
# Batch read committed labels, avoiding imports of application or model code.
label_paths=[p for p in entries if p.startswith(base+'yolo/labels/') and p.endswith('.txt')]
raw=subprocess.check_output(['git','-C',str(ROOT),'cat-file','--batch'],input=''.join(entries[p][1]+'\n' for p in label_paths).encode())
labels={}; pos=0
for p in label_paths:
    end=raw.index(b'\n',pos); size=int(raw[pos:end].split()[-1]); pos=end+1
    labels[p]=raw[pos:pos+size].decode(); pos+=size+1
hash_groups=collections.defaultdict(list)
for split in ['train','val','test']:
    paths=[p for p in entries if p.startswith(base+'yolo/images/'+split+'/')]
    counts=collections.Counter(images=len(paths)); errors=collections.Counter(); examples={}; groups=collections.defaultdict(list)
    for p in paths:
        hash_groups[entries[p][1]].append(p)
        lp=base+'yolo/labels/'+split+'/'+pathlib.PurePosixPath(p).stem+'.txt'
        if lp not in labels:errors['missing_label']+=1; continue
        lines=labels[lp].splitlines(); counts['positive_images' if lines else 'negative_images']+=1
        if pathlib.PurePosixPath(p).name.startswith('gabe_'): counts['gabe_images']+=1
        counts['boxes']+=len(lines); counts['duplicate_boxes']+=len(lines)-len(set(lines))
        for line in lines:
            try:
                c,x,y,w,h=map(float,line.split())
                if not all(math.isfinite(v) for v in [c,x,y,w,h]):raise ValueError()
                if c!=0:errors['unexpected_class']+=1
                if w<=0 or h<=0:errors['nonpositive_size']+=1
                if min(x,y,w,h)<0 or max(x,y,w,h)>1:errors['outside_normalized_range']+=1
                if x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2>1+1e-6 or y+h/2>1+1e-6:
                    errors['box_edges_outside_image']+=1; examples.setdefault('box_edges_outside_image',{'path':lp,'label':line})
                if [x,y,w,h]==[.5,.5,1.,1.]:counts['whole_frame_boxes']+=1
            except ValueError:errors['invalid_label']+=1
    report['yolo'][split]={**counts,'errors':dict(errors),'examples':examples}
for a,b in itertools.combinations(['train','val','test'],2):
    groups=[]
    for oid,paths in hash_groups.items():
        aa=[p for p in paths if '/images/'+a+'/' in p]; bb=[p for p in paths if '/images/'+b+'/' in p]
        if aa and bb: groups.append({'sha256':hashlib.sha256(read(paths[0])).hexdigest(),'a':aa,'b':bb})
    report['cross_split_duplicates'][a+'_'+b]={'unique_content_groups':len(groups),'a_images':sum(len(g['a']) for g in groups),'b_images':sum(len(g['b']) for g in groups),'groups':groups}
report['raw_coco']={}
for p in entries:
    if p.startswith(base+'raw/') and p.endswith('.json'):
        try:d=json.loads(read(p))
        except (ValueError,subprocess.CalledProcessError):continue
        if 'images' in d:report['raw_coco'][p]={'images':len(d['images']),'annotations':len(d.get('annotations',[])),'categories':d.get('categories',[])}
print(json.dumps(report,indent=2))
