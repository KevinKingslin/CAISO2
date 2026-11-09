"""Draw the proposal overview as editable SVG and font-embedded vector PDF.

Run with Python and ReportLab; render the PDF with pdftoppm for the PNG preview.
Coordinates use a top-left origin. All artwork is original vector geometry.
"""
from pathlib import Path
from math import atan2, cos, sin, pi
from html import escape
import re
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/pdf'
OUT.mkdir(parents=True, exist_ok=True)
W, H, SCALE = 1200, 790, .45
INK, MUTED = '#233344', '#475769'
BLUE, ORANGE, GREEN = '#DBEDFC', '#FFD9A6', '#D3EBD3'
pdfmetrics.registerFont(TTFont('Arial', '/System/Library/Fonts/Supplemental/Arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', '/System/Library/Fonts/Supplemental/Arial Bold.ttf'))
c = canvas.Canvas(str(OUT / 'caiso_overview.pdf'), pagesize=(W*SCALE, H*SCALE))
c.setTitle('Compound AI System Optimization: Proposed Architecture')
c.setAuthor('CAISO')
c.scale(SCALE, SCALE)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       '<title>Compound AI System Optimization</title>',
       '<desc>System constitution guides causal block discovery, block optimization, and workflow adaptation. Global validation gates both updates; optimization repeats until the budget is exhausted.</desc>']

def rect(x,y,w,h,fill='white',stroke=INK,r=10,lw=1.5,dash=False):
    c.setFillColor(fill); c.setStrokeColor(stroke); c.setLineWidth(lw)
    c.setDash([5,4] if dash else [])
    c.roundRect(x,H-y-h,w,h,r,fill=1,stroke=1)
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{lw}"'+(' stroke-dasharray="5 4"' if dash else '')+'/>')

def line(points,color=INK,lw=1.5,dash=False):
    c.setStrokeColor(color); c.setLineWidth(lw); c.setDash([5,4] if dash else [])
    p=c.beginPath(); p.moveTo(points[0][0],H-points[0][1])
    for x,y in points[1:]: p.lineTo(x,H-y)
    c.drawPath(p)
    svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="{lw}"'+(' stroke-dasharray="5 4"' if dash else '')+'/>')

def arrow(points,color=INK,lw=1.6,dash=False):
    line(points,color,lw,dash)
    (x0,y0),(x,y)=points[-2:]; a=atan2(y-y0,x-x0)
    pts=[(x,y),(x-8*cos(a-pi/6),y-8*sin(a-pi/6)),(x-8*cos(a+pi/6),y-8*sin(a+pi/6))]
    c.setFillColor(color); p=c.beginPath(); p.moveTo(x,H-y)
    for xx,yy in pts[1:]: p.lineTo(xx,H-yy)
    p.close(); c.drawPath(p,fill=1,stroke=0)
    svg.append('<polygon points="'+' '.join(f'{xx},{yy}' for xx,yy in pts)+f'" fill="{color}"/>')

def text(x,y,s,size=18,bold=False,anchor='middle',color=INK):
    font='Arial-Bold' if bold else 'Arial'
    runs=[]
    for part in re.split(r'(Q_sys|S_t|G_t|v_t|w_ij|Q_B|[SGvB]\*)',s):
        if '_' in part:
            base,sub=part.split('_'); runs += [(base,size,0),(sub,size*.7,size*.23)]
        elif re.fullmatch(r'[SGvB]\*',part):
            runs += [(part[0],size,0),('*',size*.7,-size*.28)]
        else: runs.append((part,size,0))
    width=sum(pdfmetrics.stringWidth(part,font,sz) for part,sz,dy in runs)
    xx=x-({'middle':.5,'start':0,'end':1}[anchor])*width
    for part,sz,dy in runs:
        c.setFillColor(color); c.setFont(font,sz); c.drawString(xx,H-y-dy,part)
        svg.append(f'<text x="{xx}" y="{y+dy}" font-family="Arial, Helvetica, sans-serif" font-size="{sz}" font-weight="{700 if bold else 400}" fill="{color}">{escape(part)}</text>')
        xx+=pdfmetrics.stringWidth(part,font,sz)

def lines(x,y,items,size=17,step=23,**kw):
    for i,s in enumerate(items): text(x,y+i*step,s,size,**kw)

def document(x,y,label,accent='#527FA0',w=48,h=59):
    rect(x,y,w,h,'#FFFEF6',INK,2,1.3)
    for k in range(4): line([(x+9,y+13+9*k),(x+w-9,y+13+9*k)],accent,1.6)
    if label: text(x+w/2,y+h+20,label,16)

def node(x,y,label,fill='white'):
    rect(x-18,y-14,36,28,fill,INK,5,1.2)
    text(x,y+5,label,14,True)

def graph(x,y,mode='plain'):
    coords=[(x,y+30),(x+80,y),(x+80,y+60),(x+160,y+30)]
    if mode=='blocks':
        rect(x-28,y-24,135,112,'#F3FAFF','#477DA4',12,1.2,True)
        rect(x+132,y+2,56,56,'#FFF3DA','#A06A2C',10,1.2,True)
    for a,b in [(0,1),(0,2),(1,3),(2,3)]:
        xa,ya=coords[a]; xb,yb=coords[b]
        color='#B64D45' if mode=='intervene' and (a,b)==(1,3) else '#426987'
        arrow([(xa+19,ya),(xb-20,yb)],color,2.5 if mode=='blocks' else 1.4,
              mode=='intervene' and (a,b)==(1,3))
    if mode=='modified': arrow([(x+80,y+15),(x+80,y+45)],'#3D8050',2.2)
    for i,(xx,yy) in enumerate(coords): node(xx,yy,f'C{i+1}')
    if mode=='intervene': text(x+127,y-1,'x',20,True,color='#B64D45')

# Shared constitution: a separate source of guidance above all three stages.
rect(18,10,1164,98,'#F7F8FA','#8A96A2',17)
document(38,27,None,w=40,h=57)
lines(97,42,['Training','preferences'],17,22,anchor='start')
arrow([(205,57),(247,57)])
lines(342,42,['Constitution induction','DICAI / ICAI'],18,24,bold=True)
arrow([(443,57),(483,57)])
document(499,27,None,accent='#568456',w=40,h=57)
text(559,49,'System constitution Q_sys',21,True,anchor='start')
text(559,77,'Guides diagnosis, local principles, and workflow proposals',17,anchor='start')
for xx in (194,600,1006): arrow([(xx,108),(xx,144)],'#67758D',1.5,True)

panels=[(18,BLUE,'1  Causal Block Discovery'),(424,ORANGE,'2  Block Optimization'),(830,GREEN,'3  Workflow Adaptation')]
for x,fill,title in panels:
    rect(x,144,352,548,fill,INK,29,1.7)
    text(x+176,178,title,22,True)
    line([(x+18,194),(x+334,194)],INK,1)

# Panel 1: execution, intervention, decomposition and diagnosis.
text(194,222,'Current system S_t = (G_t, v_t)',19,True)
graph(114,252,'intervene')
text(194,359,'Intervene on each communication edge',17)
text(194,382,'Repeated full-system runs estimate effects',16)
text(194,408,'w_ij = mean[R(original) - R(intervened)]',16,True)
arrow([(194,419),(194,444)])
text(194,466,'Leiden-CPM block discovery',19,True)
graph(114,499,'blocks')
text(194,609,'Q_sys + complete execution trajectories',16)
rect(40,625,308,47,'#FFFFFF',INK,9)
text(194,654,'Select the most deficient block B*',17,True)

# Panel 2: locally attached labels from global outcomes, local principles, gate.
text(600,222,'Compare two block configurations',19,True)
for xx,label in [(450,'Candidate A'),(610,'Candidate B')]:
    rect(xx,240,140,62,'#FFF8EC',INK,8)
    text(xx+70,265,label,17,True)
    text(xx+70,287,'Full-system run',16)
arrow([(520,303),(520,321),(600,321),(600,337)])
arrow([(680,303),(680,321),(600,321)])
lines(600,359,['Rank complete trajectories','Task metric + Q_sys'],18,24)
arrow([(600,393),(600,418)])
document(451,438,None,accent='#A36C32',w=34,h=48)
text(627,449,'Induce block constitution Q_B',18,True)
text(622,474,'From globally labeled preferences',16)
arrow([(600,492),(600,515)])
lines(600,538,['Optimize the selected block jointly','Prompts / weights / models / parameters'],17,24,bold=False)
arrow([(600,579),(600,602)])
rect(446,610,308,62,'#FFFFFF',INK,9)
text(600,635,'Global validation gate',19,True)
text(600,658,'Accept only an end-to-end improvement',16)

# Panel 3: conditional topology search, adaptation, acceptance and refresh.
rect(852,210,308,68,'#F3FAF1',INK,9)
text(1006,236,'Trigger workflow search',19,True)
text(1006,261,'K accepted block updates OR plateau',16)
arrow([(1006,279),(1006,301)])
text(1006,325,'Propose one DAG modification',19,True)
graph(926,352,'modified')
text(1006,454,'Add / remove / reconnect / reorder',16)
text(1006,477,'Use only the fixed component library',16)
arrow([(1006,490),(1006,514)])
lines(1006,537,['Adapt affected blocks briefly','Fixed compatibility-adaptation budget'],17,24)
arrow([(1006,578),(1006,601)])
rect(852,610,308,62,'#FFFFFF',INK,9)
text(1006,635,'Global validation gate',19,True)
text(1006,658,'Accept only an end-to-end improvement',16)

# Stage-to-stage flow; the panels are architectural stages, not graph nodes.
arrow([(374,401),(420,401)],INK,2)
arrow([(780,401),(826,401)],INK,2)

# Explicit iteration and termination; a rejected candidate preserves the state.
arrow([(1006,692),(1006,718),(194,718),(194,695)],INK,1.8)
rect(320,701,575,33,'white','white',0,0)
text(607,724,'Repeat attribution and block discovery on the accepted system',17,True)
text(600,752,'Accepted structure: refresh Q_sys. Rejected candidate: retain current system. Stop at budget exhaustion.',16)
text(600,778,'Output: optimized workflow and component configurations S* = (G*, v*)',19,True)

svg.append('</svg>')
(OUT / 'caiso_overview.svg').write_text('\n'.join(svg))
c.showPage(); c.save()
print(OUT / 'caiso_overview.pdf')
