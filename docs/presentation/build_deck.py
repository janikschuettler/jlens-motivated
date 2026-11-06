"""Build docs/presentation/deck.html: one self-contained click-through presentation.

Reads the slide fragments in docs/presentation/deck/project/slides/*.html in the
order given by deck/project/deck.json and wraps them in a small viewer:
  → / space / click / PageDown : next      ← / PageUp : previous
  Home / End : first / last                 n : speaker notes     f : fullscreen
  o : overview grid                         #7 in the URL opens slide 7
Printing (Ctrl+P, landscape, no margins) gives one slide per page (PDF export).
Slides are 1920x1080 and scaled to fit the window.

Usage: python docs/presentation/build_deck.py
"""
import html
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
PROJ = HERE / "deck" / "project"
OUT = HERE / "deck.html"

VIEWER_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:#FFFFFF;overflow:hidden}
#stage{position:fixed;inset:0;display:flex;align-items:center;justify-content:center}
.slide{position:absolute;left:50%;top:50%;width:1920px;height:1080px;overflow:hidden;
  transform-origin:center center;opacity:0;pointer-events:none;transition:opacity .25s}
.slide.on{opacity:1;pointer-events:auto}
.slide>section{width:1920px;height:1080px}
.slide aside{display:none}
#bar{position:fixed;left:0;bottom:0;height:4px;background:#2D58CC;transition:width .25s;z-index:5}
#count{position:fixed;right:14px;bottom:10px;font:13px 'IBM Plex Sans',Arial,sans-serif;color:#8B94A3;z-index:5}
#notes{position:fixed;left:0;right:0;bottom:0;max-height:34vh;overflow:auto;background:rgba(13,17,23,.95);
  color:#EEF1F5;font:18px/1.5 'IBM Plex Sans',Arial,sans-serif;padding:16px 28px 28px;display:none;z-index:6;
  border-top:1px solid #2D58CC}
#notes.on{display:block}
#help{position:fixed;left:14px;bottom:10px;font:13px 'IBM Plex Sans',Arial,sans-serif;color:#566072;z-index:5}
body.grid{overflow:auto}
body.grid #stage{position:static;display:grid;grid-template-columns:repeat(auto-fill,minmax(384px,1fr));gap:16px;padding:16px}
body.grid .slide{position:relative;left:auto;top:auto;opacity:1;pointer-events:auto;cursor:pointer;
  width:384px;height:216px;transform:none!important;outline:2px solid #2D58CC00}
body.grid .slide>section{transform:scale(.2);transform-origin:0 0}
body.grid .slide.on{outline-color:#2D58CC}
body.grid #notes,body.grid #bar{display:none}
@media print{
  @page{size:1920px 1080px;margin:0}
  html,body{overflow:visible;background:#fff}
  #stage{position:static;display:block}
  .slide{position:relative!important;left:auto;top:auto;opacity:1!important;transform:none!important;
    page-break-after:always;break-after:page}
  #bar,#count,#notes,#help{display:none!important}
}
"""

VIEWER_JS = """
(function(){
  const slides=[...document.querySelectorAll('.slide')], n=slides.length;
  const bar=document.getElementById('bar'), count=document.getElementById('count'), notes=document.getElementById('notes');
  let i=0;
  function fit(){
    if(document.body.classList.contains('grid')) return;
    const s=Math.min(innerWidth/1920, innerHeight/1080);
    slides.forEach(el=>el.style.transform='translate(-50%,-50%) scale('+s+')');
  }
  function show(k){
    i=Math.max(0,Math.min(n-1,k));
    slides.forEach((el,j)=>el.classList.toggle('on',j===i));
    bar.style.width=((i+1)/n*100)+'%';
    count.textContent=(i+1)+' / '+n+'  ·  '+slides[i].dataset.title;
    const a=slides[i].querySelector('aside');
    notes.innerHTML='<b>Notes, slide '+(i+1)+'</b><br>'+(a?a.innerHTML:'(none)');
    history.replaceState(null,'','#'+(i+1));
  }
  function grid(on){
    document.body.classList.toggle('grid',on);
    if(on){slides.forEach(el=>el.style.transform='');slides[i].scrollIntoView({block:'center'});}
    else fit();
  }
  addEventListener('keydown',e=>{
    if(e.metaKey||e.ctrlKey||e.altKey) return;
    const k=e.key;
    if(k==='o'||k==='O'){grid(!document.body.classList.contains('grid'));return;}
    if(document.body.classList.contains('grid')){if(k==='Escape'||k==='Enter')grid(false);return;}
    if(k==='ArrowRight'||k===' '||k==='PageDown'||k==='ArrowDown'){e.preventDefault();show(i+1);}
    else if(k==='ArrowLeft'||k==='PageUp'||k==='ArrowUp'||k==='Backspace'){e.preventDefault();show(i-1);}
    else if(k==='Home') show(0);
    else if(k==='End') show(n-1);
    else if(k==='n'||k==='N') notes.classList.toggle('on');
    else if(k==='f'||k==='F'){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();}
  });
  document.getElementById('stage').addEventListener('click',e=>{
    if(document.body.classList.contains('grid')){
      const el=e.target.closest('.slide'); if(el){show(slides.indexOf(el));grid(false);} return;
    }
    if(e.target.closest('a')) return;
    e.clientX<innerWidth/4?show(i-1):show(i+1);
  });
  let tx=null;
  addEventListener('touchstart',e=>{tx=e.touches[0].clientX},{passive:true});
  addEventListener('touchend',e=>{if(tx===null)return;const d=e.changedTouches[0].clientX-tx;
    if(Math.abs(d)>40)show(d<0?i+1:i-1);tx=null;});
  addEventListener('resize',fit);
  fit();
  const h=parseInt(location.hash.slice(1)); show(isNaN(h)?0:h-1);
})();
"""


def main():
    deck = json.loads((PROJ / "deck.json").read_text())
    fonts = "\n".join(f'<link rel="stylesheet" href="{f["href"]}">' for f in deck.get("faces", {}).values())
    parts = []
    for sid in deck["order"]:
        frag = (PROJ / "slides" / f"{sid}.html").read_text()
        m = re.search(r"<h[12][^>]*>(.*?)</h[12]>", frag, re.S)
        title = re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else sid
        parts.append(f'<div class="slide" data-id="{sid}" data-title="{html.escape(title, quote=True)}">\n{frag}\n</div>')
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(deck["title"])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
{fonts}
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>{VIEWER_CSS}</style>
</head>
<body>
<div id="stage">
{chr(10).join(parts)}
</div>
<div id="bar"></div><div id="count"></div>
<div id="help">← → navigate · n notes · o overview · f fullscreen · Ctrl+P for PDF</div>
<div id="notes"></div>
<script>{VIEWER_JS}</script>
</body>
</html>
"""
    OUT.write_text(doc)
    print(f"wrote {OUT} ({len(doc) // 1024} KB, {len(parts)} slides)")


if __name__ == "__main__":
    main()
