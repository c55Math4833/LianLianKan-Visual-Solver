"""
LianLianKan Exporter
Exports template images, game_profile.json for Rust solver engine,
and generates an editable local HTML page (preview.html).
"""

import os
import json
import base64
import cv2
import numpy as np

def save_image_unicode(path, img):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1]
    res, enc = cv2.imencode(ext, img)
    if res:
        with open(path, 'wb') as f:
            f.write(enc)

def to_base64_src(img):
    res, enc = cv2.imencode('.png', img)
    return "data:image/png;base64," + base64.b64encode(enc).decode('utf-8')

def export_all(
    board_roi,
    grid_info,
    empty_cells,
    obstacle_cells,
    clusters,
    all_cells_matrix,
    output_dir="templates",
    profile_path="game_profile.json",
    preview_path="preview.html"
):
    """
    Exports:
    1. templates/empty.png, templates/obstacle.png, templates/tile_01.png ...
    2. game_profile.json matching exact Rust schema
    3. preview.html (editable: obstacle classes / merging classes / per-cell override)
    """
    os.makedirs(output_dir, exist_ok=True)

    tile_mapping = []

    # 1. Export empty template
    if len(empty_cells) > 0:
        empty_sample = cv2.resize(empty_cells[0]["image"], (45, 45))
    else:
        empty_sample = np.full((45, 45, 3), 180, dtype=np.uint8)
    empty_file = os.path.join(output_dir, "empty.png").replace("\\", "/")
    save_image_unicode(empty_file, empty_sample)
    tile_mapping.append({
        "id": 0,
        "type": "empty",
        "file": empty_file
    })

    # 2. Export obstacle template
    if len(obstacle_cells) > 0:
        obstacle_sample = cv2.resize(obstacle_cells[0]["image"], (45, 45))
    else:
        obstacle_sample = np.full((45, 45, 3), 120, dtype=np.uint8)
    obstacle_file = os.path.join(output_dir, "obstacle.png").replace("\\", "/")
    save_image_unicode(obstacle_file, obstacle_sample)
    tile_mapping.append({
        "id": -1,
        "type": "obstacle",
        "file": obstacle_file
    })

    # 3. Export matchable templates
    templates_meta = []
    for c in clusters:
        cid = c["cluster_id"]
        # engine loads templates by file stem: "<int>.png", "empty.png", "obstacle.png"
        filename = f"{cid}.png"
        filepath = os.path.join(output_dir, filename).replace("\\", "/")
        save_image_unicode(filepath, c["template_image"])

        mapping_item = {
            "id": cid,
            "type": "matchable",
            "file": filepath
        }
        tile_mapping.append(mapping_item)
        templates_meta.append({
            "id": cid,
            "file": filepath,
            "color_family": c["color_family"],
            "count": c["count"],
            "coords": c["coords"],
            "image": c["template_image"]
        })

    # 4. Export game_profile.json with exact schema
    # Template dir as seen from the profile file (engine resolves it relative to its cwd,
    # which is normally the folder holding game_profile.json).
    try:
        tpl_dir = os.path.relpath(output_dir, os.path.dirname(os.path.abspath(profile_path)))
    except ValueError:
        tpl_dir = output_dir
    tpl_dir = tpl_dir.replace("\\", "/")

    # Schema consumed by engine/src/profile.rs (name, board, matcher, timing).
    # NOTE: roi must use "width"/"height" only - engine aliases "w"/"h" to them, so
    # writing both makes serde fail with "duplicate field".
    profile = {
        "name": "LianLianKan Studio Profile",
        "version": "1.0",
        "board": {
            "rows": grid_info["rows"],
            "cols": grid_info["cols"],
            "roi": {
                "x": int(board_roi["x"]),
                "y": int(board_roi["y"]),
                "width": int(board_roi["w"]),
                "height": int(board_roi["h"])
            },
            "cell_size": {
                "w": round(float(grid_info["cell_w"]), 2),
                "h": round(float(grid_info["cell_h"]), 2)
            },
            "offset": {
                "x": int(board_roi["x"]),
                "y": int(board_roi["y"])
            }
        },
        "matcher": {
            "threshold": 0.75,
            "empty_variance_threshold": 15.0,
            "template_dir": tpl_dir
        },
        "rules": {
            "allow_outer_path": True
        },
        "timing": {
            "click_interval_ms": 35,
            "animation_delay_ms": 160,
            "verify_timeout_ms": 600
        },
        "tile_mapping": tile_mapping
    }

    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    print(f"[Export] Saved game profile to {profile_path}")
    print(f"[Export] Saved {len(tile_mapping)} templates to {output_dir}/")

    # 5. Generate editable preview.html
    generate_html_report(
        profile=profile,
        empty_sample=empty_sample,
        obstacle_sample=obstacle_sample,
        templates_meta=templates_meta,
        empty_cells=empty_cells,
        obstacle_cells=obstacle_cells,
        all_cells_matrix=all_cells_matrix,
        output_html=preview_path
    )
    return profile

def generate_html_report(
    profile,
    empty_sample,
    obstacle_sample,
    templates_meta,
    empty_cells,
    obstacle_cells,
    all_cells_matrix,
    output_html="preview.html"
):
    """
    Writes an *editable* single-file HTML (no external dependency):
      * click cells -> set to empty / obstacle / any tile class
      * mark a whole class as obstacle, or merge a class into another one
      * live parity check, edits kept in localStorage, import/export of the edit config
      * downloads the edited game_profile.json (replace the one next to the engine)
    """
    rows = profile["board"]["rows"]
    cols = profile["board"]["cols"]

    coord_map = {}
    for ec in empty_cells:
        coord_map[(ec["row"], ec["col"])] = ("empty", 0)
    for oc in obstacle_cells:
        coord_map[(oc["row"], oc["col"])] = ("obstacle", -1)
    for tm in templates_meta:
        for (r, c) in tm["coords"]:
            coord_map[(r, c)] = ("matchable", tm["id"])

    cells = []
    for r in range(rows):
        for c in range(cols):
            item = all_cells_matrix.get((r, c))
            t, i = coord_map.get((r, c), ("empty", 0))
            cells.append({"r": r, "c": c, "type": t, "id": i,
                          "img": to_base64_src(item["image"]) if item else ""})

    data = {
        "rows": rows,
        "cols": cols,
        "profile": profile,
        "cells": cells,
        "templates": [{"id": t["id"], "file": t["file"], "img": to_base64_src(t["image"])}
                      for t in templates_meta],
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    cell_px = 48 if max(cols, rows) <= 20 else 32
    html = _EDITOR_HTML.replace("__DATA__", payload).replace("__CELL__", str(cell_px))
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Export] Saved editable HTML to {output_html}")

_EDITOR_HTML = r"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LianLianKan Studio 編輯器</title>
<style>
:root{--bg:#121417;--card:#1e2228;--bd:#2e3642;--tx:#e6edf3;--tx2:#8b949e;--blue:#58a6ff;--green:#3fb950;--red:#f85149;--yel:#d29922}
body{background:var(--bg);color:var(--tx);font-family:-apple-system,"Segoe UI","Microsoft JhengHei",sans-serif;margin:0;padding:20px}
h1{margin:0 0 4px;font-size:22px}h2{margin:0 0 12px;font-size:17px}
.sub{color:var(--tx2);font-size:13px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0}
.section{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px;margin-bottom:20px}
button,select{background:#2b3139;color:var(--tx);border:1px solid #444c56;border-radius:6px;padding:6px 10px;font-size:13px;cursor:pointer}
button:hover{border-color:var(--blue)}button.primary{background:#1f6feb;border-color:#1f6feb}
.badge{padding:3px 10px;border-radius:12px;font-size:13px;font-weight:600}
.pass{background:rgba(63,185,80,.2);color:#56d364;border:1px solid var(--green)}
.fail{background:rgba(248,81,73,.2);color:var(--red);border:1px solid var(--red)}
.stats{display:flex;gap:18px;flex-wrap:wrap;font-size:14px}
.board{display:inline-grid;gap:3px;background:#0d1117;padding:8px;border-radius:6px;border:1px solid var(--bd)}
.cell{width:__CELL__px;height:__CELL__px;position:relative;border-radius:4px;border:1px solid #30363d;cursor:pointer;display:flex;align-items:center;justify-content:center;background:#161b22;box-sizing:border-box}
.cell img{max-width:90%;max-height:90%;pointer-events:none}
.cell.empty{border:1px dashed #484f58;opacity:.5}.cell.obstacle{border:2px solid #8b949e;background:#21262d}
.cell.matchable{border:1px solid var(--blue)}
.cell.sel{outline:3px solid var(--yel);z-index:1}.cell.hl{outline:3px solid var(--green);z-index:1}
.cell.edited::before{content:"";position:absolute;top:1px;left:1px;width:7px;height:7px;border-radius:50%;background:var(--red)}
.tag{position:absolute;bottom:0;right:1px;font:9px monospace;background:rgba(0,0,0,.75);padding:0 2px;border-radius:2px}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px}
.card{background:#161b22;border:1px solid var(--bd);border-radius:8px;padding:10px;display:flex;gap:12px;align-items:center}
.card.odd{border-color:var(--red)}.card.merged{opacity:.55}
.card img{width:48px;height:48px;border:1px solid var(--bd);border-radius:4px;background:#000;cursor:pointer}
.card .f{display:flex;flex-direction:column;gap:5px;font-size:12px;flex:1}
.card label{color:var(--tx2)}
.hint{font-size:12px;color:var(--tx2);margin-top:8px}
</style>
</head>
<body>
<h1>LianLianKan Studio 編輯器</h1>
<div class="sub">點選棋盤格子（可多選）改類別；在下方類別庫設定「障礙物」或「合併到」其他類別。紅點＝手動修改過的格子。</div>

<div class="bar">
  <span id="parity" class="badge"></span>
  <span class="stats" id="stats"></span>
</div>

<div class="section">
  <h2>棋盤（<span id="dim"></span>）</h2>
  <div class="bar">
    <span>已選 <b id="selN">0</b> 格 →</span>
    <button onclick="setSel('empty')">設為空格</button>
    <button onclick="setSel('obstacle')">設為障礙物</button>
    <select id="selCls"></select>
    <button onclick="setSelCls()">設為此類別</button>
    <button onclick="clearOv()">還原所選格</button>
    <button onclick="sel.clear();render()">取消選取</button>
  </div>
  <div id="board" class="board"></div>
  <div class="hint">點格子切換選取；點下方類別圖示可高亮該類別的所有格子，並一次選取。</div>
</div>

<div class="section">
  <h2>類別庫</h2>
  <div class="gallery" id="gallery"></div>
</div>

<div class="section">
  <h2>輸出</h2>
  <div class="bar">
    <button class="primary" onclick="dlProfile()">下載更新後的 game_profile.json</button>
    <button onclick="dlConfig()">匯出編輯設定 studio_config.json</button>
    <button onclick="document.getElementById('imp').click()">匯入編輯設定</button>
    <input type="file" id="imp" accept=".json" hidden onchange="impConfig(this)">
    <button onclick="resetAll()">全部還原</button>
  </div>
  <div class="bar">
    <label><input type="checkbox" id="outer" onchange="setOuter(this.checked)"> 允許連線繞行棋盤外圍（取消勾選＝不能走外圍，路徑只能在棋盤內）</label>
  </div>
  <div class="hint">編輯內容會自動存在此瀏覽器（localStorage）。下載的 game_profile.json 取代原檔即可；合併後的類別以 "files" 列出所有樣本，障礙物類別 type 為 "obstacle"。</div>
</div>

<script>
const D = __DATA__;
const KEY = "llk-edit:" + D.rows + "x" + D.cols + ":" + D.templates.map(t=>t.id).join(",");
const CELL = __CELL__;
let state = {cellOv:{}, merge:{}, obstacle:{}, allowOuter:null};
function outerOn(){ return state.allowOuter===null || state.allowOuter===undefined ? !(D.profile.rules && D.profile.rules.allow_outer_path===false) : !!state.allowOuter; }
function setOuter(v){ state.allowOuter=v; save(); }
try { const s = localStorage.getItem(KEY); if (s) state = JSON.parse(s); } catch(e) {}
let sel = new Set(), hl = null;
function save(){ try{ localStorage.setItem(KEY, JSON.stringify(state)); }catch(e){} }
function pad(n){ return String(n).padStart(2,"0"); }
function root(id){ let n=0; while (state.merge[id]!=null && n++<50) id = state.merge[id]; return id; }
function eff(cell){
  const ov = state.cellOv[cell.r+","+cell.c];
  let id;
  if (ov){
    if (ov==="empty") return {type:"empty",id:0};
    if (ov==="obstacle") return {type:"obstacle",id:-1};
    id = root(+ov.split(":")[1]);
  } else {
    if (cell.type!=="matchable") return {type:cell.type,id:cell.id};
    id = root(cell.id);
  }
  return state.obstacle[id] ? {type:"obstacle",id:-1} : {type:"matchable",id:id};
}
function counts(){
  const m={}; let e=0,o=0;
  D.cells.forEach(c=>{ const x=eff(c);
    if (x.type==="matchable") m[x.id]=(m[x.id]||0)+1; else if (x.type==="empty") e++; else o++; });
  return {m:m,e:e,o:o};
}
function setSel(t){ sel.forEach(k=>state.cellOv[k]=t); done(); }
function setSelCls(){ const v=document.getElementById("selCls").value; if(!v) return; sel.forEach(k=>state.cellOv[k]="tile:"+v); done(); }
function clearOv(){ sel.forEach(k=>delete state.cellOv[k]); done(); }
function done(){ save(); render(); }
function resetAll(){ if(confirm("清除所有手動修改？")){ state={cellOv:{},merge:{},obstacle:{},allowOuter:null}; sel.clear(); hl=null; done(); } }
function toggle(k){ if(sel.has(k)) sel.delete(k); else sel.add(k); render(); }
function pickClass(id){
  hl = (hl===id) ? null : id; sel.clear();
  if (hl!==null) D.cells.forEach(c=>{ const x=eff(c); if (x.type==="matchable"&&x.id===hl) sel.add(c.r+","+c.c); });
  render();
}
function setRole(id,v){ if (v==="obstacle") state.obstacle[id]=true; else delete state.obstacle[id]; done(); }
function setMerge(id,v){
  if (v===""){ delete state.merge[id]; }
  else {
    const t=+v; let r=t, n=0;
    while (state.merge[r]!=null && n++<50) r=state.merge[r];
    if (r===id){ alert("不可循環合併"); render(); return; }
    state.merge[id]=t;
  }
  done();
}
function render(){
  const cn=counts(), m=cn.m;
  document.getElementById("dim").textContent = D.rows+" 列 × "+D.cols+" 欄";
  const odd = Object.keys(m).filter(k=>m[k]%2);
  const p=document.getElementById("parity");
  p.className="badge "+(odd.length?"fail":"pass");
  p.textContent = odd.length ? "奇偶檢查未通過（類別 "+odd.join(", ")+" 為奇數）" : "奇偶檢查通過";
  const tiles=Object.values(m).reduce((a,b)=>a+b,0);
  document.getElementById("stats").innerHTML = "<span>磚塊："+tiles+"</span><span>類別數："+Object.keys(m).length+"</span><span>空格："+cn.e+"</span><span>障礙物："+cn.o+"</span>";
  document.getElementById("selN").textContent = sel.size;
  document.getElementById("outer").checked = outerOn();
  const b=document.getElementById("board");
  b.style.gridTemplateColumns="repeat("+D.cols+","+CELL+"px)";
  b.innerHTML = D.cells.map(c=>{
    const x=eff(c), k=c.r+","+c.c;
    const tag = x.type==="matchable" ? "T"+pad(x.id) : (x.type==="obstacle" ? "OBS" : "EMP");
    const cls=["cell",x.type,sel.has(k)?"sel":"",(hl!==null&&x.type==="matchable"&&x.id===hl)?"hl":"",state.cellOv[k]?"edited":""].join(" ");
    return '<div class="'+cls+'" title="('+c.r+','+c.c+') '+x.type+' '+x.id+'" onclick="toggle(\''+k+'\')"><img src="'+c.img+'"><span class="tag">'+tag+'</span></div>';
  }).join("");
  const live = D.templates.filter(t=>state.merge[t.id]==null && !state.obstacle[t.id]);
  const sc=document.getElementById("selCls"), old=sc.value;
  sc.innerHTML = '<option value="">選擇類別…</option>'+live.map(t=>'<option value="'+t.id+'">T'+pad(t.id)+'（'+(m[t.id]||0)+'）</option>').join("");
  sc.value=old;
  document.getElementById("gallery").innerHTML = D.templates.map(t=>{
    const merged = state.merge[t.id]!=null, n=m[t.id]||0;
    const targets = D.templates.filter(x=>x.id!==t.id && state.merge[x.id]==null && !state.obstacle[x.id]);
    const absorbed = D.templates.filter(x=>x.id!==t.id && root(x.id)===t.id).map(x=>"T"+x.id);
    const status = merged ? "已合併到 T"+root(t.id)
      : "目前數量：<b>"+n+"</b>"+(n%2?" ⚠奇數":"")+(absorbed.length?"（含 "+absorbed.join(", ")+"）":"");
    return '<div class="card'+((!merged&&n%2)?" odd":"")+(merged?" merged":"")+'">'
     +'<img src="'+t.img+'" onclick="pickClass('+t.id+')" title="點擊高亮並選取此類別所有格子">'
     +'<div class="f"><b>tile_'+pad(t.id)+' (ID '+t.id+')</b><span>'+status+'</span>'
     +'<label>用途 <select onchange="setRole('+t.id+',this.value)"><option value="tile">一般磚塊</option><option value="obstacle"'+(state.obstacle[t.id]?' selected':'')+'>障礙物</option></select></label>'
     +'<label>合併到 <select onchange="setMerge('+t.id+',this.value)"><option value="">（不合併）</option>'
     +targets.map(x=>'<option value="'+x.id+'"'+(state.merge[t.id]===x.id?' selected':'')+'>T'+pad(x.id)+'</option>').join("")
     +'</select></label></div></div>';
  }).join("");
}
function dl(name,obj){
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([JSON.stringify(obj,null,2)],{type:"application/json"}));
  a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),1000);
}
function dlConfig(){ dl("studio_config.json",{version:1,board:{rows:D.rows,cols:D.cols},cellOv:state.cellOv,merge:state.merge,obstacle:state.obstacle,allowOuter:outerOn()}); }
function impConfig(inp){
  const f=inp.files[0]; if(!f) return;
  const r=new FileReader();
  r.onload=()=>{ try{ const j=JSON.parse(r.result);
    state={cellOv:j.cellOv||{},merge:j.merge||{},obstacle:j.obstacle||{},allowOuter:(typeof j.allowOuter==="boolean")?j.allowOuter:null}; done(); }catch(e){ alert("設定檔格式錯誤"); } };
  r.readAsText(f);
}
function dlProfile(){
  const p = JSON.parse(JSON.stringify(D.profile));
  const base = p.tile_mapping.filter(t=>t.type!=="matchable");
  const out = [];
  D.templates.forEach(t=>{
    if (state.merge[t.id]!=null) return;               // absorbed into another class
    const group = D.templates.filter(x=>root(x.id)===t.id);
    const item = {id:t.id, type: state.obstacle[t.id]?"obstacle":"matchable", file:t.file};
    if (group.length>1){ item.files = group.map(x=>x.file); item.merged_from = group.filter(x=>x.id!==t.id).map(x=>x.id); }
    out.push(item);
  });
  p.tile_mapping = base.concat(out);
  p.rules = Object.assign({}, p.rules, {allow_outer_path: outerOn()});
  p.studio_edits = {cellOv:state.cellOv, merge:state.merge, obstacle:state.obstacle};
  dl("game_profile.json", p);
}
render();
</script>
</body>
</html>
"""
