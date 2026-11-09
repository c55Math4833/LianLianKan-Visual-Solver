//! Tiny local HTTP server for the editor page (std only, 127.0.0.1 only).
//!
//!   GET  /         the preview page (re-read from disk on every request)
//!   GET  /version  mtime of the preview file; the page polls it and reloads when it changes
//!   GET  /ping     "llk-studio\n<preview path>"  (lets a second studio run find this server)
//!   POST /save     {profile, pngs:[{name,data}]} -> writes game_profile.json and new template PNGs
//!
//! The server exits by itself when the page has not been heard from for a while (tab closed).

use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const BASE_PORT: u16 = 8765;
const PORT_TRIES: u16 = 10;
/// Page polls every 2 s (browsers throttle hidden tabs to ~1/min), so be generous.
const IDLE_AFTER: Duration = Duration::from_secs(180);
/// If the browser never connects at all.
const NO_CLIENT_AFTER: Duration = Duration::from_secs(600);
/// Set when a server run ended because the user pressed Enter (the menu may then clear the screen at once).
pub static STOPPED_BY_ENTER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
const MAX_BODY: usize = 64 << 20;

pub struct Cfg {
    pub preview: PathBuf,
    pub profile: PathBuf,
    pub templates: PathBuf,
}

fn abs_str(p: &Path) -> String {
    std::fs::canonicalize(p)
        .map(|a| a.to_string_lossy().trim_start_matches(r"\\?\").to_string())
        .unwrap_or_else(|_| p.to_string_lossy().to_string())
}

fn open_url(port: u16) {
    let url = format!("http://127.0.0.1:{port}/");
    let _ = std::process::Command::new("cmd").args(["/C", "start", "", &url]).spawn();
}

/// Is there a studio server on `port` serving the same preview file? Returns its preview path.
fn probe(port: u16) -> Option<String> {
    let addr = format!("127.0.0.1:{port}").parse().ok()?;
    let mut s = TcpStream::connect_timeout(&addr, Duration::from_millis(500)).ok()?;
    s.set_read_timeout(Some(Duration::from_millis(1500))).ok()?;
    s.write_all(b"GET /ping HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n").ok()?;
    let mut out = String::new();
    s.read_to_string(&mut out).ok()?;
    let body = out.split("\r\n\r\n").nth(1)?;
    let mut it = body.lines();
    if it.next()? != "llk-studio" {
        return None;
    }
    Some(it.next().unwrap_or("").to_string())
}

/// Serve the editor and open it in the browser. If a studio server for the same preview file is
/// already running, just open it: its page notices the new preview by itself and reloads.
/// Returns true if this call ran a server (and it has stopped now), false if an existing one was reused.
pub fn serve(cfg: &Cfg) -> Result<bool> {
    let me = abs_str(&cfg.preview);
    for port in BASE_PORT..BASE_PORT + PORT_TRIES {
        match TcpListener::bind(("127.0.0.1", port)) {
            Ok(l) => return run_server(l, port, cfg, &me),
            Err(_) => {
                if probe(port).as_deref() == Some(me.as_str()) {
                    println!("[Preview] 已有編輯伺服器在執行 (port {port})，已開啟的頁面會自動更新。");
                    open_url(port);
                    return Ok(false);
                }
            }
        }
    }
    bail!("找不到可用的埠 ({}-{})", BASE_PORT, BASE_PORT + PORT_TRIES - 1)
}

fn token() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    let mut out = String::new();
    for i in 0..2u64 {
        let mut h = RandomState::new().build_hasher();
        h.write_u64(nanos ^ i);
        out.push_str(&format!("{:016x}", h.finish()));
    }
    out
}

fn version(preview: &Path) -> String {
    std::fs::metadata(preview)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos().to_string())
        .unwrap_or_else(|| "0".into())
}

fn run_server(listener: TcpListener, port: u16, cfg: &Cfg, me: &str) -> Result<bool> {
    listener.set_nonblocking(true)?;
    let tok = token();
    println!("[Preview] 編輯頁面: http://127.0.0.1:{port}/  （在頁面按「儲存到專案」或 Ctrl+S 直接寫入檔案）");
    println!("[Preview] 編輯完成後，在這個視窗按 [Enter] 停止伺服器並回到主選單。");
    open_url(port);
    // Enter stops the server. If stdin is unusable (EOF / no console) fall back to stopping
    // automatically once the page has not been heard from for a while.
    let (tx, rx) = std::sync::mpsc::channel::<bool>();
    std::thread::spawn(move || {
        let mut line = String::new();
        let alive = matches!(std::io::stdin().read_line(&mut line), Ok(n) if n > 0);
        let _ = tx.send(alive);
    });
    let mut stdin_dead = false;
    let started = Instant::now();
    let mut last = Instant::now();
    let mut seen = false;
    loop {
        match rx.try_recv() {
            Ok(true) => {
                STOPPED_BY_ENTER.store(true, std::sync::atomic::Ordering::SeqCst);
                println!("[Preview] 已停止編輯伺服器。");
                return Ok(true);
            }
            Ok(false) => stdin_dead = true,
            Err(_) => {}
        }
        match listener.accept() {
            Ok((mut s, _)) => {
                let _ = s.set_nonblocking(false);
                let _ = s.set_read_timeout(Some(Duration::from_secs(10)));
                let _ = s.set_write_timeout(Some(Duration::from_secs(10)));
                handle(&mut s, cfg, me, &tok);
                seen = true;
                last = Instant::now();
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(40));
                let idle = stdin_dead && if seen { last.elapsed() > IDLE_AFTER } else { started.elapsed() > NO_CLIENT_AFTER };
                if idle {
                    println!("[Preview] 頁面已關閉，結束編輯伺服器。");
                    return Ok(true);
                }
            }
            Err(_) => std::thread::sleep(Duration::from_millis(40)),
        }
    }
}

struct Req {
    method: String,
    path: String,
    host: String,
    token: String,
    body: Vec<u8>,
}

fn read_req(s: &mut TcpStream) -> Option<Req> {
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 16384];
    let head_end = loop {
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
        if buf.len() > 65536 {
            return None;
        }
        let n = s.read(&mut tmp).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let mut lines = head.lines();
    let first = lines.next()?;
    let mut parts = first.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.split('?').next()?.to_string();
    let (mut host, mut token, mut len) = (String::new(), String::new(), 0usize);
    for l in lines {
        if let Some((k, v)) = l.split_once(':') {
            let v = v.trim();
            match k.trim().to_ascii_lowercase().as_str() {
                "host" => host = v.to_string(),
                "x-llk-token" => token = v.to_string(),
                "content-length" => len = v.parse().unwrap_or(0),
                _ => {}
            }
        }
    }
    if len > MAX_BODY {
        return None;
    }
    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < len {
        let n = s.read(&mut tmp).ok()?;
        if n == 0 {
            return None;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    body.truncate(len);
    Some(Req { method, path, host, token, body })
}

fn respond(s: &mut TcpStream, code: &str, ctype: &str, body: &[u8]) {
    let head = format!(
        "HTTP/1.1 {code}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = s.write_all(head.as_bytes());
    let _ = s.write_all(body);
    let _ = s.flush();
}

fn handle(s: &mut TcpStream, cfg: &Cfg, me: &str, tok: &str) {
    let Some(req) = read_req(s) else { return };
    // refuse DNS-rebinding style requests: only talk to the loopback name we handed out
    let host_ok = req.host.starts_with("127.0.0.1") || req.host.starts_with("localhost");
    if !host_ok {
        return respond(s, "403 Forbidden", "text/plain; charset=utf-8", b"bad host");
    }
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") | ("GET", "/index.html") => match std::fs::read_to_string(&cfg.preview) {
            Ok(html) => {
                let html = html.replace("__LLK_TOKEN__", tok).replace("__LLK_VER__", &version(&cfg.preview));
                respond(s, "200 OK", "text/html; charset=utf-8", html.as_bytes());
            }
            Err(e) => respond(s, "404 Not Found", "text/plain; charset=utf-8", format!("cannot read preview: {e}").as_bytes()),
        },
        ("GET", "/version") => respond(s, "200 OK", "text/plain", version(&cfg.preview).as_bytes()),
        ("GET", "/ping") => respond(s, "200 OK", "text/plain; charset=utf-8", format!("llk-studio\n{me}\n").as_bytes()),
        ("POST", "/save") => {
            if req.token != tok {
                return respond(s, "403 Forbidden", "text/plain; charset=utf-8", "token 不符".as_bytes());
            }
            match save(cfg, &req.body) {
                Ok(v) => respond(s, "200 OK", "application/json", v.to_string().as_bytes()),
                Err(e) => respond(s, "400 Bad Request", "text/plain; charset=utf-8", format!("{e:#}").as_bytes()),
            }
        }
        _ => respond(s, "404 Not Found", "text/plain", b"not found"),
    }
}

fn save(cfg: &Cfg, body: &[u8]) -> Result<Value> {
    let v: Value = serde_json::from_slice(body)?;
    let profile = v.get("profile").filter(|p| p.get("tile_mapping").is_some()).ok_or_else(|| anyhow::anyhow!("缺少 profile"))?;
    let mut n = 0;
    if let Some(list) = v["pngs"].as_array() {
        std::fs::create_dir_all(&cfg.templates)?;
        for item in list {
            let name = item["name"].as_str().unwrap_or("");
            // digits + ".png" only: nothing else may be written
            let stem = name.strip_suffix(".png").unwrap_or("");
            if stem.is_empty() || !stem.bytes().all(|b| b.is_ascii_digit()) {
                bail!("不合法的模板檔名: {name}");
            }
            let bytes = b64_decode(item["data"].as_str().unwrap_or(""))?;
            if bytes.len() < 8 || &bytes[1..4] != b"PNG" {
                bail!("{name} 不是 PNG");
            }
            std::fs::write(cfg.templates.join(name), bytes)?;
            n += 1;
        }
    }
    std::fs::write(&cfg.profile, serde_json::to_string_pretty(profile)?)?;
    Ok(json!({"ok": true, "profile": abs_str(&cfg.profile), "pngs": n}))
}

fn b64_decode(s: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' | b'\r' | b'\n' | b' ' => continue,
            _ => bail!("base64 格式錯誤"),
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(cfg: &Cfg, raw: &str) -> String {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let me = abs_str(&cfg.preview);
        let t = std::thread::scope(|sc| {
            let h = sc.spawn(|| {
                let (mut s, _) = l.accept().unwrap();
                handle(&mut s, cfg, &me, "TOK");
            });
            let mut c = TcpStream::connect(("127.0.0.1", port)).unwrap();
            c.write_all(raw.as_bytes()).unwrap();
            let mut out = String::new();
            c.read_to_string(&mut out).unwrap();
            h.join().unwrap();
            out
        });
        t
    }

    #[test]
    fn serves_and_saves() {
        let d = std::env::temp_dir().join(format!("llk_serve_test_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let cfg = Cfg { preview: d.join("preview.html"), profile: d.join("game_profile.json"), templates: d.join("templates") };
        std::fs::write(&cfg.preview, "T=__LLK_TOKEN__ V=__LLK_VER__").unwrap();

        let r = roundtrip(&cfg, "GET / HTTP/1.1\r\nHost: 127.0.0.1:1\r\n\r\n");
        assert!(r.contains("T=TOK V=") && !r.contains("__LLK"), "{r}");
        assert!(roundtrip(&cfg, "GET /ping HTTP/1.1\r\nHost: localhost\r\n\r\n").contains("llk-studio"));
        assert!(roundtrip(&cfg, "GET / HTTP/1.1\r\nHost: evil.example\r\n\r\n").contains("403"));

        // 8-byte PNG signature, base64
        let body = r#"{"profile":{"tile_mapping":[{"id":1}]},"pngs":[{"name":"7.png","data":"iVBORw0KGgo="}]}"#;
        let post = |tok: &str, body: &str| format!("POST /save HTTP/1.1\r\nHost: 127.0.0.1\r\nX-LLK-Token: {tok}\r\nContent-Length: {}\r\n\r\n{body}", body.len());
        assert!(roundtrip(&cfg, &post("bad", body)).contains("403"));
        assert!(!cfg.profile.exists());
        let ok = roundtrip(&cfg, &post("TOK", body));
        assert!(ok.contains("200 OK") && ok.contains("\"pngs\":1"), "{ok}");
        assert!(cfg.profile.is_file());
        assert_eq!(std::fs::read(cfg.templates.join("7.png")).unwrap(), b"\x89PNG\r\n\x1a\n");
        let evil = body.replace("7.png", "..\\x.png");
        assert!(roundtrip(&cfg, &post("TOK", &evil)).contains("400"));
        std::fs::remove_dir_all(&d).ok();
    }
}
