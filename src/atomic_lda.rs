use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

#[derive(Clone)]
pub struct LdaOrbital {
    pub n: u32,
    pub l: u32,
    pub label: String,
    pub radial_r: Vec<f32>,
    pub radial_rfn: Vec<f32>,
}

#[derive(Clone)]
pub struct LdaElement {
    pub symbol: String,
    pub orbitals: Vec<LdaOrbital>,
    pub occupancy: HashMap<(u32, u32), f32>,
    pub eigenvalues: HashMap<(u32, u32), f32>,
    pub total_electrons: f32,
    pub valence_electrons: f32,
    pub r_max: f32,
}

static ELEMENT_CACHE: Lazy<RwLock<HashMap<String, LdaElement>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

const BASE_URL: &str = "https://www.openmx-square.org/atoms/LDA";

pub async fn load_lda_element(symbol: &str) -> Result<LdaElement, String> {
    if let Some(cached) = ELEMENT_CACHE
        .read()
        .map_err(|_| "cache poisoned")?
        .get(symbol)
        .cloned()
    {
        return Ok(cached);
    }

    let data_dir = data_dir();
    fs::create_dir_all(&data_dir).map_err(|e| format!("data dir: {e}"))?;

    let (url, filename) = pick_alog_url(symbol).await?;
    let local_path = data_dir.join(filename);
    if !local_path.exists() {
        download_to(&url, &local_path).await?;
    }

    let element = parse_alog(&local_path, symbol)?;
    ELEMENT_CACHE
        .write()
        .map_err(|_| "cache poisoned")?
        .insert(symbol.to_string(), element.clone());
    Ok(element)
}

fn data_dir() -> PathBuf {
    PathBuf::from("data").join("openmx_lda")
}

async fn pick_alog_url(symbol: &str) -> Result<(String, String), String> {
    let page_url = format!("{BASE_URL}/{symbol}/");
    let html = reqwest::get(&page_url)
        .await
        .map_err(|e| format!("fetch element page: {e}"))?
        .text()
        .await
        .map_err(|e| format!("read element page: {e}"))?;

    let re = Regex::new(r#"(?i)href="([^"]+\.alog)""#)
        .map_err(|e| format!("regex: {e}"))?;
    let mut links: Vec<String> = re
        .captures_iter(&html)
        .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
        .collect();
    links.sort();
    links.dedup();

    if links.is_empty() {
        return Err(format!("no LDA .alog links found for {symbol}"));
    }

    let mut best = None;
    let mut best_score = i32::MIN;
    for link in links {
        let name = link.to_lowercase();
        let mut score = 0;
        if name.ends_with("0.alog") {
            score += 100;
        }
        if name.starts_with(&symbol.to_lowercase()) {
            score += 10;
        }
        if score > best_score {
            best_score = score;
            best = Some(link);
        }
    }

    let best = best.ok_or_else(|| format!("no suitable LDA file for {symbol}"))?;
    let filename = Path::new(&best)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or(&best)
        .to_string();
    Ok((format!("{BASE_URL}/{symbol}/{best}"), filename))
}

async fn download_to(url: &str, path: &Path) -> Result<(), String> {
    let bytes = reqwest::get(url)
        .await
        .map_err(|e| format!("download {url}: {e}"))?
        .bytes()
        .await
        .map_err(|e| format!("read bytes: {e}"))?;
    fs::write(path, &bytes).map_err(|e| format!("write file: {e}"))
}

fn parse_alog(path: &Path, symbol: &str) -> Result<LdaElement, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("open LDA file: {e}"))?;
    let mut content = String::new();
    file.read_to_string(&mut content)
        .map_err(|e| format!("read LDA file: {e}"))?;

    let total_electrons = extract_value(&content, "total.electron").unwrap_or(0.0);
    let valence_electrons = extract_value(&content, "valence.electron").unwrap_or(total_electrons);

    let occupancy = parse_occupancy(&content);
    let eigenvalues = parse_eigenvalues(&content);
    let (orbitals, r_max) = parse_radial_wavefunctions(&content)?;

    Ok(LdaElement {
        symbol: symbol.to_string(),
        orbitals,
        occupancy,
        eigenvalues,
        total_electrons,
        valence_electrons,
        r_max,
    })
}

fn extract_value(content: &str, key: &str) -> Option<f32> {
    let escaped = regex::escape(key);
    let re = Regex::new(&format!(r"{escaped}\s+([0-9Ee\+\-\.]+)")).ok()?;
    re.captures(content)
        .and_then(|cap| cap.get(1))
        .and_then(|m| m.as_str().parse::<f32>().ok())
}

fn parse_occupancy(content: &str) -> HashMap<(u32, u32), f32> {
    let mut occ = HashMap::new();
    let mut in_block = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("<ocupied.electrons") {
            in_block = true;
            continue;
        }
        if in_block {
            if trimmed.starts_with("ocupied.electrons>") {
                break;
            }
            if trimmed.is_empty() {
                continue;
            }
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }
            let n = match parts[0].parse::<u32>() {
                Ok(v) => v,
                Err(_) => continue,
            };
            for (l, val) in parts.iter().skip(1).enumerate() {
                if let Ok(v) = val.parse::<f32>() {
                    if v > 0.0 {
                        occ.insert((n, l as u32), v);
                    }
                }
            }
        }
    }
    occ
}

fn parse_eigenvalues(content: &str) -> HashMap<(u32, u32), f32> {
    let mut eigen = HashMap::new();
    let re = Regex::new(r"n=\s*(\d+)\s*l=\s*(\d+)\s*([\-0-9Ee\+\.]+)")
        .unwrap();
    for cap in re.captures_iter(content) {
        if let (Ok(n), Ok(l), Ok(e)) = (
            cap[1].parse::<u32>(),
            cap[2].parse::<u32>(),
            cap[3].parse::<f32>(),
        ) {
            eigen.insert((n, l), e);
        }
    }
    eigen
}

fn parse_radial_wavefunctions(content: &str) -> Result<(Vec<LdaOrbital>, f32), String> {
    let lines: Vec<&str> = content.lines().collect();
    let mut start = None;
    let mut end = None;
    for (i, line) in lines.iter().enumerate() {
        if line.contains("Radial wave functions") {
            start = Some(i);
        }
        if start.is_some() && line.contains("Charge density") {
            end = Some(i);
            break;
        }
    }
    let start = start.ok_or_else(|| "missing radial wave function section".to_string())?;
    let end = end.unwrap_or(lines.len());

    let mut current_n: Option<u32> = None;
    let mut radial_r: HashMap<(u32, u32), Vec<f32>> = HashMap::new();
    let mut radial_v: HashMap<(u32, u32), Vec<f32>> = HashMap::new();

    for line in &lines[start..end] {
        let trimmed = line.trim();
        if trimmed.starts_with("n=") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(n) = parts[1].parse::<u32>() {
                    current_n = Some(n);
                }
            }
            continue;
        }

        let n = match current_n {
            Some(v) => v,
            None => continue,
        };
        if trimmed.is_empty() {
            continue;
        }
        if !trimmed
            .chars()
            .next()
            .map(|c| c.is_ascii_digit() || c == '-' || c == '+')
            .unwrap_or(false)
        {
            continue;
        }

        let vals: Vec<f32> = trimmed
            .split_whitespace()
            .filter_map(|v| v.parse::<f32>().ok())
            .collect();
        if vals.len() < 3 {
            continue;
        }
        let r = vals[1];
        let l_count = vals.len() - 2;
        for l in 0..l_count {
            let key = (n, l as u32);
            radial_r.entry(key).or_insert_with(Vec::new).push(r);
            radial_v
                .entry(key)
                .or_insert_with(Vec::new)
                .push(vals[2 + l]);
        }
    }

    let mut orbitals = Vec::new();
    let mut r_max = 0.0_f32;
    for (key, r_vals) in radial_r {
        if let Some(v_vals) = radial_v.get(&key) {
            if r_vals.is_empty() || v_vals.is_empty() {
                continue;
            }
            let (n, l) = key;
            let label = format!("{n}{}", l_to_letter(l));
            if let Some(last) = r_vals.last() {
                if *last > r_max {
                    r_max = *last;
                }
            }
            orbitals.push(LdaOrbital {
                n,
                l,
                label,
                radial_r: r_vals,
                radial_rfn: v_vals.clone(),
            });
        }
    }

    orbitals.sort_by(|a, b| (a.n, a.l).cmp(&(b.n, b.l)));
    Ok((orbitals, r_max))
}

fn l_to_letter(l: u32) -> &'static str {
    match l {
        0 => "s",
        1 => "p",
        2 => "d",
        3 => "f",
        4 => "g",
        5 => "h",
        6 => "i",
        _ => "?",
    }
}
