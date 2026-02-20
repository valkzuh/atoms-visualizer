use once_cell::sync::Lazy;
use quick_xml::events::Event;
use quick_xml::Reader;
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

#[derive(Clone)]
pub struct Orbital {
    pub n: u32,
    pub l: u32,
    pub label: String,
    pub radial_r: Vec<f32>,
    pub radial_chi: Vec<f32>,
}

#[derive(Clone)]
pub struct ElementData {
    pub symbol: String,
    pub orbitals: Vec<Orbital>,
    pub r_max: f32,
}

static ELEMENT_CACHE: Lazy<RwLock<HashMap<String, ElementData>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

const BASE_URL: &str = "https://pseudopotentials.quantum-espresso.org";

pub async fn load_element_data(symbol: &str, z: u32) -> Result<ElementData, String> {
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

    let upf_path = data_dir.join(format!("{symbol}.UPF"));
    if !upf_path.exists() {
        let url = pick_upf_url(symbol, z).await?;
        download_to(&url, &upf_path).await?;
    }

    let element = parse_upf(&upf_path, symbol)?;
    ELEMENT_CACHE
        .write()
        .map_err(|_| "cache poisoned")?
        .insert(symbol.to_string(), element.clone());
    Ok(element)
}

fn data_dir() -> PathBuf {
    PathBuf::from("data").join("pslibrary")
}

async fn pick_upf_url(symbol: &str, z: u32) -> Result<String, String> {
    let page_url = format!("{BASE_URL}/legacy_tables/ps-library/{}", symbol.to_lowercase());
    let html = reqwest::get(&page_url)
        .await
        .map_err(|e| format!("fetch element page: {e}"))?
        .text()
        .await
        .map_err(|e| format!("read element page: {e}"))?;

    let re = Regex::new(r#"href="(/upf_files/[^"]+\.UPF)""#)
        .map_err(|e| format!("regex: {e}"))?;
    let mut links: Vec<String> = re
        .captures_iter(&html)
        .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
        .collect();
    links.sort();
    links.dedup();

    if links.is_empty() {
        return Err(format!("no UPF links found for {symbol}"));
    }

    let mut best = None;
    let mut best_score = i32::MIN;
    for link in links {
        let name = link.to_lowercase();
        let mut score = 0;
        if name.contains("pbe") {
            score += 100;
        }
        if name.contains("kjpaw") {
            score += 60;
        }
        if name.contains("rrkjus") {
            score += 30;
        }
        if name.contains("psl.1.0.0") {
            score += 20;
        }
        if name.contains("rel-") {
            score += if z >= 36 { 10 } else { -5 };
        }
        if name.contains("pbesol") {
            score -= 5;
        }
        if name.contains("pz") {
            score -= 10;
        }
        if name.contains("0.1") {
            score -= 5;
        }

        if score > best_score {
            best_score = score;
            best = Some(link);
        }
    }

    let best = best.ok_or_else(|| format!("no suitable UPF for {symbol}"))?;
    Ok(format!("{BASE_URL}{best}"))
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

fn parse_upf(path: &Path, symbol: &str) -> Result<ElementData, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("open UPF: {e}"))?;
    let mut content = String::new();
    file.read_to_string(&mut content)
        .map_err(|e| format!("read UPF: {e}"))?;

    let mut reader = Reader::from_str(&content);
    reader.trim_text(true);
    let mut buf = Vec::new();

    let mut radial_r: Vec<f32> = Vec::new();
    let mut orbitals: Vec<Orbital> = Vec::new();
    let mut in_pp_r = false;
    let mut current_label: Option<String> = None;
    let mut current_l: Option<u32> = None;
    let mut current_vals: Vec<f32> = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let name = e.name().as_ref().to_vec();
                if name == b"PP_R" {
                    in_pp_r = true;
                } else if name.starts_with(b"PP_CHI") {
                    current_label = None;
                    current_l = None;
                    current_vals.clear();
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"label" {
                            current_label = Some(attr.unescape_value().unwrap_or_default().to_string());
                        } else if attr.key.as_ref() == b"l" {
                            if let Ok(v) = attr.unescape_value().unwrap_or_default().parse::<u32>() {
                                current_l = Some(v);
                            }
                        }
                    }
                }
            }
            Ok(Event::Text(e)) => {
                let text = e.unescape().unwrap_or_default();
                if in_pp_r {
                    radial_r.extend(parse_floats(&text));
                } else if current_label.is_some() {
                    current_vals.extend(parse_floats(&text));
                }
            }
            Ok(Event::End(e)) => {
                let name = e.name().as_ref().to_vec();
                if name == b"PP_R" {
                    in_pp_r = false;
                } else if name.starts_with(b"PP_CHI") {
                    if let (Some(label), Some(l)) = (current_label.take(), current_l.take()) {
                        let n = parse_principal_n(&label);
                        orbitals.push(Orbital {
                            n,
                            l,
                            label,
                            radial_r: radial_r.clone(),
                            radial_chi: current_vals.clone(),
                        });
                    }
                    current_vals.clear();
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(format!("XML parse error: {e}")),
            _ => {}
        }
        buf.clear();
    }

    if radial_r.is_empty() || orbitals.is_empty() {
        return Err(format!("UPF missing data for {symbol}"));
    }

    let r_max = *radial_r.last().unwrap_or(&0.0);
    Ok(ElementData {
        symbol: symbol.to_string(),
        orbitals,
        r_max,
    })
}

fn parse_principal_n(label: &str) -> u32 {
    let digits: String = label.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse::<u32>().unwrap_or(0)
}

fn parse_floats(text: &str) -> Vec<f32> {
    text.split_whitespace()
        .filter_map(|v| v.parse::<f32>().ok())
        .collect()
}

pub fn symbol_for_z(z: u32) -> Option<&'static str> {
    ELEMENT_SYMBOLS.get((z as usize).saturating_sub(1)).copied()
}

const ELEMENT_SYMBOLS: [&str; 118] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];
