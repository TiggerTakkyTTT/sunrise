//! Discover call-table roles from one file per newly exposed trait.
//! This classifies calls only: value layouts still come exclusively from Wire.
#[cfg(test)]
#[path = "trait_roles/tests.rs"]
mod tests;
use std::{
    cell::Cell,
    collections::{BTreeMap, HashSet},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

struct Roles {
    imports: Vec<&'static str>,
    exports: Vec<&'static str>,
    held: Vec<&'static str>,
}
thread_local! {
    static CURRENT: Cell<Option<&'static Roles>> = const { Cell::new(None) };
}
// Method descriptors already retain static owner names. Intern each distinct
// source surface once for the generation process; fixture roots can select
// different surfaces concurrently without contaminating one another.
type Key = (Vec<String>, Vec<String>, Vec<String>);
static SURFACES: OnceLock<Mutex<BTreeMap<Key, &'static Roles>>> = OnceLock::new();

/// The roles of one source surface, as discovered from its role files.
pub(super) struct Declared {
    pub imports: Vec<String>,
    /// Every export, `.held` ones and the capability handlers included.
    pub exports: Vec<String>,
    /// The exports the game also hands a mod objects of.
    pub held: Vec<String>,
    /// What cargo must watch: the role files and their folder.
    pub paths: Vec<PathBuf>,
}

fn current(pick: fn(&Roles) -> &[&'static str]) -> &'static [&'static str] {
    CURRENT.with(|current| current.get().map_or(&[][..], pick))
}
pub(super) fn imports() -> &'static [&'static str] {
    current(|roles| &roles.imports)
}
pub(super) fn exports() -> &'static [&'static str] {
    current(|roles| &roles.exports)
}
pub(super) fn held() -> &'static [&'static str] {
    current(|roles| &roles.held)
}
pub(super) fn select(declared: &Declared) {
    let mut surfaces = SURFACES.get_or_init(Mutex::default).lock().unwrap();
    let key = (
        declared.imports.clone(),
        declared.exports.clone(),
        declared.held.clone(),
    );
    let roles = *surfaces
        .entry(key)
        .or_insert_with_key(|(imports, exports, held)| {
            let intern = |names: &[String]| {
                names
                    .iter()
                    .map(|name| &*Box::leak(name.clone().into_boxed_str()))
                    .collect()
            };
            Box::leak(Box::new(Roles {
                imports: intern(imports),
                exports: intern(exports),
                held: intern(held),
            }))
        });
    CURRENT.with(|current| current.set(Some(roles)));
}

/// Read the role files in file-name order, which is the order of the call table (and so of the
/// call ids): `<Trait>.import`, `<Trait>.export`, or `<Trait>.held` for an export the game also
/// hands a mod objects of.
pub(super) fn declared(root: &Path) -> Result<Declared, String> {
    let directory = root.join("crates/sunrise-mod-vocab/src/wire/traits");
    let mut imports = Vec::new();
    let mut exports = Vec::new();
    let mut held = Vec::new();
    if !directory.exists() {
        return Ok(Declared {
            imports,
            exports,
            held,
            paths: Vec::new(),
        });
    }
    let mut paths = std::fs::read_dir(&directory)
        .map_err(|error| format!("{}: {error}", directory.display()))?
        .map(|entry| {
            entry
                .map(|entry| entry.path())
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    paths.sort();
    let mut seen = HashSet::new();
    for path in &paths {
        let role = path.extension().and_then(|value| value.to_str());
        let target = match role {
            Some("import") => &mut imports,
            Some("export") => &mut exports,
            Some("held") => {
                let name = path.file_stem().and_then(|value| value.to_str());
                held.extend(name.map(str::to_owned));
                &mut exports
            }
            Some("md") => continue,
            _ => {
                return Err(format!(
                    "{}: expected a .import, .export or .held trait role",
                    path.display()
                ));
            }
        };
        let name = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| format!("{}: trait name is not UTF-8", path.display()))?;
        syn::parse_str::<syn::Ident>(name)
            .map_err(|error| format!("{}: invalid trait name: {error}", path.display()))?;
        if !seen.insert(name.to_owned()) {
            return Err(format!(
                "{}: duplicate Wire trait role for {name}",
                path.display()
            ));
        }
        target.push(name.to_owned());
    }
    // Cargo must rerun when a new independent trait file is added or removed.
    paths.push(directory);
    Ok(Declared {
        imports,
        exports,
        held,
        paths,
    })
}
