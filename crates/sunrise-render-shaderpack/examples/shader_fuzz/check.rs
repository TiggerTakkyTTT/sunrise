//! One case through the game's load path, twice, and the invariants every load must keep.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::hash::Hasher;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use sunrise_render_shaderpack::pack_atlas::{PackAtlas, game_inputs};
use sunrise_shaders::{Macros, Pack, Post, PostProgram};

use crate::cases::DIMENSIONS;

/// What a case asks of the checker besides its files.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub dim: usize,
    pub flat: bool,
    /// Programs that must load or have a correctly named refusal diagnostic: the unmutated
    /// case loaded them and this candidate's active files still include them.
    pub expect: Vec<String>,
    /// Bytes a stage file may grow to through `#include` and macros.
    pub size_limit: usize,
}

/// A broken invariant: its name, and what about it (a program, a location, an error kind).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Failure {
    pub invariant: String,
    pub detail: String,
}

impl Failure {
    pub fn new(invariant: &str, detail: impl Into<String>) -> Self {
        Self {
            invariant: invariant.to_owned(),
            detail: detail.into(),
        }
    }

    pub fn signature(&self) -> String {
        format!("{}|{}", self.invariant, self.detail)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outcome {
    pub failures: Vec<Failure>,
    /// The programs that loaded, in the order `observe` lists them.
    pub loaded: Vec<String>,
    pub shadow_pass: bool,
}

/// What one load gave, reduced to what the invariants compare.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Observed {
    pub loaded: Vec<String>,
    pub folder: String,
    pub stage_files: Vec<String>,
    pub notes: Vec<String>,
    /// Loader notes that actually say a program was skipped; property/custom diagnostics are not refusals.
    pub refusals: Vec<String>,
    pub shadow_pass: bool,
    pub custom: sunrise_shaders::Custom,
    pub stage_sizes: BTreeMap<String, usize>,
    /// Each module that does not validate: `program stage`, and naga's error.
    pub invalid: Vec<(String, String)>,
    /// A hash of each part of the output, labelled.
    pub pieces: Vec<(String, u64)>,
}

/// The pack in `dir` loaded as the game loads a pack for a dimension (`shader_packs.rs`): the
/// standard macros and the renderer's at Mipmap Levels 4, `shaders.properties` with the
/// features it turns on, the dimension's folder, and the game's own inputs.
pub fn load(dir: &Path, request: &Request) -> Observed {
    let pack = Pack::open(dir).expect("the case folder is a pack");
    let mut macros = Macros::standard();
    macros.define_renderer(4);
    let (properties, feature_notes) = sunrise_shaders::read_properties(&pack, &mut macros);
    let mut notes = feature_notes;
    notes.extend(
        properties
            .notes
            .iter()
            .map(|note| format!("shaders.properties:{}: {}", note.line, note.message)),
    );
    let (id, legacy) = DIMENSIONS[request.dim];
    let folder = sunrise_shaders::dimension_folder_named(&pack, id, legacy);
    let (post, stage_sizes) = sunrise_shaders::load_post_with_sizes(
        &pack,
        &folder,
        &properties,
        &macros,
        &inputs(request.flat),
    );
    let refusals = post
        .notes
        .iter()
        .filter(|note| note.ends_with("skipped"))
        .cloned()
        .collect();
    let prefix = if post.folder.is_empty() {
        String::new()
    } else {
        format!("{}/", post.folder)
    };
    let stage_files = pack
        .all_files()
        .into_iter()
        .filter(|path| {
            let Some(relative) = path.strip_prefix(&prefix) else {
                return false;
            };
            !relative.contains('/')
                && [".vsh", ".fsh", ".gsh", ".csh", ".tcs", ".tes"]
                    .iter()
                    .any(|extension| relative.ends_with(extension))
        })
        .collect();
    notes.extend(post.notes.iter().cloned());
    notes.extend(post.config.notes.iter().cloned());
    notes.extend(post.custom.notes.iter().cloned());
    observe(&post, notes, refusals, stage_files, stage_sizes)
}

/// The game's inputs with a terrain atlas of 1024 x 1024 pixels in 16-pixel sprites: the
/// atlas's size only reaches the GLSL as constants. `flat` picks the game's flat draws.
pub fn inputs(flat: bool) -> sunrise_shaders::GameInputs {
    let terrain = PackAtlas::terrain_inputs_sized([1024, 1024], 64, 16, false, flat);
    game_inputs(Some(terrain), flat, &[3, 4])
}

/// Every program of `post` with a slot, by slot.
fn programs(post: &Post) -> Vec<(&'static str, &PostProgram)> {
    let slots: [(&'static str, &Option<PostProgram>); 23] = [
        ("terrain", &post.terrain),
        ("terrain_cutout", &post.terrain_cutout),
        ("water", &post.water),
        ("sky", &post.sky),
        ("sky_textured", &post.sky_textured),
        ("entities", &post.entities),
        ("block", &post.block),
        ("block_translucent", &post.block_translucent),
        ("entities_translucent", &post.entities_translucent),
        ("particles", &post.particles),
        ("particles_translucent", &post.particles_translucent),
        ("beacon_beam", &post.beacon_beam),
        ("damaged_block", &post.damaged_block),
        ("clouds", &post.clouds),
        ("lines", &post.lines),
        ("weather", &post.weather),
        ("hand", &post.hand),
        ("hand_water", &post.hand_water),
        ("shadow", &post.shadow),
        ("shadow_water", &post.shadow_water),
        ("shadow_water_plain", &post.shadow_water_plain),
        ("shadow_entities", &post.shadow_entities),
        ("shadow_block", &post.shadow_block),
    ];
    let mut out: Vec<(&'static str, &PostProgram)> =
        post.programs.iter().map(|p| ("pass", p)).collect();
    out.extend(post.shadow_comp.iter().map(|p| ("shadowcomp", p)));
    out.extend(
        slots
            .into_iter()
            .filter_map(|(slot, p)| Some((slot, p.as_ref()?))),
    );
    out
}

/// naga's error for `module`, validated as the translator validates (every check, every
/// capability), with its sources; `None` when it validates.
pub fn validate(module: &naga::Module) -> Option<String> {
    let flags = naga::valid::ValidationFlags::all();
    let caps = naga::valid::Capabilities::all();
    let error = naga::valid::Validator::new(flags, caps)
        .validate(module)
        .err()?;
    let mut text = error.as_inner().to_string();
    let mut source = std::error::Error::source(error.as_inner());
    while let Some(e) = source {
        let _ = write!(text, ": {e}");
        source = e.source();
    }
    Some(text)
}

/// Hashes what `Debug` writes, without holding the text.
struct HashWriter(std::hash::DefaultHasher);

impl std::fmt::Write for HashWriter {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.write(s.as_bytes());
        Ok(())
    }
}

fn hash(value: &dyn std::fmt::Debug) -> u64 {
    let mut w = HashWriter(std::hash::DefaultHasher::new());
    let _ = write!(w, "{value:?}");
    w.0.finish()
}

pub fn observe(
    post: &Post,
    notes: Vec<String>,
    refusals: Vec<String>,
    stage_files: Vec<String>,
    stage_sizes: BTreeMap<String, usize>,
) -> Observed {
    let mut seen = Observed {
        folder: post.folder.clone(),
        stage_files,
        refusals,
        shadow_pass: post.shadow_pass,
        custom: post.custom.clone(),
        stage_sizes,
        notes,
        ..Default::default()
    };
    seen.pieces.push((
        "render settings".into(),
        hash(&(
            &post.folder,
            &post.deferred_pre,
            &post.composite_pre,
            post.terrain_undrawn,
            post.water_undrawn,
            post.clouds_off,
            &post.sky_switches,
            post.shadow_pass,
            post.shadow_only_player,
            post.shadow_only_lit_blocks,
            post.translucent_late,
            post.particles_late,
        )),
    ));
    seen.pieces.push((
        "render resources".into(),
        hash(&(
            &post.textures,
            &post.sizes,
            post.shadow_colors,
            &post.buffer_objects,
            &post.images,
        )),
    ));
    seen.pieces.push((
        "render configuration".into(),
        hash(&(&post.shadow_culling, &post.config)),
    ));
    seen.pieces.push(("notes".into(), hash(&seen.notes)));
    for (slot, p) in programs(post) {
        if !seen.loaded.contains(&p.name) {
            seen.loaded.push(p.name.clone());
        }
        let label = format!("{slot} {}", p.name);
        let parts = (
            &p.buffers,
            &p.flips,
            &p.blend,
            &p.buffer_blends,
            p.mipmaps,
            p.group,
            p.vanilla_sky_blends,
            p.geometry_triangles,
            p.scale,
        );
        seen.pieces
            .push((format!("{label} settings"), hash(&parts)));
        let mut modules: Vec<(String, &naga::Module)> = Vec::new();
        if let Some(program) = &p.program {
            modules.push((format!("{label} vertex"), &program.vertex));
            modules.push((format!("{label} fragment"), &program.fragment));
            seen.pieces
                .push((format!("{label} interface"), hash(&program.interface)));
        }
        for (name, c) in &p.compute {
            modules.push((format!("{label} compute {name}"), &c.module));
            let groups = (&c.interface, c.work_groups, c.work_groups_render);
            seen.pieces
                .push((format!("{label} compute {name} interface"), hash(&groups)));
            seen.pieces
                .push((format!("{label} compute {name} notes"), hash(&c.notes)));
        }
        for (what, module) in modules {
            seen.pieces.push((what.clone(), hash(module)));
            if let Some(error) = validate(module) {
                seen.invalid.push((what, error));
            }
        }
    }
    seen
}

fn stage_owner(path: &str) -> Option<String> {
    let filename = path.rsplit('/').next()?;
    let (stem, extension) = filename.rsplit_once('.')?;
    if !["vsh", "fsh", "gsh", "csh", "tcs", "tes"].contains(&extension) {
        return None;
    }
    Some(if extension == "csh" {
        crate::cases::compute_stage_owner(stem).to_owned()
    } else {
        stem.to_owned()
    })
}

fn names_program(note: &str, program: &str, stage_files: &[String]) -> bool {
    if note
        .strip_prefix(program)
        .is_some_and(|rest| rest.starts_with(": "))
    {
        return true;
    }
    stage_files.iter().any(|path| {
        if stage_owner(path).as_deref() != Some(program) {
            return false;
        }
        let filename = path.rsplit('/').next().unwrap_or(path);
        let stem = filename.rsplit_once('.').map_or(filename, |(stem, _)| stem);
        note.strip_prefix(path)
            .or_else(|| note.strip_prefix(filename))
            .is_some_and(|rest| valid_location(rest))
            || note
                .strip_prefix(stem)
                .is_some_and(|rest| rest.starts_with(": "))
    })
}

fn valid_location(rest: &str) -> bool {
    let Some(rest) = rest.strip_prefix(':') else {
        return false;
    };
    let Some((line, suffix)) = rest.split_once(": ") else {
        return false;
    };
    !suffix.is_empty() && line.parse::<u32>().is_ok()
}

/// A diagnostic names a subject only when it identifies a loaded program or one of this
/// candidate's active stage files. Property keys, feature names and arbitrary prefixes do not.
pub fn named(note: &str, loaded: &[String], stage_files: &[String]) -> bool {
    loaded
        .iter()
        .any(|program| names_program(note, program, stage_files))
        || stage_files.iter().any(|path| {
            stage_owner(path).is_some_and(|program| names_program(note, &program, stage_files))
        })
}

/// Text that groups alike: numbers (handles, lines, sizes) read `#`, and at most 120 chars.
pub fn shape(text: &str) -> String {
    let mut out = String::new();
    let mut digits = false;
    for c in text.chars() {
        if c.is_ascii_digit() {
            if !digits {
                out.push('#');
            }
            digits = true;
        } else {
            digits = false;
            out.push(c);
        }
        if out.len() >= 120 {
            break;
        }
    }
    out
}

/// The invariants over two loads of the same case (each a panic message when it panicked).
pub fn judge(
    first: Result<Observed, String>,
    second: Result<Observed, String>,
    request: &Request,
) -> Outcome {
    let mut failures = Vec::new();
    for (label, result) in [("first", &first), ("second", &second)] {
        match result {
            Err(panic) => failures.push(Failure::new("panic", format!("{label} load: {panic}"))),
            Ok(observed) => {
                for (what, error) in &observed.invalid {
                    failures.push(Failure::new("naga", format!("{what}: {}", shape(error))));
                }
                for note in observed
                    .refusals
                    .iter()
                    .filter(|note| !named(note, &observed.loaded, &observed.stage_files))
                {
                    failures.push(Failure::new(
                        "refusal",
                        format!("unnamed note: {}", shape(note)),
                    ));
                }
                for name in &request.expect {
                    let gone = !observed.loaded.contains(name);
                    let explained = observed
                        .refusals
                        .iter()
                        .any(|note| names_program(note, name, &observed.stage_files));
                    // The shadow pass may be off now: its programs then don't load, by design.
                    let shadow = name.starts_with("shadow") && !observed.shadow_pass;
                    if gone && !explained && !shadow {
                        failures.push(Failure::new(
                            "refusal",
                            format!("{name}: not loaded, no note"),
                        ));
                    }
                }
            }
        }
    }
    if let (Ok(first), Ok(second)) = (&first, &second) {
        let differs = first
            .pieces
            .iter()
            .zip(&second.pieces)
            .find(|(a, b)| a != b)
            .map(|(a, _)| a.0.clone())
            .or_else(|| (first.pieces.len() != second.pieces.len()).then(|| "program list".into()))
            .or_else(|| (first.loaded != second.loaded).then(|| "loaded programs".into()))
            .or_else(|| (first.folder != second.folder).then(|| "dimension folder".into()))
            .or_else(|| {
                (first.stage_files != second.stage_files).then(|| "active stage files".into())
            })
            .or_else(|| (first.shadow_pass != second.shadow_pass).then(|| "shadow pass".into()))
            .or_else(|| (first.custom != second.custom).then(|| "custom definitions".into()))
            .or_else(|| {
                (first.stage_sizes != second.stage_sizes).then(|| "preprocessed stage sizes".into())
            });
        if let Some(piece) = differs {
            failures.push(Failure::new("determinism", piece));
        }
    }
    failures.sort();
    failures.dedup();
    Outcome {
        failures,
        loaded: first
            .as_ref()
            .ok()
            .map_or_else(Vec::new, |o| o.loaded.clone()),
        shadow_pass: first.as_ref().ok().is_some_and(|o| o.shadow_pass),
    }
}

thread_local! {
    /// Where the last panic on this thread happened, set by the worker's hook.
    pub static PANIC_AT: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(at) = PANIC_AT.with(|p| p.borrow_mut().take()) {
        return at;
    }
    let message = payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_default();
    shape(&message)
}

/// The loader-processed stage files' preprocessed sizes, including rejected tessellation stages;
/// failures state the measured size and limit.
pub fn oversized(stage_sizes: &BTreeMap<String, usize>, limit: usize) -> Vec<Failure> {
    stage_sizes
        .iter()
        .filter(|(_, size)| **size > limit)
        .map(|(file, size)| {
            Failure::new(
                "size",
                format!("{file}: {size} bytes after preprocessing exceeds {limit} bytes"),
            )
        })
        .collect()
}

fn oversized_loads(
    first: &Result<Observed, String>,
    second: &Result<Observed, String>,
    limit: usize,
) -> Vec<Failure> {
    [&first, &second]
        .into_iter()
        .filter_map(|result| result.as_ref().ok())
        .flat_map(|observed| oversized(&observed.stage_sizes, limit))
        .collect()
}

/// The case in `dir` checked: no panic, every program valid, every refusal named, the same
/// output twice, and every loader-processed stage within the size limit. Time and memory bounds
/// are the worker's.
pub fn check(dir: &Path, request: &Request) -> Outcome {
    let once = || catch_unwind(AssertUnwindSafe(|| load(dir, request))).map_err(panic_text);
    let first = once();
    let second = once();
    let mut size_failures = oversized_loads(&first, &second, request.size_limit);
    let mut outcome = judge(first, second, request);
    outcome.failures.append(&mut size_failures);
    outcome.failures.sort();
    outcome.failures.dedup();
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(expect: &[&str]) -> Request {
        Request {
            dim: 0,
            flat: true,
            expect: expect.iter().map(|s| (*s).to_owned()).collect(),
            size_limit: 1 << 24,
        }
    }

    fn seen(loaded: &[&str], notes: &[&str]) -> Observed {
        Observed {
            loaded: loaded.iter().map(|s| (*s).to_owned()).collect(),
            folder: String::new(),
            stage_files: [
                "composite.vsh",
                "composite.fsh",
                "final.vsh",
                "final.fsh",
                "shadow.vsh",
                "shadow.fsh",
            ]
            .map(str::to_owned)
            .to_vec(),
            notes: notes.iter().map(|s| (*s).to_owned()).collect(),
            refusals: notes
                .iter()
                .filter(|note| note.ends_with("skipped"))
                .map(|s| (*s).to_owned())
                .collect(),
            shadow_pass: false,
            custom: Default::default(),
            stage_sizes: BTreeMap::new(),
            invalid: Vec::new(),
            pieces: vec![("composite fragment".into(), 7)],
        }
    }

    #[test]
    fn notes_name_their_subject() {
        let loaded = ["composite".to_owned()];
        let stages = [
            "composite.vsh".to_owned(),
            "composite.fsh".to_owned(),
            "composite1_a.csh".to_owned(),
        ];
        for ok in [
            "composite: skipped",
            "composite1_a: skipped",
            "composite1_a.csh:12: unknown directive `#foo`",
        ] {
            assert!(named(ok, &loaded, &stages), "{ok}");
        }
        for unnamed in [
            "skipped",
            ": skipped",
            " composite: x",
            "arbitrary-prefix: refused",
            "arbitrary-prefix: skipped",
            "iris.features.required: unknown feature",
            "shaders.properties:7: not a key=value line",
            "uniform.float.fzA: expression did not parse",
            "`fzA` reads 0: unknown",
            "no program for translucent terrain; nothing draws it",
        ] {
            assert!(!named(unnamed, &loaded, &stages), "{unnamed}");
        }
    }

    #[test]
    fn a_panic_is_the_only_failure_of_its_load() {
        let out = judge(
            Err("post.rs:1".into()),
            Ok(seen(&["final"], &[])),
            &request(&["final"]),
        );
        assert_eq!(
            out.failures,
            [Failure::new("panic", "first load: post.rs:1")]
        );
        let out = judge(
            Ok(seen(&["final"], &[])),
            Err("post.rs:1".into()),
            &request(&["final"]),
        );
        assert_eq!(
            out.failures,
            [Failure::new("panic", "second load: post.rs:1")]
        );
    }

    #[test]
    fn a_program_dropped_without_a_note_is_an_unnamed_refusal() {
        let req = request(&["composite", "final", "shadow"]);
        let out = judge(
            Ok(seen(&["deferred"], &["final: skipped"])),
            Ok(seen(&["deferred"], &["final: skipped"])),
            &req,
        );
        // `final` is named and the shadow pass is off: only `composite` went silently.
        assert_eq!(
            out.failures,
            [Failure::new("refusal", "composite: not loaded, no note")]
        );
        let out = judge(
            Ok(seen(&[], &["skipped"])),
            Ok(seen(&[], &["skipped"])),
            &request(&[]),
        );
        assert_eq!(
            out.failures,
            [Failure::new("refusal", "unnamed note: skipped")]
        );
    }

    #[test]
    fn feature_and_parser_diagnostics_do_not_explain_a_silent_drop() {
        let notes = [
            "iris.features.required: unknown feature",
            "shaders.properties:7: not a key=value line",
            "uniform.float.fzA: expression did not parse",
        ];
        let out = judge(
            Ok(seen(&[], &notes)),
            Ok(seen(&[], &notes)),
            &request(&["composite"]),
        );
        assert_eq!(
            out.failures,
            [Failure::new("refusal", "composite: not loaded, no note")]
        );
    }

    #[test]
    fn a_named_skip_explains_the_expected_program() {
        let out = judge(
            Ok(seen(&[], &["composite: skipped"])),
            Ok(seen(&[], &["composite: skipped"])),
            &request(&["composite"]),
        );
        assert!(out.failures.is_empty(), "{:?}", out.failures);
    }

    #[test]
    fn the_second_load_is_also_checked_against_the_size_limit() {
        let first = Ok(Observed::default());
        let second = Ok(Observed {
            stage_sizes: [("composite.fsh".to_owned(), 17)].into(),
            ..Default::default()
        });
        assert_eq!(
            oversized_loads(&first, &second, 16),
            [Failure::new(
                "size",
                "composite.fsh: 17 bytes after preprocessing exceeds 16 bytes"
            )]
        );
    }

    #[test]
    fn two_loads_that_differ_name_the_first_part() {
        let mut other = seen(&["composite"], &[]);
        other.pieces[0].1 = 8;
        let out = judge(Ok(seen(&["composite"], &[])), Ok(other), &request(&[]));
        assert_eq!(
            out.failures,
            [Failure::new("determinism", "composite fragment")]
        );
        let out = judge(
            Ok(seen(&["composite"], &[])),
            Err("x".into()),
            &request(&[]),
        );
        assert_eq!(out.failures, [Failure::new("panic", "second load: x")]);
        let mut shadow = seen(&["composite"], &[]);
        shadow.shadow_pass = true;
        let out = judge(Ok(seen(&["composite"], &[])), Ok(shadow), &request(&[]));
        assert_eq!(out.failures, [Failure::new("determinism", "shadow pass")]);
    }

    #[test]
    fn render_driving_load_results_are_compared() {
        fn loaded(properties: &str) -> Observed {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!(
                "sunrise-shader-fuzz-{}-{nonce}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let files = BTreeMap::from([
                (
                    "composite.vsh".to_owned(),
                    b"#version 120\nvoid main() { gl_Position = gl_Vertex; }".to_vec(),
                ),
                (
                    "composite.fsh".to_owned(),
                    b"#version 120\nvoid main() { gl_FragColor = vec4(1.0); }".to_vec(),
                ),
                (
                    "shaders.properties".to_owned(),
                    properties.as_bytes().to_vec(),
                ),
            ]);
            crate::write_pack(&dir, &files).unwrap();
            let seen = load(&dir, &request(&[]));
            std::fs::remove_dir_all(dir).unwrap();
            seen
        }

        let base = loaded("");
        for (property, expected) in [
            ("size.buffer.colortex0=128 128", "render resources"),
            ("bufferObject.0=16", "render resources"),
            ("shadow.enabled=true", "render settings"),
            ("uniform.float.fzAudit=1.0", "custom definitions"),
        ] {
            let changed = loaded(property);
            let outcome = judge(Ok(base.clone()), Ok(changed), &request(&[]));
            assert_eq!(
                outcome.failures,
                [Failure::new("determinism", expected)],
                "{property}"
            );
        }
    }

    #[test]
    fn an_invalid_module_fails_validation_by_program() {
        // A uniform without a binding.
        let mut module = naga::Module::default();
        let ty = module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Scalar(naga::Scalar::F32),
            },
            naga::Span::UNDEFINED,
        );
        module.global_variables.append(
            naga::GlobalVariable {
                name: Some("x".into()),
                space: naga::AddressSpace::Uniform,
                binding: None,
                ty,
                init: None,
                memory_decorations: Default::default(),
            },
            naga::Span::UNDEFINED,
        );
        let error = validate(&module).expect("a uniform needs a binding");
        assert_eq!(validate(&naga::Module::default()), None);
        let mut bad = seen(&["composite"], &[]);
        bad.invalid
            .push(("pass composite fragment".into(), error.clone()));
        let out = judge(Ok(bad.clone()), Ok(bad), &request(&[]));
        assert_eq!(
            out.failures,
            [Failure::new(
                "naga",
                format!("pass composite fragment: {}", shape(&error))
            )]
        );
    }

    #[test]
    fn validation_errors_on_the_second_load_are_still_naga_failures() {
        let mut bad = seen(&["composite"], &[]);
        bad.invalid
            .push(("pass composite fragment".into(), "invalid".into()));
        let good = seen(&["composite"], &[]);
        let second = judge(Ok(good.clone()), Ok(bad.clone()), &request(&[]));
        let first = judge(Ok(bad), Ok(good), &request(&[]));
        assert_eq!(first.failures, second.failures);
        assert_eq!(
            first.failures,
            [Failure::new("naga", "pass composite fragment: invalid")]
        );
    }

    #[test]
    fn custom_definitions_are_compared_as_loaded_state() {
        let macros = Macros::default();
        let first_properties =
            sunrise_shaders::Properties::parse("uniform.float.answer = 1.0\n", &macros);
        let second_properties =
            sunrise_shaders::Properties::parse("uniform.float.answer = 2.0\n", &macros);
        let mut first = seen(&["composite"], &[]);
        first.custom = sunrise_shaders::Custom::parse(&first_properties);
        let mut second = seen(&["composite"], &[]);
        second.custom = sunrise_shaders::Custom::parse(&second_properties);
        let out = judge(Ok(first), Ok(second), &request(&[]));
        assert_eq!(
            out.failures,
            [Failure::new("determinism", "custom definitions")]
        );
    }

    #[test]
    fn numbers_do_not_split_groups() {
        assert_eq!(shape("Function [12] at 3:4"), "Function [#] at #:#");
    }
}
