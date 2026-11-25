//! One GLSL program (a vertex and a fragment shader, as a GL program links them) and a
//! description of what its pipeline gives it ([`Inputs`]), as two validated naga modules. The
//! crate knows no game: which blocks and samplers exist, where includes are found and what each
//! define means are the caller's.
//!
//! The GLSL (`#include`, or another directive the caller names that pastes a file once, with
//! explicit locations or without) is written again as the Vulkan-flavoured GLSL 450 naga reads:
//! - GLSL a GL compiler takes and naga doesn't is rewritten first ([`glsl_fixups`]: a helper
//!   taking the `sampler2D` it reads, an array sized by its initializer only);
//! - includes are inlined ([`Includes`]), under the defines the program is compiled with;
//! - the uniform blocks get the bindings of [`Inputs::blocks`], and each sampler of
//!   [`Inputs::samplers`] is split into a texture and a sampler object, combined again where the
//!   sampler is named;
//! - the vertex inputs become globals the caller's [`Glue`] fills before the program's `main`
//!   (renamed `hz_main`) runs, and `gl_VertexID` and `gl_VertexIndex` read the glue's expression;
//!   the glue's [`Glue::fragment_finish`] runs after the fragment shader's `main`, renamed too;
//! - varyings get locations by name across the stages, the fragment output location 0;
//! - an integer texel buffer (`uniform isamplerBuffer Faces;`) becomes a read-only storage buffer
//!   of `int`s (`uint`s for `usamplerBuffer`), and `texelFetch` on it reads one into the first
//!   channel of an `ivec4`: wgpu has no texel buffers. Like any block the caller gives beyond
//!   its own, it is bound at [`Translated::other_binding`], and refused when not given.
//!
//! Each line keeps where it came from through every rewrite ([`Expanded`]), so an error names the
//! file (the stage's or an include's id) and the line the author wrote ([`Error`]).
//!
//! A caller that writes the stage as naga reads it itself (`sunrise-shaders`, for Iris/OptiFine
//! programs) compiles it through [`compile_stage`] instead, its own resources moved through a
//! [`BindingMap`].

use std::collections::HashSet;

use naga::ShaderStage;
use naga::front::glsl::{Frontend, InterfaceLimits, Options};
use naga::valid::{Capabilities, ValidationFlags, Validator};

mod glsl_fixups;
mod located;
mod stage;

pub use located::{Error, Expanded, Location, describe};
use located::{Lines, Origin};
pub use stage::{BindingMap, Source, compile_stage};

/// What a program's pipeline gives it, beyond its [`Glue`].
pub struct Inputs<'a> {
    /// Uniform blocks by name, with their binding in set 0.
    pub blocks: &'a [(&'a str, u32)],
    /// Samplers by name: the texture's binding in set 0, the sampler object's the next one.
    pub samplers: &'a [(&'a str, u32)],
    /// Loose `uniform T name;` declarations dropped where they stand, as a block already declares
    /// the name (an older version of the block kept in an `#else` branch); any other loose
    /// uniform is an error.
    pub loose: &'a [&'a str],
    /// The defines the program is compiled with, in order: `NAME` or `NAME value`.
    pub defines: &'a [&'a str],
    pub includes: &'a Includes<'a>,
}

/// Where a program's includes come from.
pub struct Includes<'a> {
    /// A file id's text (`ns:path`), none when nobody has it, or why it can't be read, which
    /// refuses the program.
    pub read: &'a dyn Fn(&str) -> Result<Option<String>, String>,
    /// The namespace of an `<path>` include that names none.
    pub namespace: &'a str,
    /// The folder `<ns:path>` is under, ending in `/`; `"relative"` is from the including file's.
    pub folder: &'a str,
    /// Directives besides `#include` that paste a file, each file once per stage (`#include`
    /// pastes it every time, the file's own guards keeping one copy).
    pub once: &'a [&'a str],
}

/// What the caller's pipeline gives the program's vertex shader, and what it runs after the
/// fragment shader's.
pub struct Glue<'a> {
    /// The inputs the program's vertex shader may declare: name and GLSL type.
    pub inputs: &'a [(&'a str, &'a str)],
    /// Declarations before the program's code: the real attributes, helpers.
    pub prelude: &'a str,
    /// Statements before the program's `main`: `inputs` filled from the attributes.
    pub setup: &'a str,
    /// Statements after it, on `gl_Position`.
    pub finish: &'a str,
    /// The expression the program's `gl_VertexID` and `gl_VertexIndex` read.
    pub vertex_id: &'a str,
    /// Statements after the fragment shader's `main` (renamed `hz_main`, as the vertex
    /// shader's is), on its outputs, once `main` returns; a `discard` ends the invocation
    /// before them. None leaves the fragment shader as the program wrote it. Only [`translate`] runs it: a caller of [`compile_stage`]
    /// writes its own `main`.
    pub fragment_finish: Option<&'a str>,
    /// The stage interface limits the stages are translated with (a device's, or the lowest
    /// any device has when checking without one).
    pub interface: InterfaceLimits,
}

/// A pipeline's two stages, validated.
#[derive(Debug)]
pub struct Translated {
    pub vertex: naga::Module,
    pub fragment: naga::Module,
    /// The blocks and samplers that are none of [`Inputs::blocks`] and [`Inputs::samplers`], in
    /// binding order: the `i`th is at [`Translated::other_binding`]`(i)` (a sampler's object one
    /// after its texture; a block leaves that binding unused).
    pub other: Vec<String>,
    /// The resources of set 0 the stages' entry points read, each with those stages
    /// ([`bindings_read`], from the validation that made the program).
    pub read: Vec<(u32, wgpu_types::ShaderStages)>,
    /// First set-0 binding assigned to [`Translated::other`], strictly above every binding in
    /// [`Inputs::blocks`] and [`Inputs::samplers`].
    other_base: u32,
    /// The code each stage was parsed from, with where its lines came from, and where the
    /// fragment stage declares each output location (a byte of its code): what
    /// [`Translated::declared_at`] and [`Translated::output_at`] read. None for a program made
    /// from modules alone ([`Translated::new`]).
    sources: Option<Box<Sources>>,
}

#[derive(Debug)]
struct Sources {
    code: [Expanded; 2],
    outputs: Vec<(u32, usize)>,
}

/// Where the first of `stages` (each a module and the code it was parsed from) that declares
/// the resource at `binding` of set 0 declares it, preferring one whose entry point reads it
/// (`reading`, in the stages' order); the first stage's file with no line when none declares
/// it.
pub fn declared_at(
    stages: [(&naga::Module, &Expanded); 2],
    reading: [bool; 2],
    binding: u32,
) -> Location {
    let declaration = |(module, code): (&naga::Module, &Expanded)| {
        (module.global_variables.iter())
            .find(|(_, g)| {
                g.binding
                    .as_ref()
                    .is_some_and(|b| b.group == 0 && b.binding == binding)
            })
            .map(|(handle, _)| code.location_in(module.global_variables.get_span(handle)))
    };
    let order = [0, 1].into_iter().filter(|&i| reading[i]).chain([0, 1]);
    order
        .filter_map(|i| declaration(stages[i]))
        .next()
        .unwrap_or_else(|| Location {
            file: stages[0].1.file().to_owned(),
            line: 0,
        })
}

/// The integer texel buffers a stage declares (`uniform isamplerBuffer` or `usamplerBuffer`).
fn texel_buffers(text: &str) -> HashSet<&str> {
    text.lines()
        .flat_map(declaration_lines)
        .filter_map(declaration)
        .filter(|d| d.storage == "uniform" && matches!(d.ty, "isamplerBuffer" | "usamplerBuffer"))
        .map(|d| d.name)
        .collect()
}

/// `texelFetch(Buffer, i)` on each integer texel buffer as `hz_fetch_Buffer(i)`, which the
/// buffer's declaration defines (see `rewrite`); `texelFetch` on anything else is left alone.
/// One pass over the text, however many buffers it declares: each `texelFetch` is looked at
/// once, its first argument's name looked up among them.
fn texel_fetches(lines: &mut Lines) {
    let text = lines.text();
    let buffers = texel_buffers(text);
    if buffers.is_empty() {
        return;
    }
    let mut edits = Vec::new();
    let mut from = 0;
    while let Some(found) = text[from..].find("texelFetch") {
        let at = from + found;
        let after = at + "texelFetch".len();
        from = after;
        // A name only ending in `texelFetch` is not it; one only starting with it has no `(`
        // next, so it isn't rewritten either.
        let whole = !text[..at].chars().next_back().is_some_and(is_ident);
        let args = text[after..]
            .trim_start()
            .strip_prefix('(')
            .map(str::trim_start);
        let Some(args) = args.filter(|_| whole) else {
            continue;
        };
        let name = &args[..args.find(|c: char| !is_ident(c)).unwrap_or(args.len())];
        if !buffers.contains(name) {
            continue;
        }
        if let Some(tail) = args[name.len()..].trim_start().strip_prefix(',') {
            let end = text.len() - tail.len();
            edits.push((at..end, format!("hz_fetch_{name}(")));
            from = end;
        }
    }
    lines.edit(edits);
}

impl Translated {
    /// The program of two stages and their validation, with what they read
    /// ([`Translated::read`]).
    pub fn new(
        (vertex, vertex_info): (naga::Module, naga::valid::ModuleInfo),
        (fragment, fragment_info): (naga::Module, naga::valid::ModuleInfo),
        other: Vec<String>,
        other_base: u32,
    ) -> Self {
        let read = bindings_read([(&vertex, &vertex_info), (&fragment, &fragment_info)]);
        Self {
            vertex,
            fragment,
            other,
            read,
            other_base,
            sources: None,
        }
    }

    /// Where a stage declares the resource at `binding` of set 0, a stage reading it first
    /// ([`declared_at`]): what a refusal of that declaration names. Nowhere known for a program
    /// made from modules alone.
    pub fn declared_at(&self, binding: u32) -> Location {
        let Some(sources) = &self.sources else {
            return Location::default();
        };
        let stages = self.stages_reading(binding);
        declared_at(
            [
                (&self.vertex, &sources.code[0]),
                (&self.fragment, &sources.code[1]),
            ],
            [
                stages.contains(wgpu_types::ShaderStages::VERTEX),
                stages.contains(wgpu_types::ShaderStages::FRAGMENT),
            ],
            binding,
        )
    }

    /// Where the fragment stage declares its output at `location`; its file with no line when
    /// it declares none there (nowhere known for a program made from modules alone).
    pub fn output_at(&self, location: u32) -> Location {
        let Some(sources) = &self.sources else {
            return Location::default();
        };
        let at = sources.outputs.iter().find(|(l, _)| *l == location);
        sources.code[1].location_at(at.map(|(_, at)| *at))
    }

    /// The binding of the `index`th block or sampler of [`Translated::other`] in set 0.
    pub fn other_binding(&self, index: usize) -> u32 {
        self.other_base + 2 * index as u32
    }

    /// The stages whose entry points read the resource at `binding` of set 0, through any
    /// function they call: what a pipeline layout must make that binding visible to. Empty
    /// for one a stage only declares (or the preprocessor kept but nothing reads).
    pub fn stages_reading(&self, binding: u32) -> wgpu_types::ShaderStages {
        stages_reading(&self.read, binding)
    }

    /// Whether every stage that reads the block at `binding` of set 0 declares it as one float
    /// (`uniform Name { float value; };`), what an expression uniform holds. A stage that only
    /// declares it, whatever its shape, binds nothing of it.
    pub fn is_float_block(&self, binding: u32) -> bool {
        let stages = self.stages_reading(binding);
        [
            (&self.vertex, wgpu_types::ShaderStages::VERTEX),
            (&self.fragment, wgpu_types::ShaderStages::FRAGMENT),
        ]
        .into_iter()
        .filter(|(_, stage)| stages.contains(*stage))
        .map(|(module, _)| module)
        .all(|module| {
            module
                .global_variables
                .iter()
                .filter(|(_, g)| {
                    g.binding
                        .as_ref()
                        .is_some_and(|b| b.group == 0 && b.binding == binding)
                })
                .all(|(_, g)| match (g.space, &module.types[g.ty].inner) {
                    (naga::AddressSpace::Uniform, naga::TypeInner::Struct { members, .. }) => {
                        matches!(
                            members.as_slice(),
                            [member] if matches!(
                                module.types[member.ty].inner,
                                naga::TypeInner::Scalar(naga::Scalar::F32)
                            )
                        )
                    }
                    _ => false,
                })
        })
    }

    /// The locations the fragment stage writes.
    pub fn fragment_writes(&self) -> Vec<u32> {
        locations(&self.fragment, false)
    }

    pub fn reads_binding(&self, binding: u32) -> bool {
        [&self.vertex, &self.fragment].into_iter().any(|module| {
            let bound: HashSet<naga::Handle<naga::GlobalVariable>> = module
                .global_variables
                .iter()
                .filter(|(_, g)| {
                    g.binding
                        .as_ref()
                        .is_some_and(|b| b.group == 0 && b.binding == binding)
                })
                .map(|(h, _)| h)
                .collect();
            let reads = |f: &naga::Function| {
                f.expressions.iter().any(
                    |(_, e)| matches!(e, naga::Expression::GlobalVariable(h) if bound.contains(h)),
                )
            };
            module.functions.iter().any(|(_, f)| reads(f))
                || module.entry_points.iter().any(|e| reads(&e.function))
        })
    }
}

/// The resources of set 0 that the entry points of each stage read, through any function they
/// call (naga's per entry point use, from the stage's own validation), each with the stages
/// reading it.
pub fn bindings_read(
    stages: [(&naga::Module, &naga::valid::ModuleInfo); 2],
) -> Vec<(u32, wgpu_types::ShaderStages)> {
    let mut read: Vec<(u32, wgpu_types::ShaderStages)> = Vec::new();
    for ((module, info), stage) in stages.into_iter().zip([
        wgpu_types::ShaderStages::VERTEX,
        wgpu_types::ShaderStages::FRAGMENT,
    ]) {
        for (handle, global) in module.global_variables.iter() {
            let Some(binding) = global.binding.as_ref().filter(|b| b.group == 0) else {
                continue;
            };
            let reads =
                (0..module.entry_points.len()).any(|i| !info.get_entry_point(i)[handle].is_empty());
            if !reads {
                continue;
            }
            match read.iter_mut().find(|(b, _)| *b == binding.binding) {
                Some((_, stages)) => *stages |= stage,
                None => read.push((binding.binding, stage)),
            }
        }
    }
    read
}

/// The stages in `read` ([`bindings_read`]) that read the resource at `binding`.
pub fn stages_reading(
    read: &[(u32, wgpu_types::ShaderStages)],
    binding: u32,
) -> wgpu_types::ShaderStages {
    (read.iter())
        .find(|(b, _)| *b == binding)
        .map_or(wgpu_types::ShaderStages::NONE, |(_, stages)| *stages)
}

/// A program's two stages, their file ids (`ns:path`, which errors in them name), and the folder
/// they are in (`ns:path/`), which their quoted (relative) includes start from.
pub struct Stages<'a> {
    pub dir: &'a str,
    pub vertex: &'a str,
    pub fragment: &'a str,
    pub vertex_file: &'a str,
    pub fragment_file: &'a str,
}

/// `error` raised in the `stage` shader, saying so as the messages always have.
fn in_stage(stage: ShaderStage, mut error: Error) -> Error {
    let name = if stage == ShaderStage::Vertex {
        "vertex"
    } else {
        "fragment"
    };
    error.message = format!("{name} shader: {}", error.message);
    error
}

/// The program of `sources` under `inputs`, its vertex stage fed by `glue`. Blocks and samplers
/// beyond [`Inputs::blocks`] and [`Inputs::samplers`] are bound at [`Translated::other`]'s
/// bindings when `provided` says yes to their name (a block's and a sampler's alike: the caller
/// knows which kind each one is), and are an error otherwise. Errors name the file and line the
/// author wrote ([`Error`]).
pub fn translate(
    sources: &Stages,
    inputs: &Inputs,
    glue: &Glue,
    provided: &dyn Fn(&str) -> bool,
) -> Result<Translated, Vec<Error>> {
    let Stages {
        dir,
        vertex,
        fragment,
        vertex_file,
        fragment_file,
    } = *sources;
    let stage = |text: &str, file: &str| -> Result<Expanded, Vec<Error>> {
        let mut expanded = expand_includes(text, file, dir, inputs.includes, Some(inputs.defines))
            .map_err(|e| vec![Error::from(e)])?;
        expanded.lines = glsl_fixups::fixups(expanded.lines);
        texel_fetches(&mut expanded.lines);
        Ok(expanded)
    };
    let (vertex_text, fragment_text) =
        (stage(vertex, vertex_file)?, stage(fragment, fragment_file)?);
    let base = other_binding_base(inputs).map_err(|message| {
        vec![Error {
            file: vertex_file.to_owned(),
            line: 0,
            message,
        }]
    })?;
    let mut names = Names::new(base);
    let rewritten = |text: &Expanded, stage, names: &mut Names| {
        rewrite(text, stage, inputs, glue, names).map_err(|e| vec![in_stage(stage, e)])
    };
    let vertex_code = rewritten(&vertex_text, ShaderStage::Vertex, &mut names)?;
    let fragment_code = rewritten(&fragment_text, ShaderStage::Fragment, &mut names)?;
    let compiled = |code: &Expanded, stage| {
        compile(
            stage,
            code,
            inputs.defines,
            &names,
            provided,
            glue.interface,
        )
        .map_err(|errors: Vec<Error>| {
            errors
                .into_iter()
                .map(|e| in_stage(stage, e))
                .collect::<Vec<_>>()
        })
    };
    let vertex = compiled(&vertex_code, ShaderStage::Vertex)?;
    let fragment: (naga::Module, naga::valid::ModuleInfo) =
        compiled(&fragment_code, ShaderStage::Fragment)?;
    let given = locations(&vertex.0, false);
    if let Some(missing) = locations(&fragment.0, true)
        .into_iter()
        .find(|l| !given.contains(l))
    {
        // At the fragment shader's input of that location.
        let at = names.inputs.iter().find(|(l, _)| *l == missing);
        return Err(vec![fragment_code.error_at(
            at.map(|(_, at)| *at),
            format!(
                "the fragment shader reads location {missing}, which the vertex shader doesn't write"
            ),
        )]);
    }
    let mut program = Translated::new(vertex, fragment, names.other, names.other_base);
    program.sources = Some(Box::new(Sources {
        code: [vertex_code, fragment_code],
        outputs: names.outputs,
    }));
    Ok(program)
}

/// A bound on a program's includes, which are pasted every time.
const MAX_PASTES: u32 = 1024;
pub const MAX_EXPANDED: usize = 1 << 20;
/// A bound on the bytes the expansion scans (each pasted file, every time), however little of
/// them it keeps: lines of imports already pasted add nothing to the shader.
const MAX_SCANNED: usize = 8 * MAX_EXPANDED;

/// Why a shader's includes refuse it ([`expand_includes`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImportRefusal {
    /// A target that is neither `<ns:path>` nor a quoted path from the including file's folder.
    Bad(String),
    Directive(String),
    /// A file nobody has (its id).
    NotFound(String),
    /// A file that can't be read: why, naming it.
    Unreadable(String),
    /// Includes nested more than 16 deep.
    TooDeep,
    /// More than [`MAX_PASTES`] files pasted.
    TooManyPastes,
    /// The shader grown past [`MAX_EXPANDED`] bytes.
    TooLarge,
    /// More than [`MAX_SCANNED`] bytes scanned.
    TooMuchScanned,
}

impl std::fmt::Display for ImportRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Directive(why) => f.write_str(why),
            Self::Bad(target) => write!(f, "bad include {target}"),
            Self::NotFound(id) => write!(f, "{id} not found"),
            Self::Unreadable(why) => f.write_str(why),
            Self::TooDeep => f.write_str("includes nest more than 16 deep"),
            Self::TooManyPastes => write!(f, "includes paste more than {MAX_PASTES} files"),
            Self::TooLarge => write!(f, "includes grow the shader past {MAX_EXPANDED} bytes"),
            Self::TooMuchScanned => {
                write!(
                    f,
                    "includes make the translator read past {MAX_SCANNED} bytes"
                )
            }
        }
    }
}

/// An [`ImportRefusal`] at the file and line of the directive it stops at (line 0: the file as
/// a whole, as for an `#if` it leaves open).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportError {
    pub file: String,
    pub line: u32,
    pub refusal: ImportRefusal,
}

impl From<ImportError> for Error {
    fn from(e: ImportError) -> Self {
        Error {
            file: e.file,
            line: e.line,
            message: e.refusal.to_string(),
        }
    }
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Error::from(self.clone()).fmt(f)
    }
}

/// The file `file` (its id, `ns:path`) with its includes inlined (`#include <ns:path>`,
/// `#include "relative"` to the including file's folder, and the same with each of
/// [`Includes::once`]), `#version` and `#extension` lines dropped. An `#include` is pasted every
/// time, as a GL compiler's includes are (the files' `#ifndef` guards keep one copy); a
/// once-directive pastes a file once per stage. With `defines`, only the includes the compiler
/// reaches under them are resolved, and macros an included file defines govern the rest of its
/// caller; without, every branch's are. The expansion is bounded (16 levels, [`MAX_PASTES`]
/// pastes, [`MAX_EXPANDED`] bytes kept, [`MAX_SCANNED`] bytes scanned). Each line of the result
/// keeps the file and line it came from ([`Expanded`]), and a refusal names the directive's.
pub fn expand_includes(
    text: &str,
    file: &str,
    dir: &str,
    includes: &Includes,
    defines: Option<&[&str]>,
) -> Result<Expanded, ImportError> {
    expand_with(
        text,
        file,
        dir,
        includes,
        defines.map(import_conditions::Conditions::new),
    )
}

/// naga's options for one `stage` of a program with its `defines` (`NAME` or `NAME value`, in
/// order) and the device's stage interface `limits`.
pub fn glsl_options(stage: ShaderStage, defines: &[&str], limits: InterfaceLimits) -> Options {
    let mut options = Options {
        interface_limits: limits,
        ..Options::from(stage)
    };
    for define in defines {
        let (name, value) = define.split_once(' ').unwrap_or((define, ""));
        options.defines.insert(name.to_owned(), value.to_owned());
    }
    options
}

mod import_conditions;

fn expand_with(
    text: &str,
    file: &str,
    dir: &str,
    includes: &Includes,
    conditions: Option<import_conditions::Conditions>,
) -> Result<Expanded, ImportError> {
    struct Walk<'a> {
        includes: &'a Includes<'a>,
        conditions: Option<import_conditions::Conditions>,
        seen: HashSet<String>,
        pastes: u32,
        scanned: usize,
        out: Lines,
        /// The ids of the files read, the stage's first, and each one's index.
        files: Vec<String>,
        index: std::collections::HashMap<String, u32>,
    }
    impl Walk<'_> {
        /// The index of the file `id`, added the first time.
        fn file(&mut self, id: &str) -> u32 {
            if let Some(&index) = self.index.get(id) {
                return index;
            }
            let index = u32::try_from(self.files.len()).unwrap_or(u32::MAX);
            self.files.push(id.to_owned());
            self.index.insert(id.to_owned(), index);
            index
        }

        fn refused(&self, file: u32, line: u32, refusal: ImportRefusal) -> ImportError {
            ImportError {
                file: self.files[file as usize].clone(),
                line,
                refusal,
            }
        }

        /// Counts `text` against [`MAX_SCANNED`].
        fn scan(&mut self, text: &str) -> Result<(), ImportRefusal> {
            self.scanned += text.len();
            if self.scanned > MAX_SCANNED {
                return Err(ImportRefusal::TooMuchScanned);
            }
            Ok(())
        }

        fn walk(
            &mut self,
            text: &str,
            file: u32,
            dir: &str,
            depth: u32,
        ) -> Result<(), ImportError> {
            let clean;
            let mut physical = Vec::new();
            let text = if self.conditions.is_some() {
                // Splicing precedes comment removal. Remember the first physical line of
                // each logical line, including in continued directives and line comments.
                let mut joined = String::with_capacity(text.len());
                physical.push(1u32);
                for (n, line) in text.split_inclusive('\n').enumerate() {
                    if let Some(part) = line
                        .strip_suffix("\\\r\n")
                        .or_else(|| line.strip_suffix("\\\n"))
                    {
                        joined.push_str(part);
                    } else {
                        joined.push_str(line);
                        physical.push(n as u32 + 2);
                    }
                }
                clean = strip_comments(&joined);
                &clean
            } else {
                text
            };
            let mut branches = import_conditions::Conditionals::default();
            let mut line_offset = 0i64;
            for (n, line) in text.lines().enumerate() {
                // The line the author wrote it on: the first of a continued line.
                let written = physical
                    .get(n)
                    .copied()
                    .unwrap_or_else(|| u32::try_from(n + 1).unwrap_or(u32::MAX));
                let origin = Some(Origin {
                    file,
                    line: written,
                });
                let directive = line.trim_start().strip_prefix('#').map(str::trim_start);
                let (word, rest) = directive.map_or(("", ""), |d| {
                    let i = d.find(|c| !is_ident(c)).unwrap_or(d.len());
                    (&d[..i], d[i..].trim())
                });
                if let Some(state) = &mut self.conditions {
                    // Whether the line is the preprocessor's alone (a branch's or `#line`).
                    let consumed: Result<bool, ImportRefusal> = 'directive: {
                        let location = u32::try_from(i64::from(physical[n]) + line_offset);
                        let Ok(location) = location else {
                            break 'directive Err(ImportRefusal::Directive(
                                "line number outside GLSL range".into(),
                            ));
                        };
                        match branches.apply_with(word, |kind| state.test(kind, rest, location)) {
                            Ok(skipped) if skipped || !branches.active() => {
                                break 'directive Ok(true);
                            }
                            Ok(_) => {}
                            Err(why) => break 'directive Err(ImportRefusal::Directive(why)),
                        }
                        match word {
                            "define" => state.define(rest),
                            "undef" => state.undef(rest),
                            "version" if depth == 0 => {
                                state.define(&format!(
                                    "__VERSION__ {}",
                                    rest.split_whitespace().next().unwrap_or("110")
                                ));
                            }
                            "line" => {
                                let next = match state.line(rest, location) {
                                    Ok(next) => next,
                                    Err(why) => {
                                        break 'directive Err(ImportRefusal::Directive(why));
                                    }
                                };
                                line_offset = i64::from(next)
                                    - i64::from(
                                        physical.get(n + 1).copied().unwrap_or(physical[n] + 1),
                                    );
                                break 'directive Ok(true);
                            }
                            "" if state.uses_line(line) => {
                                let directive = format!("#line {}\n", location.saturating_sub(1));
                                self.out.push(&directive, origin);
                            }
                            _ => {}
                        }
                        Ok(false)
                    };
                    match consumed {
                        Ok(true) => continue,
                        Ok(false) => {}
                        Err(refusal) => return Err(self.refused(file, written, refusal)),
                    }
                }
                match word {
                    _ if word == "include" || self.includes.once.contains(&word) => {
                        match self.include(word, rest, dir, depth) {
                            Ok(()) => {}
                            Err(Included::Here(refusal)) => {
                                return Err(self.refused(file, written, refusal));
                            }
                            Err(Included::Inside(error)) => return Err(error),
                        }
                    }
                    "version" | "extension" => {}
                    _ => self.out.push(line, origin),
                }
                self.out.push("\n", origin);
                if self.out.text().len() > MAX_EXPANDED {
                    return Err(self.refused(file, written, ImportRefusal::TooLarge));
                }
            }
            if branches.open() {
                return Err(self.refused(
                    file,
                    0,
                    ImportRefusal::Directive("#if without #endif".into()),
                ));
            }
            Ok(())
        }

        /// The file an include directive `word` names (`rest`) pasted, when it is to be.
        fn include(
            &mut self,
            word: &str,
            rest: &str,
            dir: &str,
            depth: u32,
        ) -> Result<(), Included> {
            let id = include_id(rest, dir, self.includes)
                .ok_or_else(|| Included::Here(ImportRefusal::Bad(rest.to_owned())))?;
            if !(self.seen.insert(id.clone()) || word == "include") {
                return Ok(());
            }
            self.pastes += 1;
            if self.pastes > MAX_PASTES {
                return Err(Included::Here(ImportRefusal::TooManyPastes));
            }
            let text = (self.includes.read)(&id)
                .map_err(|why| Included::Here(ImportRefusal::Unreadable(why)))?
                .ok_or_else(|| Included::Here(ImportRefusal::NotFound(id.clone())))?;
            if depth + 1 > 16 {
                return Err(Included::Here(ImportRefusal::TooDeep));
            }
            self.scan(&text).map_err(Included::Here)?;
            let parent = &id[..id.rfind('/').map_or(id.len(), |i| i + 1)];
            let file = self.file(&id);
            self.walk(&text, file, parent, depth + 1)
                .map_err(Included::Inside)
        }
    }
    /// Why an include isn't pasted: at its directive, or inside the file.
    enum Included {
        Here(ImportRefusal),
        Inside(ImportError),
    }
    let mut walk = Walk {
        includes,
        conditions,
        seen: HashSet::new(),
        pastes: 0,
        scanned: 0,
        out: Lines::default(),
        files: Vec::new(),
        index: std::collections::HashMap::new(),
    };
    let top = walk.file(file);
    walk.scan(text)
        .map_err(|refusal| walk.refused(top, 0, refusal))?;
    walk.walk(text, top, dir, 0)?;
    Ok(Expanded {
        lines: walk.out,
        files: walk.files,
    })
}

/// `<ns:path>` under [`Includes::folder`] ([`Includes::namespace`] when there's no namespace), or
/// `"path"` from the including file's folder.
fn include_id(target: &str, dir: &str, includes: &Includes) -> Option<String> {
    if let Some(inner) = target.strip_prefix('<').and_then(|t| t.strip_suffix('>')) {
        let (ns, path) = inner.split_once(':').unwrap_or((includes.namespace, inner));
        return Some(format!("{ns}:{}{path}", includes.folder));
    }
    let relative = target.strip_prefix('"')?.strip_suffix('"')?;
    let (ns, base) = dir.split_once(':')?;
    let mut parts: Vec<&str> = base.split('/').filter(|p| !p.is_empty()).collect();
    for part in relative.split('/') {
        match part {
            "." | "" => {}
            ".." => {
                parts.pop()?;
            }
            _ => parts.push(part),
        }
    }
    Some(format!("{ns}:{}", parts.join("/")))
}

/// Comments as spaces, line breaks kept.
fn strip_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    // Keep comment bytes as spaces: a pasted file still counts toward the stage budget.
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, chars.peek()) {
            ('/', Some('/')) => {
                out.push(' ');
                while let Some(&c) = chars.peek() {
                    if c == '\n' {
                        break;
                    }
                    out.extend(std::iter::repeat_n(' ', c.len_utf8()));
                    chars.next();
                }
            }
            ('/', Some('*')) => {
                chars.next();
                out.push_str("  ");
                let mut last = ' ';
                for c in chars.by_ref() {
                    if c == '\n' {
                        out.push('\n');
                    } else {
                        out.extend(std::iter::repeat_n(' ', c.len_utf8()));
                    }
                    if last == '*' && c == '/' {
                        break;
                    }
                    last = c;
                }
            }
            _ => out.push(c),
        }
    }
    out
}

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// `text` with each whole identifier `from` written as `to`.
fn replace_ident(text: &str, from: &str, to: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(from) {
        let before = rest[..at].chars().next_back();
        let after = rest[at + from.len()..].chars().next();
        out.push_str(&rest[..at]);
        if before.is_some_and(is_ident) || after.is_some_and(is_ident) {
            out.push_str(from);
        } else {
            out.push_str(to);
        }
        rest = &rest[at + from.len()..];
    }
    out.push_str(rest);
    out
}

/// A global declaration on one line: `layout(...) qualifiers storage type name;`.
struct Decl<'a> {
    /// `layout(location = N)`'s N.
    location: Option<u32>,
    qualifiers: Vec<&'a str>,
    storage: &'a str,
    ty: &'a str,
    name: &'a str,
}

fn declaration(line: &str) -> Option<Decl<'_>> {
    let mut rest = line.trim();
    let mut location = None;
    if let Some(after) = rest.strip_prefix("layout") {
        let after = after.trim_start().strip_prefix('(')?;
        let close = after.find(')')?;
        location = after[..close].split(',').find_map(|q| {
            let (key, value) = q.split_once('=')?;
            (key.trim() == "location").then(|| value.trim().parse().ok())?
        });
        rest = after[close + 1..].trim_start();
    }
    let body = rest.strip_suffix(';')?.trim_end();
    let words: Vec<&str> = body.split_whitespace().collect();
    let storage_at = words
        .iter()
        .position(|w| matches!(*w, "in" | "out" | "uniform"))?;
    let [ty, name] = words[storage_at + 1..] else {
        return None;
    };
    if !name.chars().all(is_ident) || !ty.chars().all(is_ident) {
        return None;
    }
    Some(Decl {
        location,
        qualifiers: words[..storage_at].to_vec(),
        storage: words[storage_at],
        ty,
        name,
    })
}

/// GLSL pack shaders are often minified, with several global declarations on one line.
/// Keep the line-oriented rewriter simple by separating only declaration statements; ordinary
/// statements and function bodies remain untouched.
fn declaration_lines(line: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    while let Some(relative) = line[start..].find(';') {
        let end = start + relative + 1;
        if declaration(&line[start..end]).is_some() {
            lines.push(&line[start..end]);
            start = end;
        } else {
            break;
        }
    }
    if start == 0 {
        vec![line]
    } else {
        if start < line.len() {
            lines.push(&line[start..]);
        }
        lines
    }
}

/// The block a `uniform Name {` line opens, with the line's text before `uniform`.
fn block_name(line: &str) -> Option<&str> {
    let at = line.find("uniform")?;
    let head = line[..at].trim();
    if !(head.is_empty() || head.starts_with("layout")) {
        return None;
    }
    let rest = line[at + "uniform".len()..].trim();
    let name_end = rest.find(|c: char| !is_ident(c)).unwrap_or(rest.len());
    let (name, tail) = rest.split_at(name_end);
    // `{` alone, or the whole block on the line (`uniform Day { float daylight; };`).
    (!name.is_empty() && (tail.trim().is_empty() || tail.trim_start().starts_with('{')))
        .then_some(name)
}

/// First set-0 binding strictly above every caller-declared block (`b`) and sampler (`b`, `b + 1`).
fn other_binding_base(inputs: &Inputs) -> Result<u32, String> {
    let mut base = 0u32;
    for &(_, b) in inputs.blocks {
        let next = b
            .checked_add(1)
            .ok_or_else(|| "block binding index overflow".to_owned())?;
        base = base.max(next);
    }
    for &(_, b) in inputs.samplers {
        let next = b
            .checked_add(2)
            .ok_or_else(|| "sampler binding index overflow".to_owned())?;
        base = base.max(next);
    }
    Ok(base)
}

/// What the two stages share: the vertex stage's outputs and their locations, and the names of
/// the blocks and samplers of other pipelines in binding order.
struct Names {
    varyings: Vec<(String, u32)>,
    other: Vec<String>,
    other_base: u32,
    /// The fragment stage's inputs: each one's location and where the rewritten stage declares
    /// it (a byte of its text), which an error about the location names.
    inputs: Vec<(u32, usize)>,
    /// The fragment stage's outputs, the same way.
    outputs: Vec<(u32, usize)>,
}

impl Names {
    fn new(other_base: u32) -> Self {
        Self {
            varyings: Vec::new(),
            other: Vec::new(),
            other_base,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    fn location(&self, varying: &str) -> Option<u32> {
        self.varyings
            .iter()
            .find(|(v, _)| v == varying)
            .map(|(_, location)| *location)
    }

    /// A block or sampler of another pipeline: its binding (two for a sampler).
    fn other_binding(&mut self, name: &str) -> Result<u32, String> {
        let index = match self.other.iter().position(|n| n == name) {
            Some(index) => index,
            None => {
                self.other.push(name.to_owned());
                self.other.len() - 1
            }
        };
        let offset = u32::try_from(index)
            .ok()
            .and_then(|i| i.checked_mul(2))
            .ok_or_else(|| "other binding index overflow".to_owned())?;
        let binding = self
            .other_base
            .checked_add(offset)
            .ok_or_else(|| "other binding index overflow".to_owned())?;
        binding
            .checked_add(1)
            .ok_or_else(|| "other binding index overflow".to_owned())?;
        Ok(binding)
    }
}

/// One stage written again for naga (see the module's notes), each line keeping where it came
/// from; an error at the line it is about.
fn rewrite(
    source: &Expanded,
    stage: ShaderStage,
    inputs: &Inputs,
    glue: &Glue,
    names: &mut Names,
) -> Result<Expanded, Error> {
    let text = source.text();
    let vertex = stage == ShaderStage::Vertex;
    let mut out = Lines::uniform("#version 450 core\n", None);
    if vertex {
        out.push(glue.prelude, None);
        out.push("\n", None);
        for (name, ty) in glue.inputs {
            out.push(&format!("{ty} {name};\n"), None);
        }
    }
    // Where the program's own code starts: the glue above it is left as it is.
    let pack_start = out.text().len();
    let mut outputs: Vec<&str> = Vec::new();
    let explicit: Vec<u32> = if vertex {
        text.lines()
            .flat_map(declaration_lines)
            .filter_map(declaration)
            .filter(|d| d.storage == "out")
            .filter_map(|d| d.location)
            .collect()
    } else {
        Vec::new()
    };
    let mut start = 0;
    for (n, source_line) in text.lines().enumerate() {
        let origin = source.lines.origin(n);
        let here = start;
        start += text[start..]
            .find('\n')
            .map_or(text.len() - start, |n| n + 1);
        let at = |message: String| source.error_at(Some(here), message);
        let write = |out: &mut Lines, code: &str| out.push(code, origin);
        for line in declaration_lines(source_line) {
            if let Some(name) = block_name(line) {
                let binding = match inputs.blocks.iter().find(|(block, _)| *block == name) {
                    Some((_, b)) => *b,
                    None => names.other_binding(name).map_err(at)?,
                };
                let tail = line[line.find("uniform").unwrap_or(0)..].trim();
                write(
                    &mut out,
                    &format!("layout(std140, set = 0, binding = {binding}) {tail}\n"),
                );
                continue;
            }
            let Some(decl) = declaration(line) else {
                write(&mut out, line);
                write(&mut out, "\n");
                continue;
            };
            let qualifiers = decl.qualifiers.join(" ");
            match decl.storage {
                "uniform" if matches!(decl.ty, "isamplerBuffer" | "usamplerBuffer") => {
                    let binding = names.other_binding(decl.name).map_err(at)?;
                    let name = decl.name;
                    let (scalar, vector) = if decl.ty == "isamplerBuffer" {
                        ("int", "ivec4")
                    } else {
                        ("uint", "uvec4")
                    };
                    write(
                        &mut out,
                        &format!(
                            "layout(std430, set = 0, binding = {binding}) readonly buffer hz_buffer_{name} {{ {scalar} hz_data_{name}[]; }};\n\
                             {vector} hz_fetch_{name}(int i) {{ return {vector}(hz_data_{name}[i], 0, 0, 1); }}\n"
                        ),
                    );
                }
                "uniform" if decl.ty == "sampler2D" => {
                    let binding = match inputs
                        .samplers
                        .iter()
                        .find(|(sampler, _)| *sampler == decl.name)
                    {
                        Some((_, b)) => *b,
                        None => names.other_binding(decl.name).map_err(at)?,
                    };
                    let name = decl.name;
                    write(
                        &mut out,
                        &format!(
                            "layout(set = 0, binding = {binding}) uniform texture2D hz_tex_{name};\n\
                         layout(set = 0, binding = {}) uniform sampler hz_smp_{name};\n\
                         #define {name} sampler2D(hz_tex_{name}, hz_smp_{name})\n",
                            binding + 1
                        ),
                    );
                }
                "uniform" => {
                    // A loose declaration of a name a block declares (an older version kept in
                    // an `#else` branch): the preprocessor picks the branch later, but this pass
                    // sees both.
                    if inputs.loose.contains(&decl.name) {
                        write(&mut out, "\n");
                        continue;
                    }
                    return Err(at(format!("uniform {} outside a block", decl.name)));
                }
                "in" if vertex => {
                    if glue.inputs.iter().any(|(name, _)| *name == decl.name) {
                        // Declared by the glue as a global.
                        write(&mut out, "\n");
                    } else {
                        // Another format's input (`UV2` behind `#if !defined(IS_GUI)`): the
                        // preprocessor drops it when it is for another pipeline, else it reads 0.
                        write(&mut out, &format!("{} {};\n", decl.ty, decl.name));
                    }
                }
                "out" if vertex => {
                    // One location each: naga takes scalars and vectors only.
                    let location = decl
                        .location
                        .or_else(|| names.location(decl.name))
                        .unwrap_or_else(|| {
                            (0..)
                                .find(|l| {
                                    !explicit.contains(l)
                                        && names.varyings.iter().all(|(_, v)| v != l)
                                })
                                .unwrap_or_default()
                        });
                    let varying = (decl.name.to_owned(), location);
                    if !names.varyings.contains(&varying) {
                        names.varyings.push(varying);
                    }
                    write(
                        &mut out,
                        &format!(
                            "layout(location = {location}) {qualifiers} out {} {};\n",
                            decl.ty, decl.name
                        ),
                    );
                }
                "in" => {
                    let location = decl
                        .location
                        .or_else(|| names.location(decl.name))
                        .ok_or_else(|| {
                            at(format!(
                                "input {} isn't written by the vertex shader",
                                decl.name
                            ))
                        })?;
                    names.inputs.push((location, out.text().len()));
                    write(
                        &mut out,
                        &format!(
                            "layout(location = {location}) {qualifiers} in {} {};\n",
                            decl.ty, decl.name
                        ),
                    );
                }
                _ => {
                    // Pipelines with more targets (OIT's, behind `#ifdef OIT`) name them; the
                    // colour this pipeline draws is the one at location 0.
                    let location = decl.location.unwrap_or_else(|| {
                        let at = outputs.iter().position(|o| *o == decl.name);
                        at.unwrap_or_else(|| {
                            outputs.push(decl.name);
                            outputs.len() - 1
                        }) as u32
                    });
                    names.outputs.push((location, out.text().len()));
                    write(
                        &mut out,
                        &format!(
                            "layout(location = {location}) out {} {};\n",
                            decl.ty, decl.name
                        ),
                    );
                }
            }
        }
    }
    if let (false, Some(finish)) = (vertex, glue.fragment_finish) {
        let code = &out.text()[pack_start..];
        let edits = main_at(code)
            .map(|at| (at + pack_start..at + pack_start + 4, "hz_main".to_owned()))
            .collect();
        out.edit(edits);
        out.push(&format!("void main() {{\nhz_main();\n{finish}\n}}\n"), None);
    }
    if vertex {
        // The program's `gl_VertexID` and `gl_VertexIndex` both read the glue's expression (a
        // pipeline drawing six vertices per quad turns its own index back into the program's
        // four per quad). Only the program's code is rewritten, so the glue's expression keeps
        // reading the real index.
        let vertex_id = format!("({})", glue.vertex_id);
        let code = &out.text()[pack_start..];
        let mut edits: Vec<(std::ops::Range<usize>, String)> = main_at(code)
            .map(|at| (at..at + 4, "hz_main".to_owned()))
            .chain(ident_at(code, "gl_VertexIndex").map(|r| (r, vertex_id.clone())))
            .chain(ident_at(code, "gl_VertexID").map(|r| (r, vertex_id.clone())))
            .map(|(r, with)| (r.start + pack_start..r.end + pack_start, with))
            .collect();
        edits.sort_by_key(|(r, _)| r.start);
        out.edit(edits);
        out.push(
            &format!(
                "void main() {{\n{}\nhz_main();\n{}\n}}\n",
                glue.setup, glue.finish
            ),
            None,
        );
    }
    Ok(Expanded {
        lines: out,
        files: source.files.clone(),
    })
}

/// Where `text` has `from` as a whole identifier.
fn ident_at<'a>(text: &'a str, from: &'a str) -> impl Iterator<Item = std::ops::Range<usize>> + 'a {
    text.match_indices(from)
        .map(|(at, _)| at..at + from.len())
        .filter(|r| {
            !text[..r.start].chars().next_back().is_some_and(is_ident)
                && !text[r.end..].chars().next().is_some_and(is_ident)
        })
}

/// Where the `main` of each `void main(` is, which becomes `hz_main`.
fn main_at(text: &str) -> impl Iterator<Item = usize> + '_ {
    ident_at(text, "main")
        .filter(|r| {
            text[..r.start].trim_end().ends_with("void")
                && text[r.end..].trim_start().starts_with('(')
        })
        .map(|r| r.start)
}

/// Parses and validates one stage; errors at the file and line each part of it came from.
fn compile(
    stage: ShaderStage,
    code: &Expanded,
    defines: &[&str],
    names: &Names,
    provided: &dyn Fn(&str) -> bool,
    limits: InterfaceLimits,
) -> Result<(naga::Module, naga::valid::ModuleInfo), Vec<Error>> {
    let options = glsl_options(stage, defines, limits);
    let text = code.text();
    let module = Frontend::default().parse(&options, text).map_err(|e| {
        // The line naga read, as the translator wrote it, each found among the line breaks.
        let breaks: Vec<usize> = text.match_indices('\n').map(|(at, _)| at).collect();
        code.errors_in(e.errors.iter().map(|e| {
            let start = e.meta.to_range().map_or(0, |r| r.start);
            let line = breaks.partition_point(|&b| b < start);
            let from = line.checked_sub(1).map_or(0, |l| breaks[l] + 1);
            let to = breaks.get(line).copied().unwrap_or(text.len());
            let source = text[from..to].trim();
            (e.meta, format!("({source}): {}", e.kind))
        }))
    })?;
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map_err(|e| {
            // The innermost span naga gives (an expression inside the function the error names).
            let span = (e.spans().map(|(s, _)| *s))
                .filter(|s| s.to_range().is_some())
                .last();
            let span = span.unwrap_or_default();
            vec![code.error_in(span, stage::validation_message(e.into_inner()))]
        })?;
    let entry = info.get_entry_point(0);
    for (handle, global) in module.global_variables.iter() {
        let Some(binding) = global
            .binding
            .as_ref()
            .filter(|b| b.group == 0)
            .map(|b| b.binding)
        else {
            continue;
        };
        if binding >= names.other_base && !entry[handle].is_empty() {
            // The name it was declared with (a block's instance may have none).
            let index = ((binding - names.other_base) / 2) as usize;
            let name = names.other.get(index).map_or("?", String::as_str);
            if provided(name) {
                continue;
            }
            return Err(vec![code.error_in(
                module.global_variables.get_span(handle),
                format!("{name} is not a block or sampler of this pipeline"),
            )]);
        }
    }
    Ok((module, info))
}

/// The locations the entry point reads (`inputs`) or writes.
fn locations(module: &naga::Module, inputs: bool) -> Vec<u32> {
    let mut out = Vec::new();
    let mut add = |ty: naga::Handle<naga::Type>, binding: Option<&naga::Binding>| {
        if let Some(naga::Binding::Location { location, .. }) = binding {
            out.push(*location);
        } else if let naga::TypeInner::Struct { members, .. } = &module.types[ty].inner {
            for m in members {
                if let Some(naga::Binding::Location { location, .. }) = &m.binding {
                    out.push(*location);
                }
            }
        }
    };
    for entry in &module.entry_points {
        if inputs {
            for arg in &entry.function.arguments {
                add(arg.ty, arg.binding.as_ref());
            }
        } else if let Some(result) = &entry.function.result {
            add(result.ty, result.binding.as_ref());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Blocks, samplers and loose uniforms in a game's shape: the tests' own.
    pub(crate) const BLOCKS: [(&str, u32); 4] = [
        ("Globals", 0),
        ("Projection", 1),
        ("DynamicTransforms", 2),
        ("Fog", 3),
    ];
    pub(crate) const SAMPLERS: [(&str, u32); 3] =
        [("Sampler0", 4), ("Sampler1", 6), ("Sampler2", 8)];
    const LOOSE: [&str; 3] = ["ProjMat", "ModelViewMat", "GameTime"];
    /// The lowest stage interface limits a device has.
    pub(crate) const LIMITS: InterfaceLimits = InterfaceLimits {
        inter_stage_variables: 32,
        vertex_attributes: 32,
    };

    /// The stages' files in the tests.
    pub(crate) const VSH_FILE: &str = "minecraft:shaders/core/test.vsh";
    pub(crate) const FSH_FILE: &str = "minecraft:shaders/core/test.fsh";

    fn includes<'a>(read: &'a dyn Fn(&str) -> Result<Option<String>, String>) -> Includes<'a> {
        Includes {
            read,
            namespace: "minecraft",
            folder: "shaders/include/",
            once: &["moj_import"],
        }
    }

    pub(crate) fn translate_core_at(
        sources: &Stages,
        defines: &[&str],
        read: &dyn Fn(&str) -> Result<Option<String>, String>,
        glue: &Glue,
        provided: &dyn Fn(&str) -> bool,
    ) -> Result<Translated, Vec<Error>> {
        let inputs = Inputs {
            blocks: &BLOCKS,
            samplers: &SAMPLERS,
            loose: &LOOSE,
            defines,
            includes: &includes(read),
        };
        translate(sources, &inputs, glue, provided)
    }

    pub(crate) fn translate_core_with(
        vertex: &str,
        fragment: &str,
        defines: &[&str],
        include: &dyn Fn(&str) -> Option<String>,
        glue: &Glue,
        provided: &dyn Fn(&str) -> bool,
    ) -> Result<Translated, Vec<Error>> {
        let sources = Stages {
            dir: "minecraft:shaders/core/",
            vertex,
            fragment,
            vertex_file: VSH_FILE,
            fragment_file: FSH_FILE,
        };
        translate_core_at(&sources, defines, &|id| Ok(include(id)), glue, provided)
    }

    pub(crate) fn translate_core(
        vertex: &str,
        fragment: &str,
        defines: &[&str],
        include: &dyn Fn(&str) -> Option<String>,
        glue: &Glue,
    ) -> Result<Translated, Vec<Error>> {
        translate_core_with(vertex, fragment, defines, include, glue, &|_| false)
    }

    fn expand_imports(
        text: &str,
        dir: &str,
        read: &dyn Fn(&str) -> Result<Option<String>, String>,
    ) -> Result<String, ImportRefusal> {
        expand_includes(text, VSH_FILE, dir, &includes(read), None)
            .map(|e| e.text().to_owned())
            .map_err(|e| e.refusal)
    }

    fn expand(
        text: &str,
        dir: &str,
        include: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, ImportRefusal> {
        expand_imports(text, dir, &|id| Ok(include(id)))
    }

    fn expand_core_imports(
        text: &str,
        dir: &str,
        defines: &[&str],
        read: &dyn Fn(&str) -> Result<Option<String>, String>,
    ) -> Result<String, ImportRefusal> {
        expand_includes(text, VSH_FILE, dir, &includes(read), Some(defines))
            .map(|e| e.text().to_owned())
            .map_err(|e| e.refusal)
    }

    const INCLUDES: [(&str, &str); 4] = [
        (
            "minecraft:shaders/include/dynamictransforms.glsl",
            "#ifndef DT\n#define DT\nlayout(std140) uniform DynamicTransforms {\n    mat4 ModelViewMat;\n    mat4 TextureMat;\n    vec4 ColorModulator;\n    vec3 ModelOffset;\n};\n#endif\n",
        ),
        (
            "minecraft:shaders/include/projection.glsl",
            "layout(std140) uniform Projection {\n    mat4 ProjMat;\n};\n",
        ),
        (
            "minecraft:shaders/include/globals.glsl",
            "layout(std140) uniform Globals {\n    ivec3 CameraBlockPos;\n    float GlintAlpha;\n    vec3 CameraOffset;\n    float GameTime;\n    vec2 ScreenSize;\n    int MenuBlurRadius;\n    int UseRgss;\n};\n",
        ),
        (
            "minecraft:shaders/include/fog.glsl",
            "layout(std140) uniform Fog {\n    vec4 FogColor;\n    float FogEnvironmentalStart;\n};\nvec4 apply_fog(vec4 c) { return mix(c, FogColor, FogEnvironmentalStart); }\n",
        ),
    ];

    fn include(id: &str) -> Option<String> {
        INCLUDES
            .iter()
            .find(|(path, _)| *path == id)
            .map(|(_, text)| (*text).to_owned())
    }

    const GLUE: Glue = Glue {
        inputs: &[("Position", "vec3"), ("Color", "vec4"), ("UV0", "vec2")],
        prelude: "layout(location = 0) in vec2 hz_pos;\nlayout(location = 1) in vec2 hz_uv;\nlayout(location = 2) in vec4 hz_color;\n\
                  const int hz_corner[6] = int[6](0, 3, 2, 0, 2, 1);\n\
                  int hz_vertex_id() { int i = int(gl_VertexIndex) - int(gl_InstanceIndex); return i / 6 * 4 + hz_corner[i % 6]; }",
        setup: "Position = vec3(hz_pos, 0.0); Color = hz_color; UV0 = hz_uv;",
        finish: "gl_Position.z = 0.5 * gl_Position.w;",
        vertex_id: "hz_vertex_id()",
        fragment_finish: None,
        interface: LIMITS,
    };

    /// 26.3's `text.vsh` and `text.fsh` in shape: `#include`, explicit locations, an extension.
    const VANILLA_VSH: &str = "#version 330\n#extension GL_ARB_separate_shader_objects : require\n\
        #if !defined(IS_GUI)\n#include <minecraft:fog.glsl>\n#endif\n\
        #include <minecraft:dynamictransforms.glsl>\n#include <minecraft:projection.glsl>\n\
        layout(location = 0) in vec3 Position;\nlayout(location = 1) in vec4 Color;\nlayout(location = 2) in vec2 UV0;\n\
        layout(location = 2) out vec4 vertexColor;\nlayout(location = 3) out vec2 texCoord0;\n\
        void main() {\n    gl_Position = ProjMat * ModelViewMat * vec4(Position, 1.0);\n    vertexColor = Color;\n    texCoord0 = UV0;\n}\n";
    const VANILLA_FSH: &str = "#version 330\n#extension GL_ARB_separate_shader_objects : require\n\
        #include <minecraft:dynamictransforms.glsl>\n\
        uniform sampler2D Sampler0;\n\
        layout(location = 2) in vec4 vertexColor;\nlayout(location = 3) in vec2 texCoord0;\n\
        layout(location = 0) out vec4 fragColor;\n\
        void main() {\n    vec4 color = texture(Sampler0, texCoord0) * vertexColor * ColorModulator;\n    if (color.a < 0.1) {\n        discard;\n    }\n    fragColor = color;\n}\n";

    #[test]
    fn vanilla_26_3_text_shaders_translate() {
        let program =
            translate_core(VANILLA_VSH, VANILLA_FSH, &["IS_GUI"], &include, &GLUE).unwrap();
        assert_eq!(program.vertex.entry_points.len(), 1);
        assert_eq!(program.fragment.entry_points.len(), 1);
    }

    /// Whether `function`'s body calls the function named `callee`.
    fn calls(module: &naga::Module, function: &naga::Function, callee: &str) -> bool {
        function.body.iter().any(|s| {
            matches!(s, naga::Statement::Call { function, .. }
                if module.functions[*function].name.as_deref() == Some(callee))
        })
    }

    /// The fragment stage's GLSL `main` (naga's entry point calls it).
    fn glsl_main(module: &naga::Module) -> &naga::Function {
        let (_, main) = (module.functions.iter())
            .find(|(_, f)| f.name.as_deref() == Some("main"))
            .expect("main");
        main
    }

    /// `fragment_finish` runs in the fragment stage's `main` after the program's `main`, renamed
    /// `hz_main` (whose `discard` stays its own).
    #[test]
    fn a_fragment_finish_runs_after_the_programs_main() {
        let glue = Glue {
            fragment_finish: Some("fragColor = fragColor.bgra;"),
            ..GLUE
        };
        let program = translate_core(VANILLA_VSH, VANILLA_FSH, &["IS_GUI"], &include, &glue)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let fragment = &program.fragment;
        let main = glsl_main(fragment);
        let body: Vec<&naga::Statement> = main.body.iter().collect();
        let call = (body.iter())
            .position(|s| {
                matches!(s, naga::Statement::Call { function, .. }
                if fragment.functions[*function].name.as_deref() == Some("hz_main"))
            })
            .unwrap_or_else(|| panic!("main calls hz_main: {:#?}", main.body));
        let bgra = [
            naga::SwizzleComponent::Z,
            naga::SwizzleComponent::Y,
            naga::SwizzleComponent::X,
            naga::SwizzleComponent::W,
        ];
        let swizzled = body[call..].iter().any(|s| {
            matches!(s, naga::Statement::Store { value, .. }
                if matches!(main.expressions[*value], naga::Expression::Swizzle { pattern, .. } if pattern == bgra))
        });
        assert!(swizzled, "{:#?}", main.body);
        let (_, program_main) = (fragment.functions.iter())
            .find(|(_, f)| f.name.as_deref() == Some("hz_main"))
            .expect("the program's main, renamed");
        assert!(
            (program_main.body.iter()).any(|s| matches!(s, naga::Statement::If { .. })),
            "the program's discard stays in its main"
        );
    }

    /// Without `fragment_finish` the fragment stage's `main` is the program's, as written.
    #[test]
    fn no_fragment_finish_leaves_the_fragment_main_as_written() {
        let program = translate_core(VANILLA_VSH, VANILLA_FSH, &["IS_GUI"], &include, &GLUE)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let fragment = &program.fragment;
        assert!(!calls(fragment, glsl_main(fragment), "hz_main"));
        assert!(
            (fragment.functions.iter()).all(|(_, f)| f.name.as_deref() != Some("hz_main")),
            "no hz_main"
        );
    }

    /// A 26.3 pack's `gl_VertexIndex` means vanilla's vertex number, four per quad. The text
    /// pipeline draws six per glyph, so the pack's code must read the glue's `hz_vertex_id()`
    /// (as `gl_VertexID` does), never the raw index.
    #[test]
    fn a_packs_gl_vertex_index_reads_the_glues_vertex_number() {
        let vsh = "#version 330\n#include <minecraft:dynamictransforms.glsl>\n#include <minecraft:projection.glsl>\n\
            layout(location = 0) in vec3 Position;\nlayout(location = 1) in vec4 Color;\nlayout(location = 2) in vec2 UV0;\n\
            layout(location = 2) out vec4 vertexColor;\nlayout(location = 3) out vec2 texCoord0;\n\
            void main() {\n    int corner = gl_VertexIndex % 4;\n\
                gl_Position = ProjMat * ModelViewMat * vec4(Position + vec3(float(corner), 0.0, 0.0), 1.0);\n\
                vertexColor = Color;\n    texCoord0 = UV0;\n}\n";
        let program = translate_core(vsh, VANILLA_FSH, &["IS_GUI"], &include, &GLUE).unwrap();
        let module = &program.vertex;
        let named = |name: &str| {
            module
                .functions
                .iter()
                .find(|(_, f)| f.name.as_deref() == Some(name))
                .unwrap_or_else(|| panic!("no {name}"))
        };
        let (vertex_id, _) = named("hz_vertex_id");
        let (_, pack_main) = named("hz_main");
        fn calls(block: &naga::Block, target: naga::Handle<naga::Function>) -> bool {
            block.iter().any(|statement| match statement {
                naga::Statement::Call { function, .. } => *function == target,
                naga::Statement::Block(inner) => calls(inner, target),
                naga::Statement::If { accept, reject, .. } => {
                    calls(accept, target) || calls(reject, target)
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => calls(body, target) || calls(continuing, target),
                _ => false,
            })
        }
        assert!(calls(&pack_main.body, vertex_id));
    }

    /// A 26.2-style pack shader as a HUD pack writes it: `#moj_import`, no locations, a flat
    /// varying, `gl_VertexID`, `texelFetch` and `textureSize` on the glyph atlas, `GameTime`.
    #[test]
    fn a_hud_pack_text_shader_translates() {
        let vsh = "#version 330\n\
            #moj_import <minecraft:dynamictransforms.glsl>\n#moj_import <minecraft:projection.glsl>\n\
            in vec3 Position;\nin vec4 Color;\nin vec2 UV0;\n\
            out vec4 vertexColor;\nout vec2 texCoord0;\n\
            #ifdef IS_GUI\n#moj_import <minecraft:globals.glsl>\nuniform sampler2D Sampler0; // the glyphs\nflat out vec4 effect;\n#endif\n\
            /* moved\n   glyphs */\n\
            void main() {\n    vec3 p = Position;\n#ifdef IS_GUI\n    int corner = gl_VertexID % 4;\n\
                ivec2 size = textureSize(Sampler0, 0);\n\
                if (ivec4(round(texelFetch(Sampler0, ivec2(UV0 * vec2(size)), 0) * 255.0)) == ivec4(177, 126, 91, 1)) {\n\
                    p.x += float(corner) + ScreenSize.x * GameTime;\n    }\n    effect = vec4(float((ivec3(Color.rgb * 255.0) >> 4) == ivec3(3, 14, 10)));\n#endif\n\
                gl_Position = ProjMat * ModelViewMat * vec4(p, 1.0);\n    vertexColor = Color;\n    texCoord0 = UV0;\n}\n";
        let fsh = "#version 330\n#moj_import <minecraft:dynamictransforms.glsl>\n#moj_import <minecraft:globals.glsl>\n\
            uniform sampler2D Sampler0;\nin vec4 vertexColor;\nin vec2 texCoord0;\nflat in vec4 effect;\nout vec4 fragColor;\n\
            void main() {\n    vec4 color = texture(Sampler0, texCoord0) * vertexColor;\n    if (effect.x > 0.5) {\n        fragColor = vertexColor * ColorModulator;\n        return;\n    }\n\
                if (color.a < 0.1) {\n        discard;\n    }\n    fragColor = color * sin(GameTime);\n}\n";
        translate_core(vsh, fsh, &["IS_GUI"], &include, &GLUE).unwrap();
    }

    #[test]
    fn varyings_meet_at_the_same_locations() {
        let vsh = "#version 330\nin vec3 Position;\nout float b;\nout vec2 m;\nout vec4 a;\n\
            void main() { gl_Position = vec4(Position, 1.0); b = 1.0; m = vec2(1.0); a = vec4(b); }\n";
        let fsh = "#version 330\nin vec4 a;\nin float b;\nout vec4 fragColor;\nvoid main() { fragColor = a * b; }\n";
        let program = translate_core(vsh, fsh, &[], &include, &GLUE).unwrap();
        let mut read = locations(&program.fragment, true);
        read.sort_unstable();
        assert_eq!(read, [0, 2]);
    }

    #[test]
    fn explicit_varying_locations_are_kept() {
        let vsh = "#version 330\nin vec3 Position;\n\
            #ifdef IS_GUI\nlayout(location = 4) out float gui;\n#else\nlayout(location = 4) out float world;\n#endif\n\
            layout(location = 13) out vec4 a;\nout float b;\n\
            void main() { gl_Position = vec4(Position, 1.0); a = vec4(1.0); b = 1.0;\n\
            #ifdef IS_GUI\ngui = 1.0;\n#else\nworld = 1.0;\n#endif\n}\n";
        let fsh = "#version 330\n#ifdef IS_GUI\nlayout(location = 4) in float gui;\n#endif\n\
            layout(location = 13) in vec4 a;\nin float b;\nout vec4 fragColor;\n\
            void main() { fragColor = a * b;\n#ifdef IS_GUI\nfragColor *= gui;\n#endif\n}\n";
        let program = translate_core(vsh, fsh, &["IS_GUI"], &include, &GLUE).unwrap();
        let mut read = locations(&program.fragment, true);
        read.sort_unstable();
        assert_eq!(read, [0, 4, 13]);
    }

    #[test]
    fn an_include_is_pasted_every_time_and_a_moj_import_once() {
        let file = |id: &str| {
            (id == "minecraft:shaders/include/g.glsl")
                .then(|| "#ifndef G\n#define G\nconst float g = 1.0;\n#endif\n".to_owned())
        };
        let twice =
            "#ifdef A\n#include <minecraft:g.glsl>\n#else\n#include <minecraft:g.glsl>\n#endif\n";
        let expanded = expand(twice, "minecraft:shaders/core/", &file).unwrap();
        assert_eq!(expanded.matches("const float g").count(), 2);
        let imported = expand(
            &twice.replace("#include", "#moj_import"),
            "minecraft:shaders/core/",
            &file,
        )
        .unwrap();
        assert_eq!(imported.matches("const float g").count(), 1);
    }

    #[test]
    fn a_location_another_branch_gave_the_same_name_isnt_reused() {
        let vsh = "#version 330\nin vec3 Position;\n\
            #ifdef IS_GUI\nlayout(location = 0) out vec4 vertexColor;\n\
            #else\nlayout(location = 1) out vec4 vertexColor;\nlayout(location = 0) out vec4 worldOther;\n#endif\n\
            void main() { gl_Position = vec4(Position, 1.0); vertexColor = vec4(1.0);\n\
            #ifndef IS_GUI\nworldOther = vec4(0.5);\n#endif\n}\n";
        let fsh = "#version 330\n\
            #ifdef IS_GUI\nlayout(location = 0) in vec4 vertexColor;\n\
            #else\nlayout(location = 1) in vec4 vertexColor;\nlayout(location = 0) in vec4 worldOther;\n#endif\n\
            out vec4 fragColor;\nvoid main() { fragColor = vertexColor;\n\
            #ifndef IS_GUI\nfragColor *= worldOther;\n#endif\n}\n";
        let world = translate_core(vsh, fsh, &[], &include, &GLUE).unwrap();
        let mut written = locations(&world.vertex, false);
        written.sort_unstable();
        assert_eq!(written, [0, 1]);
        let mut read = locations(&world.fragment, true);
        read.sort_unstable();
        assert_eq!(read, [0, 1]);
        let gui = translate_core(vsh, fsh, &["IS_GUI"], &include, &GLUE).unwrap();
        assert_eq!(locations(&gui.vertex, false), [0]);
    }

    #[test]
    fn includes_that_multiply_are_refused_once_over_budget() {
        let file = |id: &str| {
            let level: u32 = id
                .strip_prefix("minecraft:shaders/include/l")?
                .strip_suffix(".glsl")?
                .parse()
                .ok()?;
            let next = format!("#include <minecraft:l{}.glsl>\n", level + 1).repeat(4);
            Some(format!(
                "#ifndef L{level}\n#define L{level}\n{}#endif\n",
                if level < 15 { next.as_str() } else { "" }
            ))
        };
        let error = expand(
            "#include <minecraft:l0.glsl>\n",
            "minecraft:shaders/core/",
            &file,
        )
        .unwrap_err();
        assert_eq!(error, ImportRefusal::TooManyPastes);
    }

    /// The bytes scanned are bounded too, not only those kept: a file of imports already pasted
    /// keeps a line break each, and is still read in full every time it is included. Seven
    /// pastes of 1.05 MB fit [`MAX_SCANNED`] (8 MiB), the eighth is refused.
    #[test]
    fn includes_that_keep_little_are_refused_once_scanned_past_the_budget() {
        let seen = "#moj_import <b.glsl>\n".repeat(50_000);
        let file = |id: &str| match id {
            "minecraft:shaders/include/a.glsl" => Some(seen.clone()),
            "minecraft:shaders/include/b.glsl" => Some(String::new()),
            _ => None,
        };
        let main = |n: usize| format!("#moj_import <b.glsl>\n{}", "#include <a.glsl>\n".repeat(n));
        let dir = "minecraft:shaders/core/";
        assert!(expand(&main(7), dir, &file).is_ok());
        assert_eq!(
            expand(&main(8), dir, &file),
            Err(ImportRefusal::TooMuchScanned)
        );
    }

    #[test]
    fn a_define_with_a_value_is_defined_to_it() {
        let vsh = "#version 330\nin vec3 Position;\n\
            void main() { gl_Position = vec4(Position, float(COUNT)); }\n";
        let fsh =
            "#version 330\nout vec4 fragColor;\nvoid main() { fragColor = vec4(COUNT / 4); }\n";
        assert!(translate_core(vsh, fsh, &["COUNT 8"], &include, &GLUE).is_ok());
    }

    #[test]
    fn minified_varying_declarations_are_rewritten() {
        let vsh = "#version 330\nout float sphericalVertexDistance;out float cylindricalVertexDistance;\
            void main() { sphericalVertexDistance = 0.0; cylindricalVertexDistance = 0.0; \
            gl_Position = vec4(0.0); }\n";
        let fsh = "#version 330\nin float sphericalVertexDistance;in float cylindricalVertexDistance;\
            out vec4 fragColor;void main() { fragColor = vec4(sphericalVertexDistance + cylindricalVertexDistance); }\n";
        translate_core(vsh, fsh, &[], &include, &GLUE).unwrap();
    }

    #[test]
    fn legacy_uniforms_next_to_modern_blocks_are_ignored() {
        let vsh = "#version 330\n#include <minecraft:projection.glsl>\n\
            uniform mat4 ProjMat;out vec4 vertexColor;\
            void main() { vertexColor = vec4(1.0); gl_Position = ProjMat[0]; }\n";
        let fsh = "#version 330\nin vec4 vertexColor;out vec4 fragColor;\
            void main() { fragColor = vertexColor; }\n";
        translate_core(vsh, fsh, &[], &include, &GLUE).unwrap();
    }

    /// A block declared on one line, as packs write a one-float uniform, is bound like one
    /// over several lines; the blocks and samplers the caller provides are accepted, in binding
    /// order, and one it doesn't provide is not.
    #[test]
    fn a_block_on_one_line_is_bound_and_provided_blocks_are_accepted() {
        let vsh = "#version 330\nlayout(std140) uniform Day { float daylight; };\nout vec4 vertexColor;\n\
            void main() { vertexColor = vec4(daylight); gl_Position = vec4(0.0); }\n";
        let fsh = "#version 330\nuniform sampler2D Bounds;\nin vec4 vertexColor;\nout vec4 fragColor;\n\
            void main() { fragColor = vertexColor * texture(Bounds, vec2(0.5)); }\n";
        let both = |name: &str| name == "Day" || name == "Bounds";
        let program = translate_core_with(vsh, fsh, &[], &include, &GLUE, &both).unwrap();
        assert_eq!(program.other, ["Day", "Bounds"]);
        // The bindings the caller gives them at.
        let bindings = |module: &naga::Module| -> Vec<(Option<String>, u32)> {
            module
                .global_variables
                .iter()
                .filter_map(|(_, g)| Some((g.name.clone(), g.binding.as_ref()?.binding)))
                .collect()
        };
        assert!(
            bindings(&program.vertex)
                .iter()
                .any(|(_, b)| *b == program.other_binding(0)),
            "{:?}",
            bindings(&program.vertex)
        );
        let fragment = bindings(&program.fragment);
        assert!(
            fragment.contains(&(Some("hz_tex_Bounds".into()), program.other_binding(1))),
            "{fragment:?}"
        );
        assert!(
            fragment.contains(&(Some("hz_smp_Bounds".into()), program.other_binding(1) + 1)),
            "{fragment:?}"
        );
        // Each at the line that declares it.
        let error = translate_core(vsh, fsh, &[], &include, &GLUE).unwrap_err();
        assert_eq!(
            error,
            vec![Error {
                file: VSH_FILE.into(),
                line: 2,
                message: "vertex shader: Day is not a block or sampler of this pipeline".into(),
            }]
        );
        let only_day = |name: &str| name == "Day";
        let error = translate_core_with(vsh, fsh, &[], &include, &GLUE, &only_day).unwrap_err();
        assert_eq!(
            error,
            vec![Error {
                file: FSH_FILE.into(),
                line: 2,
                message: "fragment shader: Bounds is not a block or sampler of this pipeline"
                    .into(),
            }]
        );
    }

    /// Only a uniform block's declaration names a block, written over several lines or on one.
    #[test]
    fn block_names_are_read_from_block_declarations_only() {
        assert_eq!(block_name("layout(std140) uniform Day {"), Some("Day"));
        assert_eq!(block_name("uniform Day { float daylight; };"), Some("Day"));
        assert_eq!(block_name("uniform Foo{"), Some("Foo"));
        assert_eq!(block_name("uniform Globals"), Some("Globals"));
        assert_eq!(block_name("uniform float x;"), None);
        assert_eq!(block_name("uniform sampler2D Sampler0;"), None);
    }

    /// An array too large to lay out (#4134) is the shader's error at its line, not a panic.
    #[test]
    fn an_array_too_large_to_lay_out_is_an_error() {
        let vsh = "#version 330\nout vec4 vertexColor;\nvoid main() { vertexColor = vec4(1.0); }\n";
        let fsh = "#version 330\nin vec4 vertexColor;\nfloat big[0x7fffffff];\nout vec4 fragColor;\n\
            void main() { fragColor = vertexColor; }\n";
        let error = translate_core(vsh, fsh, &[], &include, &GLUE).unwrap_err();
        assert_eq!((error[0].file.as_str(), error[0].line), (FSH_FILE, 3));
        assert!(
            error[0]
                .message
                .starts_with("fragment shader: (float big[0x7fffffff];)"),
            "{error:?}"
        );
    }

    /// An output array whose second element would take location 2^32 (#4912) is the shader's
    /// error at its line, not an overflow.
    #[test]
    fn an_output_whose_locations_pass_u32_is_an_error() {
        let vsh = "#version 330\nout vec4 vertexColor;\nvoid main() { vertexColor = vec4(1.0); }\n";
        let fsh = "#version 330\nin vec4 vertexColor;\nout vec4 fragColor;\n\
            layout(location = 4294967295u) out vec4 extra[2];\n\
            void main() { fragColor = vertexColor; extra[0] = vertexColor; extra[1] = vertexColor; }\n";
        let error = translate_core(vsh, fsh, &[], &include, &GLUE).unwrap_err();
        assert_eq!((error[0].file.as_str(), error[0].line), (FSH_FILE, 4));
        assert!(
            error[0].message.starts_with(
                "fragment shader: (layout(location = 4294967295u) out vec4 extra[2];)"
            ),
            "{error:?}"
        );
    }

    #[test]
    fn what_sunrise_does_not_provide_is_an_error() {
        let fsh = "#version 330\nin vec4 vertexColor;\nout vec4 fragColor;\nvoid main() { fragColor = vertexColor; }\n";
        // Another format's input is a global the shader may read: not an error.
        let normal = "#version 330\nin vec3 Normal;\nout vec4 vertexColor;\nvoid main() { vertexColor = vec4(Normal, 1.0); }\n";
        translate_core(normal, fsh, &[], &include, &GLUE).unwrap();
        let lightmap = "#version 330\nuniform sampler2D Lightmap;\nout vec4 vertexColor;\nvoid main() { vertexColor = texelFetch(Lightmap, ivec2(0), 0); }\n";
        let error = translate_core(lightmap, fsh, &[], &include, &GLUE).unwrap_err();
        assert_eq!(
            error,
            vec![Error {
                file: VSH_FILE.into(),
                line: 2,
                message: "vertex shader: Lightmap is not a block or sampler of this pipeline"
                    .into(),
            }]
        );
        // Behind a define the pipeline doesn't have, it is no error.
        let oit = "#version 330\n#ifdef OIT\nuniform sampler2D Coeff0;\n#endif\nout vec4 vertexColor;\n\
            void main() {\n#ifdef OIT\n    vertexColor = texelFetch(Coeff0, ivec2(0), 0);\n#endif\n}\n";
        translate_core(oit, fsh, &[], &include, &GLUE).unwrap();
        let missing = "#version 330\n#moj_import <minecraft:nothing.glsl>\n\
            out vec4 vertexColor;\nvoid main() { vertexColor = vec4(1.0); }\n";
        // At the line that imports it.
        let error = translate_core(missing, fsh, &[], &include, &GLUE).unwrap_err();
        assert_eq!(
            error,
            vec![Error {
                file: VSH_FILE.into(),
                line: 2,
                message: "minecraft:shaders/include/nothing.glsl not found".into(),
            }]
        );
        // An include the defines drop is never read, as shaderc preprocesses: missing, it is no
        // error; kept by a define, it is.
        let branch = "#version 330\n#ifdef WANTS\n#moj_import <minecraft:nothing.glsl>\n#endif\n\
            out vec4 vertexColor;\nvoid main() { vertexColor = vec4(1.0); }\n";
        translate_core(branch, fsh, &[], &include, &GLUE).unwrap();
        let error = translate_core(branch, fsh, &["WANTS"], &include, &GLUE).unwrap_err();
        assert_eq!(
            error,
            vec![Error {
                file: VSH_FILE.into(),
                line: 3,
                message: "minecraft:shaders/include/nothing.glsl not found".into(),
            }]
        );
        let active_after_skipped =
            branch.replace("#endif", "#endif\n#moj_import <minecraft:nothing.glsl>");
        let error = translate_core(&active_after_skipped, fsh, &[], &include, &GLUE).unwrap_err();
        assert_eq!(
            error,
            vec![Error {
                file: VSH_FILE.into(),
                line: 5,
                message: "minecraft:shaders/include/nothing.glsl not found".into(),
            }]
        );
        let unreadable = |id: &str| -> Result<Option<String>, String> {
            if id.ends_with("nothing.glsl") {
                Err(format!("{id}: not text"))
            } else {
                Ok(include(id))
            }
        };
        let stages = Stages {
            dir: "minecraft:shaders/core/",
            vertex: branch,
            fragment: fsh,
            vertex_file: VSH_FILE,
            fragment_file: FSH_FILE,
        };
        translate_core_at(&stages, &[], &unreadable, &GLUE, &|_| false).unwrap();
        let error =
            translate_core_at(&stages, &["WANTS"], &unreadable, &GLUE, &|_| false).unwrap_err();
        assert_eq!(
            error,
            vec![Error {
                file: VSH_FILE.into(),
                line: 3,
                message: "minecraft:shaders/include/nothing.glsl: not text".into(),
            }]
        );
    }

    #[test]
    fn unconditional_imports_share_their_dependency() {
        let large = " ".repeat(MAX_EXPANDED / 2 + 1);
        let calls = std::cell::Cell::new(0);
        let read = |id: &str| {
            if id.ends_with("/shared") {
                calls.set(calls.get() + 1);
                Ok(Some(large.clone()))
            } else {
                Ok(Some("#moj_import <shared>".to_owned()))
            }
        };
        expand_core_imports("#moj_import <left>\n#moj_import <right>", "", &[], &read).unwrap();
        assert_eq!(calls.get(), 1);
        calls.set(0);
        expand_core_imports(
            "#ifdef ENABLED\n#moj_import <left>\n#moj_import <right>\n#endif",
            "",
            &["ENABLED"],
            &read,
        )
        .unwrap();
        assert_eq!(calls.get(), 1);
        let repeated = expand_core_imports("#include <shared>\n#include <shared>", "", &[], &read);
        assert_eq!(repeated.unwrap_err(), ImportRefusal::TooLarge);
    }

    #[test]
    fn includes_follow_glsl_line_numbers_and_macros_from_the_caller_and_child() {
        let read = |id: &str| match id.rsplit('/').next() {
            Some("switch") => Ok(Some("#define ENABLE(x) ((x) + 2)".into())),
            Some("active") => Ok(Some("float kept;".into())),
            _ => panic!("read skipped include {id}"),
        };
        let source = "#version 330\n#if __LINE__ == 2\n#include <switch>\n#endif\n#if __LINE__ == 5 && ENABLE(3) == 5\n#include <active>\n#else\n#include <skipped>\n#endif";
        assert!(
            expand_core_imports(source, "", &[], &read)
                .unwrap()
                .contains("float kept;")
        );
    }

    #[test]
    fn line_directives_and_continuations_keep_the_callers_include_location() {
        let source = "#line 40\n#if defined(__LINE__) && \\\n__LINE__ == 41\n#include <inner>\n#endif\n#if __LINE__ == 46\n#include <wrong>\n#endif";
        let read = |id: &str| {
            assert!(id.ends_with("/inner"), "{id}");
            Ok(Some("#if __LINE__ == 1\nfloat included;\n#endif".into()))
        };
        assert!(
            expand_core_imports(source, "", &[], &read)
                .unwrap()
                .contains("float included;")
        );
    }

    #[test]
    fn independent_guards_do_not_replay_unrelated_definitions() {
        let source = format!(
            "#define UNUSED {}\n{}",
            "a".repeat(50_000),
            "#if 1\nfloat a;\n#endif\n".repeat(210)
        );
        assert_eq!(
            expand_core_imports(&source, "", &[], &|_| Ok(None))
                .unwrap()
                .matches("float a;")
                .count(),
            210
        );
    }

    #[test]
    fn only_an_active_include_consumes_a_legacy_import() {
        let fragment = "out vec4 fragColor;\nvoid main() { fragColor = vec4(1.0); }";
        let include = |_: &str| Some("float imported() { return 1.0; }".to_owned());
        let siblings = "#if 0\n#moj_import <a>\n#elif 0\n#moj_import <a>\n#else\n#moj_import <a>\n#endif\nvoid main() { gl_Position = vec4(imported()); }";
        translate_core(siblings, fragment, &[], &include, &GLUE).unwrap();
        for first in ["#include", "#moj_import"] {
            let vertex = format!(
                "#define hz_moj_import_0\n#if 0\n#moj_import <a>\n#endif\n{first} <a>\n#moj_import <a>\nvoid main() {{ gl_Position = vec4(imported()); }}"
            );
            translate_core(&vertex, fragment, &[], &include, &GLUE).unwrap();
        }
    }

    #[test]
    fn includes_resolve_by_id_or_from_the_including_folder() {
        assert_eq!(
            include_id(
                "<fog.glsl>",
                "minecraft:shaders/core/",
                &includes(&|_| Ok(None))
            )
            .as_deref(),
            Some("minecraft:shaders/include/fog.glsl")
        );
        assert_eq!(
            include_id(
                "<bitmade:hud/lanes.glsl>",
                "minecraft:shaders/core/",
                &includes(&|_| Ok(None))
            )
            .as_deref(),
            Some("bitmade:shaders/include/hud/lanes.glsl")
        );
        assert_eq!(
            include_id(
                "\"../include/fog.glsl\"",
                "minecraft:shaders/core/",
                &includes(&|_| Ok(None))
            )
            .as_deref(),
            Some("minecraft:shaders/include/fog.glsl")
        );
    }

    #[test]
    fn translate_supports_caller_high_bindings_and_custom_includes() {
        let read = |id: &str| -> Result<Option<String>, String> {
            match id {
                "demo:custom/inc/scale.glsl" => {
                    Ok(Some("float scale_it(float v) { return v * 0.5; }\n".into()))
                }
                _ => Ok(None),
            }
        };
        let custom_includes = Includes {
            read: &read,
            namespace: "demo",
            folder: "custom/inc/",
            once: &[],
        };
        let inputs = Inputs {
            blocks: &[("Camera", 17), ("Params", 24)],
            samplers: &[("Albedo", 15)],
            loose: &[],
            defines: &[],
            includes: &custom_includes,
        };
        let vsh = "#version 330\n\
            #include <scale.glsl>\n\
            layout(std140) uniform Camera { mat4 ViewProj; };\n\
            layout(std140) uniform Extra { float extra_v; };\n\
            out vec2 uv;\n\
            void main() {\n\
                uv = vec2(scale_it(extra_v), 0.5);\n\
                gl_Position = ViewProj * vec4(0.0, 0.0, 0.0, 1.0);\n\
            }\n";
        let fsh = "#version 330\n\
            layout(std140) uniform Params { vec4 Tint; };\n\
            uniform sampler2D Albedo;\n\
            uniform sampler2D ExtraTex;\n\
            in vec2 uv;\n\
            out vec4 fragColor;\n\
            void main() {\n\
                fragColor = Tint * texture(Albedo, uv) * texture(ExtraTex, uv);\n\
            }\n";
        let stages = Stages {
            dir: "demo:custom/core/",
            vertex: vsh,
            fragment: fsh,
            vertex_file: "demo:custom/core/test.vsh",
            fragment_file: "demo:custom/core/test.fsh",
        };
        let provided = |name: &str| name == "Extra" || name == "ExtraTex";
        let program = translate(&stages, &inputs, &GLUE, &provided)
            .expect("caller bindings >= 15 and provided others must translate");
        assert_eq!(program.other, ["Extra", "ExtraTex"]);
        // Highest caller binding is Params at 24 (Albedo uses 15 and 16, Camera uses 17),
        // so `other` starts at 25 without colliding with any caller binding.
        assert_eq!(program.other_binding(0), 25);
        assert_eq!(program.other_binding(1), 27);
        assert_eq!(
            program.stages_reading(15),
            wgpu_types::ShaderStages::FRAGMENT
        );
        assert_eq!(
            program.stages_reading(16),
            wgpu_types::ShaderStages::FRAGMENT
        );
        assert_eq!(program.stages_reading(17), wgpu_types::ShaderStages::VERTEX);
        assert_eq!(
            program.stages_reading(24),
            wgpu_types::ShaderStages::FRAGMENT
        );
        assert_eq!(program.stages_reading(25), wgpu_types::ShaderStages::VERTEX);
        assert_eq!(
            program.stages_reading(27),
            wgpu_types::ShaderStages::FRAGMENT
        );

        // An unprovided extra block is still refused by its real name.
        let err = translate(&stages, &inputs, &GLUE, &|_| false).unwrap_err();
        assert_eq!(
            err,
            vec![Error {
                file: "demo:custom/core/test.vsh".into(),
                line: 4,
                message: "vertex shader: Extra is not a block or sampler of this pipeline".into(),
            }]
        );

        // With `once: &[]`, `#moj_import` is not treated as an include directive.
        let untouched = expand_includes(
            "#moj_import <scale.glsl>\nfloat x;\n",
            "demo:custom/core/test.vsh",
            "demo:custom/core/",
            &custom_includes,
            None,
        )
        .unwrap();
        assert!(
            untouched.text().contains("#moj_import <scale.glsl>"),
            "{}",
            untouched.text()
        );
    }
}

#[cfg(test)]
#[path = "texel_tests.rs"]
mod texel_buffer_tests;

#[cfg(test)]
#[path = "located_tests.rs"]
mod located_program_tests;
