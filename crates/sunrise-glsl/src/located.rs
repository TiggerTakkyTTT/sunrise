//! Where each line of a stage came from, carried through every rewrite, so an error naga finds
//! in the code it parses names the file and the line the author wrote ([`Error`]).

use std::ops::Range;

/// A diagnostic at the file (`ns:path`, as includes name it) and line (from 1) the author wrote.
/// Line 0 is the file as a whole: the error has no line of its own (a link between the stages,
/// or code Sunrise wrote around the author's).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub file: String,
    pub line: u32,
    pub message: String,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.file.as_str(), self.line) {
            ("", _) => f.write_str(&self.message),
            (file, 0) => write!(f, "{file}: {}", self.message),
            (file, line) => write!(f, "{file}:{line}: {}", self.message),
        }
    }
}

impl std::error::Error for Error {}

/// Where something is in the author's files: a file id and a line from 1, as [`Error`] has them
/// (line 0: the file as a whole; no file: nowhere known, for a program made without its
/// sources).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Location {
    pub file: String,
    pub line: u32,
}

impl Location {
    /// `message` here.
    pub fn error(&self, message: impl std::fmt::Display) -> Error {
        Error {
            file: self.file.clone(),
            line: self.line,
            message: message.to_string(),
        }
    }
}

/// Errors as one line, `; ` between them: a load note's text.
pub fn describe(errors: &[Error]) -> String {
    let mut out = String::new();
    for (n, error) in errors.iter().enumerate() {
        if n > 0 {
            out.push_str("; ");
        }
        out.push_str(&error.to_string());
    }
    out
}

/// A line's file (an index into [`Expanded`]'s files) and its line number from 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Origin {
    pub file: u32,
    pub line: u32,
}

/// Text and where each of its lines came from (`None`: Sunrise's own code). There is one
/// origin per line: one more than the text has line breaks.
#[derive(Clone, Debug)]
pub(crate) struct Lines {
    text: String,
    origins: Vec<Option<Origin>>,
}

impl Default for Lines {
    fn default() -> Self {
        Self {
            text: String::new(),
            origins: vec![None],
        }
    }
}

impl Lines {
    /// `text` with all its lines from `origin`.
    pub fn uniform(text: &str, origin: Option<Origin>) -> Self {
        let mut lines = Self::default();
        lines.push(text, origin);
        lines
    }

    /// `text` as file 0 wrote it, line for line.
    #[cfg(test)]
    pub fn from_text(text: &str) -> Self {
        let mut lines = Self::default();
        for (n, piece) in text.split_inclusive('\n').enumerate() {
            let line = u32::try_from(n + 1).unwrap_or(u32::MAX);
            lines.push(piece, Some(Origin { file: 0, line }));
        }
        lines
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    /// Where the line holding byte `at` came from.
    pub fn origin_at(&self, at: usize) -> Option<Origin> {
        let line = self.text[..at.min(self.text.len())].matches('\n').count();
        self.origins.get(line).copied().flatten()
    }

    /// Where line `line` (from 0) came from.
    pub fn origin(&self, line: usize) -> Option<Origin> {
        self.origins.get(line).copied().flatten()
    }

    /// The origins of the line holding byte `at` and of every line after it.
    pub fn origins_from(&self, at: usize) -> &[Option<Origin>] {
        let line = self.text[..at.min(self.text.len())].matches('\n').count();
        &self.origins[line..]
    }

    /// Appends `text`, its lines from `origin`. A line it continues keeps the origin it has,
    /// unless nothing is on it yet.
    pub fn push(&mut self, text: &str, origin: Option<Origin>) {
        if text.is_empty() {
            return;
        }
        if self.at_line_start() {
            *self.origins.last_mut().expect("one origin per line") = origin;
        }
        for _ in text.matches('\n') {
            self.origins.push(origin);
        }
        self.text.push_str(text);
    }

    /// Appends `other` with its origins, under the same rule as [`Lines::push`].
    pub fn push_lines(&mut self, other: &Lines) {
        if other.text.is_empty() {
            return;
        }
        if self.at_line_start() {
            *self.origins.last_mut().expect("one origin per line") = other.origins[0];
        }
        self.origins.extend_from_slice(&other.origins[1..]);
        self.text.push_str(&other.text);
    }

    fn at_line_start(&self) -> bool {
        self.text.is_empty() || self.text.ends_with('\n')
    }

    /// Appends `src`'s text in `range`, whose first byte is on `src`'s line `line`, with the
    /// origins of its lines; the line of `range.end`.
    fn push_slice(&mut self, src: &Lines, range: Range<usize>, line: usize) -> usize {
        let mut line = line;
        for piece in src.text[range].split_inclusive('\n') {
            self.push(piece, src.origins[line]);
            if piece.ends_with('\n') {
                line += 1;
            }
        }
        line
    }

    /// Each range (in order, apart) replaced by its text. A replacement with as many line breaks
    /// as the text it replaces keeps that text's origins line for line; any other one comes
    /// from the line its range starts on.
    pub fn edit(&mut self, edits: Vec<(Range<usize>, String)>) {
        let mut line = 0;
        let mut pos = 0;
        let edits = edits
            .into_iter()
            .map(|(range, with)| {
                line += self.text[pos..range.start].matches('\n').count();
                pos = range.start;
                let replaced = self.text[range.clone()].matches('\n').count();
                let lines = if with.matches('\n').count() == replaced {
                    let mut lines = Lines::default();
                    for (n, piece) in with.split_inclusive('\n').enumerate() {
                        lines.push(piece, self.origins[line + n]);
                    }
                    lines
                } else {
                    Lines::uniform(&with, self.origins[line])
                };
                (range, lines)
            })
            .collect();
        self.edit_lines(edits);
    }

    /// Each range (in order, apart) replaced by its lines, which bring their own origins. The
    /// first line of a replacement continues the line before its range when that has text; the
    /// text after the range continues its last line, taking its own origin when that is empty.
    pub fn edit_lines(&mut self, edits: Vec<(Range<usize>, Lines)>) {
        let mut out = Lines::default();
        let (mut pos, mut line) = (0, 0);
        for (range, with) in edits {
            line = out.push_slice(self, pos..range.start, line);
            line += self.text[range.clone()].matches('\n').count();
            out.push_lines(&with);
            pos = range.end;
        }
        out.push_slice(self, pos..self.text.len(), line);
        *self = out;
    }

    /// [`Lines::edit`] of one range.
    pub fn replace(&mut self, range: Range<usize>, with: &str) {
        self.edit(vec![(range, with.to_owned())]);
    }

    /// [`Lines::edit_lines`] of one range.
    pub fn replace_lines(&mut self, range: Range<usize>, with: Lines) {
        self.edit_lines(vec![(range, with)]);
    }
}

/// A stage with its includes inlined ([`crate::expand_includes`]), and where each of its lines
/// came from: the file id (`ns:path`) and the line. Rewriting it keeps them
/// ([`Expanded::edit`], [`Expanded::wrap`]).
#[derive(Clone, Debug)]
pub struct Expanded {
    pub(crate) lines: Lines,
    /// The stage file first, then each include, by id.
    pub(crate) files: Vec<String>,
}

impl Expanded {
    pub fn text(&self) -> &str {
        self.lines.text()
    }

    /// The stage's own file id.
    pub fn file(&self) -> &str {
        &self.files[0]
    }

    /// The error at byte `at` of [`Expanded::text`], at the file and line that byte came from.
    /// Sunrise's own lines, or no byte, name the stage file with no line.
    pub fn error_at(&self, at: Option<usize>, message: String) -> Error {
        self.location_at(at).error(message)
    }

    /// [`Expanded::error_at`] the start of naga's `span` (none for an unknown one).
    pub fn error_in(&self, span: naga::Span, message: String) -> Error {
        self.error_at(span.to_range().map(|r| r.start), message)
    }

    /// [`Expanded::error_in`] for each of naga's errors, the text's lines found once for all.
    pub fn errors_in(&self, errors: impl IntoIterator<Item = (naga::Span, String)>) -> Vec<Error> {
        let text = self.text();
        let breaks: Vec<usize> = text.match_indices('\n').map(|(at, _)| at).collect();
        (errors.into_iter())
            .map(|(span, message)| {
                let origin = span.to_range().and_then(|r| {
                    let start = r.start.min(text.len());
                    self.lines.origin(breaks.partition_point(|&b| b < start))
                });
                self.located(origin, message)
            })
            .collect()
    }

    fn located(&self, origin: Option<Origin>, message: String) -> Error {
        self.location(origin).error(message)
    }

    /// The file and line of `origin`; the stage file with no line for Sunrise's own code.
    fn location(&self, origin: Option<Origin>) -> Location {
        match origin {
            Some(origin) => Location {
                file: self.files[origin.file as usize].clone(),
                line: origin.line,
            },
            None => Location {
                file: self.file().to_owned(),
                line: 0,
            },
        }
    }

    /// Where byte `at` of [`Expanded::text`] came from ([`Expanded::error_at`]).
    pub fn location_at(&self, at: Option<usize>) -> Location {
        self.location(at.and_then(|at| self.lines.origin_at(at)))
    }

    /// Where the start of naga's `span` came from.
    pub fn location_in(&self, span: naga::Span) -> Location {
        self.location_at(span.to_range().map(|r| r.start))
    }

    /// Each range of [`Expanded::text`] (in order, apart) replaced by its text. A replacement
    /// with as many line breaks as the text it replaces keeps that text's lines, line for line;
    /// one with any other count takes the line its range starts on for all its lines.
    pub fn edit(&mut self, edits: Vec<(Range<usize>, String)>) {
        self.lines.edit(edits);
    }

    /// Sunrise's own code put `before` and `after` the text.
    pub fn wrap(&mut self, before: &str, after: &str) {
        let mut lines = Lines::uniform(before, None);
        lines.push_lines(&self.lines);
        lines.push(after, None);
        self.lines = lines;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(lines: &Lines, needle: &str) -> Option<u32> {
        lines
            .origin_at(lines.text().find(needle).expect("in the text"))
            .map(|o| o.line)
    }

    /// Text added in the middle of a line comes from that line; lines it adds too; the rest of
    /// the line keeps its own.
    #[test]
    fn an_edit_that_adds_lines_takes_its_starts_line() {
        let mut lines = Lines::from_text("a\nb X c\nd\n");
        let x = lines.text().find('X').unwrap();
        lines.replace(x..x + 1, "1\n2\n3");
        assert_eq!(lines.text(), "a\nb 1\n2\n3 c\nd\n");
        assert_eq!(
            [
                at(&lines, "a"),
                at(&lines, "b"),
                at(&lines, "2"),
                at(&lines, "3 c"),
                at(&lines, "d")
            ],
            [Some(1), Some(2), Some(2), Some(2), Some(3)]
        );
    }

    /// A replacement of as many lines as it replaces keeps their origins line for line; text
    /// removed with its line breaks leaves the next lines theirs.
    #[test]
    fn an_edit_of_the_same_lines_keeps_them_and_a_removal_keeps_the_rest() {
        let mut lines = Lines::from_text("a\nb\nc\nd\ne\n");
        let b = lines.text().find('b').unwrap();
        let d = lines.text().find('d').unwrap();
        lines.edit(vec![(b..b + 3, "B\nC".into()), (d..d + 2, String::new())]);
        assert_eq!(lines.text(), "a\nB\nC\ne\n");
        assert_eq!(
            [at(&lines, "B"), at(&lines, "C"), at(&lines, "e")],
            [Some(2), Some(3), Some(5)]
        );
    }

    /// Lines brought with their own origins keep them; the text after the range continues the
    /// last one, with its own origin when that line is empty.
    #[test]
    fn edited_lines_keep_their_origins() {
        let mut lines = Lines::from_text("a\nb\nc\n");
        let b = lines.text().find('b').unwrap();
        let mut with = Lines::uniform("p\n", Some(Origin { file: 1, line: 7 }));
        with.push("q\n", Some(Origin { file: 1, line: 8 }));
        lines.replace_lines(b..b, with);
        assert_eq!(lines.text(), "a\np\nq\nb\nc\n");
        assert_eq!(
            [at(&lines, "p"), at(&lines, "q"), at(&lines, "b")],
            [Some(7), Some(8), Some(2)]
        );
    }

    #[test]
    fn an_error_names_its_file_and_line_or_the_file_alone() {
        let error = |line| Error {
            file: "minecraft:shaders/core/text.fsh".into(),
            line,
            message: "fragment shader: oops".into(),
        };
        assert_eq!(
            describe(&[error(3), error(0)]),
            "minecraft:shaders/core/text.fsh:3: fragment shader: oops; \
             minecraft:shaders/core/text.fsh: fragment shader: oops"
        );
    }
}
