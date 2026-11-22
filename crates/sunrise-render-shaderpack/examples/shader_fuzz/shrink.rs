//! Failing cases made small while the same invariant still fails, and grouped by invariant and
//! first failing step.

use std::collections::BTreeMap;
use std::time::Instant;

use crate::cases::Case;
use crate::check::{Outcome, Request};

pub type Files = BTreeMap<String, Vec<u8>>;

/// Runs a case: a worker process in a run, the checker itself (or a fake) in tests.
pub trait Evaluate {
    fn eval(&mut self, files: &Files, request: &Request) -> Outcome;
}

impl<F: FnMut(&Files, &Request) -> Outcome> Evaluate for F {
    fn eval(&mut self, files: &Files, request: &Request) -> Outcome {
        self(files, request)
    }
}

fn fails(ev: &mut dyn Evaluate, files: &Files, request: &Request, signature: &str) -> bool {
    ev.eval(files, request)
        .failures
        .iter()
        .any(|f| f.signature() == signature)
}

/// The first step that makes `signature` fail, deriving the request from each candidate.
pub fn first_step_with(
    ev: &mut dyn Evaluate,
    case: &Case,
    signature: &str,
    mut request_for: impl FnMut(&Case, &Files) -> Request,
) -> String {
    for n in 0..case.steps.len() {
        let candidate = Case {
            steps: case.steps[..n].to_vec(),
            ..case.clone()
        };
        let files = candidate.materialize();
        let request = request_for(&candidate, &files);
        if fails(ev, &files, &request, signature) {
            return match n {
                0 => "base".to_owned(),
                n => case.steps[n - 1].kind.name().to_owned(),
            };
        }
    }
    case.steps
        .last()
        .map_or("base".to_owned(), |s| s.kind.name().to_owned())
}

/// How long a shrink may go on.
#[derive(Clone, Copy)]
pub struct Budget {
    pub evals: usize,
    pub until: Instant,
}

impl Budget {
    fn spend(&mut self) -> bool {
        if self.evals == 0 || Instant::now() >= self.until {
            return false;
        }
        self.evals -= 1;
        true
    }
}

#[derive(Debug, PartialEq)]
pub struct Shrunk {
    /// The steps still needed, with the files they were applied to.
    pub steps: Vec<crate::cases::Step>,
    pub files: Files,
}

/// Shrinks a failure while recalculating expectations for every candidate file set.
pub fn shrink_with(
    ev: &mut dyn Evaluate,
    case: &Case,
    signature: &str,
    mut budget: Budget,
    mut request_for: impl FnMut(&Case, &Files) -> Request,
) -> Shrunk {
    let fails_candidate = |ev: &mut dyn Evaluate,
                           candidate: &Case,
                           files: &Files,
                           request_for: &mut dyn FnMut(&Case, &Files) -> Request,
                           signature: &str| {
        let request = request_for(candidate, files);
        fails(ev, files, &request, signature)
    };
    let mut steps = case.steps.clone();
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..steps.len() {
            if !budget.spend() {
                break;
            }
            let mut fewer = steps.clone();
            fewer.remove(i);
            let candidate = Case {
                steps: fewer.clone(),
                ..case.clone()
            };
            let files = candidate.materialize();
            if fails_candidate(ev, &candidate, &files, &mut request_for, signature) {
                steps = fewer;
                changed = true;
                break;
            }
        }
    }
    let mut files = Case {
        steps: steps.clone(),
        ..case.clone()
    }
    .materialize();
    // Biggest first: they hold the most to remove.
    let mut paths: Vec<String> = files.keys().cloned().collect();
    paths.sort_by_key(|p| std::cmp::Reverse(files[p].len()));
    for path in &paths {
        if !budget.spend() {
            break;
        }
        let mut fewer = files.clone();
        fewer.remove(path);
        let candidate = Case {
            steps: steps.clone(),
            ..case.clone()
        };
        if fails_candidate(ev, &candidate, &fewer, &mut request_for, signature) {
            files = fewer;
        }
    }
    let mut paths: Vec<String> = files.keys().cloned().collect();
    paths.sort_by_key(|p| std::cmp::Reverse(files[p].len()));
    for path in paths {
        let mut lines: Vec<Vec<u8>> = files[&path]
            .split(|&b| b == b'\n')
            .map(<[u8]>::to_vec)
            .collect();
        let mut run = lines.len().div_ceil(2).max(1);
        loop {
            let mut i = 0;
            while i < lines.len() && budget.spend() {
                let mut fewer = lines.clone();
                fewer.drain(i..(i + run).min(lines.len()));
                let text = fewer.join(&b'\n');
                let mut candidate = files.clone();
                let emptied = text.iter().all(u8::is_ascii_whitespace);
                candidate.insert(path.clone(), text);
                let case_candidate = Case {
                    steps: steps.clone(),
                    ..case.clone()
                };
                if !emptied
                    && fails_candidate(ev, &case_candidate, &candidate, &mut request_for, signature)
                {
                    lines = fewer;
                    files = candidate;
                } else {
                    i += run;
                }
            }
            if run == 1 || budget.evals == 0 || Instant::now() >= budget.until {
                break;
            }
            run = run.div_ceil(2);
        }
    }
    // Files the lines left empty, or no longer needed.
    let paths: Vec<String> = files.keys().cloned().collect();
    for path in paths {
        if !budget.spend() {
            break;
        }
        let mut fewer = files.clone();
        fewer.remove(&path);
        let candidate = Case {
            steps: steps.clone(),
            ..case.clone()
        };
        if fails_candidate(ev, &candidate, &fewer, &mut request_for, signature) {
            files = fewer;
        }
    }
    Shrunk { steps, files }
}

/// A failing case found by a run.
#[derive(Clone, Debug)]
pub struct Found {
    pub case: Case,
    pub request: Request,
    pub failure: crate::check::Failure,
    pub step: String,
}

impl Found {
    /// The group: invariant, first failing step, and what about it.
    pub fn group(&self) -> String {
        format!(
            "{}|{}|{}",
            self.failure.invariant, self.step, self.failure.detail
        )
    }
}

/// Failures by group, each group's cases smallest first.
pub fn group(found: Vec<Found>) -> BTreeMap<String, Vec<Found>> {
    let mut groups: BTreeMap<String, Vec<Found>> = BTreeMap::new();
    for f in found {
        groups.entry(f.group()).or_default().push(f);
    }
    for members in groups.values_mut() {
        members.sort_by_key(|f| {
            (
                f.case.materialize().values().map(Vec::len).sum::<usize>(),
                f.case.seed,
            )
        });
    }
    groups
}
