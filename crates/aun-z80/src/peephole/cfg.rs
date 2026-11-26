use super::*;

/// Tail-merging (cross-jumping) for `ret`-terminated basic blocks within the same function:
/// when two `_bb<digits>` blocks share an identical trailing instruction sequence of at least
/// 4 instructions ending in `ret`, inserts a fresh `_bb<N>` label before the later block's
/// suffix and replaces the earlier block's duplicate suffix with `jp <label>`.
pub(super) fn merge_ret_tails(lines: &[String]) -> (Vec<String>, bool) {
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();

    // Collect instruction-only spans for each `_bb<digits>` block that ends in an unconditional `ret`.
    // Each entry: (fn_prefix, inst_line_indices)
    let mut ret_blocks: Vec<(String, Vec<usize>)> = Vec::new();
    let mut existing_labels: HashSet<String> = HashSet::new();
    let mut cur_fn_prefix: Option<String> = None;
    let mut cur_insts: Vec<usize> = Vec::new();

    for (idx, kind) in kinds.iter().enumerate() {
        match kind {
            LineKind::Label(lbl) => {
                existing_labels.insert(lbl.clone());
                cur_insts.clear();
                if let Some((prefix, _)) = lbl.rsplit_once("_bb")
                    && is_bb_label(lbl)
                {
                    cur_fn_prefix = Some(prefix.to_string());
                } else {
                    cur_fn_prefix = None;
                }
            }
            LineKind::Directive => {
                cur_fn_prefix = None;
                cur_insts.clear();
            }
            LineKind::Inst { mnemonic, ops } => {
                if let Some(ref prefix) = cur_fn_prefix {
                    cur_insts.push(idx);
                    if mnemonic == "ret" && ops.is_empty() {
                        ret_blocks.push((prefix.clone(), std::mem::take(&mut cur_insts)));
                        cur_fn_prefix = None;
                    } else if matches!(mnemonic.as_str(), "jp" | "jr" | "reti" | "retn" | "halt")
                        && ops.len() <= 1
                    {
                        cur_fn_prefix = None;
                        cur_insts.clear();
                    }
                }
            }
            LineKind::EmptyOrComment => {}
        }
    }

    // Find the best pair within the same function sharing >= 4 trailing instructions.
    let mut best: Option<(String, usize, usize, usize)> = None; // (prefix, block_a_idx, block_b_idx, common_len)
    for a in 0..ret_blocks.len() {
        for b in (a + 1)..ret_blocks.len() {
            if ret_blocks[a].0 != ret_blocks[b].0 {
                continue;
            }
            let ia = &ret_blocks[a].1;
            let ib = &ret_blocks[b].1;
            let mut k = 0usize;
            while k < ia.len() && k < ib.len() {
                if kinds[ia[ia.len() - 1 - k]] == kinds[ib[ib.len() - 1 - k]] {
                    k += 1;
                } else {
                    break;
                }
            }
            if k >= 4 && best.as_ref().is_none_or(|&(_, _, _, bk)| k > bk) {
                best = Some((ret_blocks[a].0.clone(), a, b, k));
            }
        }
    }

    let Some((prefix, a_idx, b_idx, common_len)) = best else {
        return (lines.to_vec(), false);
    };

    let ia = &ret_blocks[a_idx].1;
    let ib = &ret_blocks[b_idx].1;
    let a_start_line = ia[ia.len() - common_len];
    let a_end_line = *ia.last().unwrap();
    let b_start_line = ib[ib.len() - common_len];

    // Generate a fresh `_bb<N>` label so `asm.rs` JP->JR relaxation applies cleanly.
    let mut n = 9000u32;
    let tail_lbl = loop {
        let cand = format!("{prefix}_bb{n}");
        if !existing_labels.contains(&cand) {
            break cand;
        }
        n += 1;
    };

    let mut out: Vec<String> = Vec::with_capacity(lines.len() + 1);
    for (idx, line) in lines.iter().enumerate() {
        if idx == a_start_line {
            out.push(format!("    jp {tail_lbl}"));
        } else if idx > a_start_line && idx <= a_end_line {
            continue;
        } else if idx == b_start_line {
            out.push(format!("{tail_lbl}:"));
            out.push(line.clone());
        } else {
            out.push(line.clone());
        }
    }

    (out, true)
}

/// Jump-threading for compiler-generated `_bb<digits>` trampoline labels:
/// if a `_bb<digits>` label `L1` is immediately followed by an unconditional `jp L2` / `jr L2`
/// (where `L2` is also `_bb<digits>` and `L2 != L1`), rewrites all `jp/jr [cc,] L1` to target `L2`.
pub(super) fn thread_empty_jumps(lines: &[String]) -> (Vec<String>, bool) {
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
    let mut redirect: HashMap<String, String> = HashMap::new();

    for i in 0..kinds.len() {
        if let LineKind::Label(lbl) = &kinds[i]
            && is_bb_label(lbl)
        {
            // Skip past comments and adjacent labels to find the first instruction
            let mut k = i + 1;
            while k < kinds.len()
                && matches!(kinds[k], LineKind::EmptyOrComment | LineKind::Label(_))
            {
                k += 1;
            }
            if k < kinds.len()
                && let LineKind::Inst { mnemonic, ops } = &kinds[k]
                && (mnemonic == "jp" || mnemonic == "jr")
                && ops.len() == 1
                && is_bb_label(&ops[0])
                && &ops[0] != lbl
            {
                redirect.insert(lbl.clone(), ops[0].clone());
            }
        }
    }
    if redirect.is_empty() {
        return (lines.to_vec(), false);
    }

    let mut out = Vec::with_capacity(lines.len());
    let mut changed = false;
    for (i, line) in lines.iter().enumerate() {
        // Only thread `jp` (never `jr`, which has a +-127 byte range limit) and guard cycles
        if let LineKind::Inst { mnemonic, ops } = &kinds[i]
            && mnemonic == "jp"
            && let Some(target) = ops.last()
            && let Some(new_tgt) = redirect.get(target)
            && redirect.get(new_tgt).map(String::as_str) != Some(target.as_str())
        {
            if ops.len() == 1 {
                out.push(format!("    jp {new_tgt}"));
            } else {
                out.push(format!("    jp {}, {new_tgt}", ops[0]));
            }
            changed = true;
            continue;
        }
        out.push(line.clone());
    }
    (out, changed)
}
