//! Control-flow graph, Cooper-Harvey-Kennedy dominator tree, dominance frontier,
//! and natural loop detection for `aun-ir`.

use crate::{BlockId, IrFunction, Terminator};
use std::collections::{BTreeSet, HashMap};

/// A natural loop identified by a back-edge `latch -> header` where `header` dominates `latch`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NaturalLoop {
    pub header: BlockId,
    pub latch: BlockId,
    pub body: BTreeSet<BlockId>,
}

/// Control-flow and dominance analysis for a single `IrFunction`.
#[derive(Debug, Clone)]
pub struct Cfg {
    id_to_idx: HashMap<BlockId, usize>,
    idx_to_id: Vec<BlockId>,
    entry_idx: Option<usize>,
    reachable: Vec<bool>,
    rpo_indices: Vec<usize>,
    rpo_ids: Vec<BlockId>,
    rpo_num: Vec<Option<usize>>,
    succs: Vec<Vec<BlockId>>,
    preds: Vec<Vec<BlockId>>,
    idom_idx: Vec<Option<usize>>,
    dom_children: Vec<Vec<BlockId>>,
    df: Vec<Vec<BlockId>>,
    loops: Vec<NaturalLoop>,
}

impl Cfg {
    /// Builds CFG, dominator tree, dominance frontiers, and natural loops for `func`.
    pub fn build(func: &IrFunction) -> Self {
        let num_blocks = func.blocks.len();
        let mut id_to_idx = HashMap::with_capacity(num_blocks);
        let mut idx_to_id = Vec::with_capacity(num_blocks);
        for (idx, bb) in func.blocks.iter().enumerate() {
            id_to_idx.insert(bb.id, idx);
            idx_to_id.push(bb.id);
        }

        let entry_idx = id_to_idx.get(&0).copied().or_else(|| {
            if num_blocks > 0 {
                Some(0)
            } else {
                None
            }
        });

        let mut succ_indices = vec![Vec::new(); num_blocks];
        let mut pred_indices = vec![Vec::new(); num_blocks];
        let mut succs = vec![Vec::new(); num_blocks];
        let mut preds = vec![Vec::new(); num_blocks];

        for (idx, bb) in func.blocks.iter().enumerate() {
            match bb.term {
                Terminator::Goto(target) => {
                    if let Some(&t_idx) = id_to_idx.get(&target) {
                        succ_indices[idx].push(t_idx);
                        succs[idx].push(target);
                        pred_indices[t_idx].push(idx);
                        preds[t_idx].push(bb.id);
                    }
                }
                Terminator::Branch {
                    then_bb, else_bb, ..
                } => {
                    if let Some(&t_idx) = id_to_idx.get(&then_bb) {
                        succ_indices[idx].push(t_idx);
                        succs[idx].push(then_bb);
                        pred_indices[t_idx].push(idx);
                        preds[t_idx].push(bb.id);
                    }
                    if else_bb != then_bb
                        && let Some(&e_idx) = id_to_idx.get(&else_bb)
                    {
                        succ_indices[idx].push(e_idx);
                        succs[idx].push(else_bb);
                        pred_indices[e_idx].push(idx);
                        preds[e_idx].push(bb.id);
                    }
                }
                Terminator::Return(_) | Terminator::Unreachable => {}
            }
        }

        // Iterative DFS to compute reachability and postorder.
        let mut reachable = vec![false; num_blocks];
        let mut postorder = Vec::with_capacity(num_blocks);
        if let Some(entry) = entry_idx {
            let mut stack: Vec<(usize, usize)> = Vec::new();
            reachable[entry] = true;
            stack.push((entry, 0));
            while let Some((node, next_edge)) = stack.last_mut() {
                let u = *node;
                if *next_edge < succ_indices[u].len() {
                    let v = succ_indices[u][*next_edge];
                    *next_edge += 1;
                    if !reachable[v] {
                        reachable[v] = true;
                        stack.push((v, 0));
                    }
                } else {
                    postorder.push(u);
                    stack.pop();
                }
            }
        }

        postorder.reverse();
        let rpo_indices = postorder;
        let rpo_ids: Vec<BlockId> = rpo_indices.iter().map(|&idx| idx_to_id[idx]).collect();
        let mut rpo_num = vec![None; num_blocks];
        for (pos, &idx) in rpo_indices.iter().enumerate() {
            rpo_num[idx] = Some(pos);
        }

        // Cooper-Harvey-Kennedy iterative dominator algorithm.
        let mut idom_idx: Vec<Option<usize>> = vec![None; num_blocks];
        if let Some(entry) = entry_idx {
            idom_idx[entry] = Some(entry);
            let mut changed = true;
            while changed {
                changed = false;
                for &b in rpo_indices.iter().skip(1) {
                    let mut new_idom: Option<usize> = None;
                    for &p in &pred_indices[b] {
                        if idom_idx[p].is_some() {
                            new_idom = Some(match new_idom {
                                None => p,
                                Some(cur) => intersect(p, cur, &idom_idx, &rpo_num),
                            });
                        }
                    }
                    if new_idom.is_some() && idom_idx[b] != new_idom {
                        idom_idx[b] = new_idom;
                        changed = true;
                    }
                }
            }
        }

        // Dominator tree children (ordered by RPO).
        let mut dom_children = vec![Vec::new(); num_blocks];
        for &b in rpo_indices.iter().skip(1) {
            if let Some(parent) = idom_idx[b] {
                dom_children[parent].push(idx_to_id[b]);
            }
        }

        // Dominance frontiers (Cooper-Harvey-Kennedy algorithm).
        let mut df_sets: Vec<BTreeSet<BlockId>> = vec![BTreeSet::new(); num_blocks];
        for &b in &rpo_indices {
            let reach_preds: Vec<usize> = pred_indices[b]
                .iter()
                .copied()
                .filter(|&p| reachable[p])
                .collect();
            if reach_preds.len() >= 2 {
                let b_idom = idom_idx[b].expect("reachable join node must have idom");
                let b_id = idx_to_id[b];
                for p in reach_preds {
                    let mut runner = p;
                    while runner != b_idom {
                        df_sets[runner].insert(b_id);
                        runner = idom_idx[runner].expect("dominator chain must reach entry");
                    }
                }
            }
        }
        let df: Vec<Vec<BlockId>> = df_sets
            .into_iter()
            .map(|s| s.into_iter().collect())
            .collect();

        // Natural loop detection via back-edges `latch -> header` where `header` dominates `latch`.
        let mut loops = Vec::new();
        for &latch in &rpo_indices {
            for &header in &succ_indices[latch] {
                if dominates_idx(header, latch, entry_idx, &reachable, &idom_idx) {
                    let mut body_idx = BTreeSet::new();
                    body_idx.insert(header);
                    body_idx.insert(latch);
                    if latch != header {
                        let mut worklist = vec![latch];
                        while let Some(node) = worklist.pop() {
                            for &p in &pred_indices[node] {
                                if reachable[p] && body_idx.insert(p) {
                                    worklist.push(p);
                                }
                            }
                        }
                    }
                    let body: BTreeSet<BlockId> =
                        body_idx.into_iter().map(|i| idx_to_id[i]).collect();
                    loops.push(NaturalLoop {
                        header: idx_to_id[header],
                        latch: idx_to_id[latch],
                        body,
                    });
                }
            }
        }
        loops.sort_by_key(|lp| {
            let h_rpo = id_to_idx
                .get(&lp.header)
                .and_then(|&i| rpo_num[i])
                .unwrap_or(usize::MAX);
            let l_rpo = id_to_idx
                .get(&lp.latch)
                .and_then(|&i| rpo_num[i])
                .unwrap_or(usize::MAX);
            (h_rpo, l_rpo)
        });

        Self {
            id_to_idx,
            idx_to_id,
            entry_idx,
            reachable,
            rpo_indices,
            rpo_ids,
            rpo_num,
            succs,
            preds,
            idom_idx,
            dom_children,
            df,
            loops,
        }
    }

    pub fn num_blocks(&self) -> usize {
        self.idx_to_id.len()
    }

    pub fn entry_block(&self) -> Option<BlockId> {
        self.entry_idx.map(|idx| self.idx_to_id[idx])
    }

    pub fn is_reachable(&self, bb: BlockId) -> bool {
        self.id_to_idx
            .get(&bb)
            .is_some_and(|&idx| self.reachable[idx])
    }

    /// Reachable blocks in Reverse Postorder.
    pub fn rpo(&self) -> &[BlockId] {
        &self.rpo_ids
    }

    /// RPO position of a reachable block (`0` for the entry block).
    pub fn rpo_pos(&self, bb: BlockId) -> Option<usize> {
        self.id_to_idx.get(&bb).and_then(|&idx| self.rpo_num[idx])
    }

    /// Dense block indices in Reverse Postorder.
    pub fn rpo_indices(&self) -> &[usize] {
        &self.rpo_indices
    }

    pub fn succs(&self, bb: BlockId) -> &[BlockId] {
        self.id_to_idx
            .get(&bb)
            .map_or(&[], |&idx| &self.succs[idx])
    }

    pub fn preds(&self, bb: BlockId) -> &[BlockId] {
        self.id_to_idx
            .get(&bb)
            .map_or(&[], |&idx| &self.preds[idx])
    }

    pub fn reachable_preds(&self, bb: BlockId) -> Vec<BlockId> {
        self.preds(bb)
            .iter()
            .copied()
            .filter(|&p| self.is_reachable(p))
            .collect()
    }

    /// Immediate strict dominator of `bb`, or `None` for the entry block and unreachable blocks.
    pub fn idom(&self, bb: BlockId) -> Option<BlockId> {
        let idx = *self.id_to_idx.get(&bb)?;
        if Some(idx) == self.entry_idx || !self.reachable[idx] {
            return None;
        }
        self.idom_idx[idx].map(|parent| self.idx_to_id[parent])
    }

    /// Direct children of `bb` in the dominator tree.
    pub fn dom_children(&self, bb: BlockId) -> &[BlockId] {
        self.id_to_idx
            .get(&bb)
            .map_or(&[], |&idx| &self.dom_children[idx])
    }

    /// Dominance frontier of `bb`.
    pub fn df(&self, bb: BlockId) -> &[BlockId] {
        self.id_to_idx.get(&bb).map_or(&[], |&idx| &self.df[idx])
    }

    /// Returns `true` if reachable block `a` dominates reachable block `b`.
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        let Some(&a_idx) = self.id_to_idx.get(&a) else {
            return false;
        };
        let Some(&b_idx) = self.id_to_idx.get(&b) else {
            return false;
        };
        dominates_idx(a_idx, b_idx, self.entry_idx, &self.reachable, &self.idom_idx)
    }

    /// Returns `true` if `a != b` and `a` dominates `b`.
    pub fn strictly_dominates(&self, a: BlockId, b: BlockId) -> bool {
        a != b && self.dominates(a, b)
    }

    pub fn natural_loops(&self) -> &[NaturalLoop] {
        &self.loops
    }
}

fn intersect(
    mut b1: usize,
    mut b2: usize,
    idom: &[Option<usize>],
    rpo_num: &[Option<usize>],
) -> usize {
    while b1 != b2 {
        while rpo_num[b1].expect("reachable") > rpo_num[b2].expect("reachable") {
            b1 = idom[b1].expect("idom exists");
        }
        while rpo_num[b2].expect("reachable") > rpo_num[b1].expect("reachable") {
            b2 = idom[b2].expect("idom exists");
        }
    }
    b1
}

fn dominates_idx(
    a_idx: usize,
    b_idx: usize,
    entry_idx: Option<usize>,
    reachable: &[bool],
    idom_idx: &[Option<usize>],
) -> bool {
    let Some(entry) = entry_idx else {
        return false;
    };
    if !reachable[a_idx] || !reachable[b_idx] {
        return false;
    }
    if a_idx == b_idx || a_idx == entry {
        return true;
    }
    let mut cur = b_idx;
    while cur != entry {
        if cur == a_idx {
            return true;
        }
        let Some(parent) = idom_idx[cur] else {
            return false;
        };
        cur = parent;
    }
    cur == a_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BasicBlock, IrType};

    fn make_fn(blocks: Vec<BasicBlock>) -> IrFunction {
        IrFunction {
            id: 0,
            name: "test".into(),
            is_pub: false,
            param_count: 0,
            locals: vec![],
            ret_ty: IrType::Unit,
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            owner_state: None,
            blocks,
            next_val: 1,
        }
    }

    #[test]
    fn diamond_cfg_domtree_and_df() {
        // bb0 -> bb1 | bb2; bb1 -> bb3; bb2 -> bb3; bb3 -> return
        let func = make_fn(vec![
            BasicBlock {
                id: 0,
                insts: vec![],
                term: Terminator::Branch {
                    cond: 0,
                    then_bb: 1,
                    else_bb: 2,
                },
            },
            BasicBlock {
                id: 1,
                insts: vec![],
                term: Terminator::Goto(3),
            },
            BasicBlock {
                id: 2,
                insts: vec![],
                term: Terminator::Goto(3),
            },
            BasicBlock {
                id: 3,
                insts: vec![],
                term: Terminator::Return(None),
            },
        ]);

        let cfg = Cfg::build(&func);
        assert_eq!(cfg.idom(0), None);
        assert_eq!(cfg.idom(1), Some(0));
        assert_eq!(cfg.idom(2), Some(0));
        assert_eq!(cfg.idom(3), Some(0));

        assert!(cfg.dominates(0, 3));
        assert!(!cfg.dominates(1, 3));
        assert!(!cfg.dominates(2, 3));

        assert_eq!(cfg.df(0), &[] as &[BlockId]);
        assert_eq!(cfg.df(1), &[3]);
        assert_eq!(cfg.df(2), &[3]);
        assert_eq!(cfg.df(3), &[] as &[BlockId]);
        assert!(cfg.natural_loops().is_empty());
    }

    #[test]
    fn loop_cfg_domtree_df_and_natural_loop() {
        // bb0 -> bb1 (header)
        // bb1 -> bb2 (body) | bb3 (exit)
        // bb2 -> bb1 (latch back-edge)
        // bb3 -> return
        let func = make_fn(vec![
            BasicBlock {
                id: 0,
                insts: vec![],
                term: Terminator::Goto(1),
            },
            BasicBlock {
                id: 1,
                insts: vec![],
                term: Terminator::Branch {
                    cond: 0,
                    then_bb: 2,
                    else_bb: 3,
                },
            },
            BasicBlock {
                id: 2,
                insts: vec![],
                term: Terminator::Goto(1),
            },
            BasicBlock {
                id: 3,
                insts: vec![],
                term: Terminator::Return(None),
            },
        ]);

        let cfg = Cfg::build(&func);
        assert_eq!(cfg.idom(0), None);
        assert_eq!(cfg.idom(1), Some(0));
        assert_eq!(cfg.idom(2), Some(1));
        assert_eq!(cfg.idom(3), Some(1));

        assert_eq!(cfg.df(1), &[1]);
        assert_eq!(cfg.df(2), &[1]);

        let loops = cfg.natural_loops();
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].header, 1);
        assert_eq!(loops[0].latch, 2);
        assert_eq!(loops[0].body, BTreeSet::from([1, 2]));
    }
}
