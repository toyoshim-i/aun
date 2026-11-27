//! Module tree resolution, visibility checking (`pub`), `use` binding, and AST flattening.

use crate::ast::*;
use crate::diag::Diagnostic;
use aun_ir::Span;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DefKind {
    Fn,
    ExternFn,
    Struct,
    Enum,
    Machine,
    Static,
    ExternStatic,
    Const,
}

#[derive(Debug, Clone)]
struct DefEntry {
    def_kind: DefKind,
    is_pub: bool,
    mangled: String,
}

#[derive(Debug, Clone)]
enum ImportTarget {
    Def(DefEntry),
    Mod { mod_id: usize, is_pub: bool },
}

impl ImportTarget {
    fn import_is_pub(&self) -> bool {
        match self {
            ImportTarget::Def(d) => d.is_pub,
            ImportTarget::Mod { is_pub, .. } => *is_pub,
        }
    }
}

#[derive(Debug)]
struct ModuleNode {
    parent: Option<usize>,
    path_prefix: String,
    is_pub: bool,
    /// Unit root modules keep top-level `pub fn` unmangled and `is_pub = true` for C/WPO exports.
    is_unit_root: bool,
    children: HashMap<String, usize>,
    child_order: Vec<usize>,
    defs: HashMap<String, DefEntry>,
    imports: HashMap<String, ImportTarget>,
    uses: Vec<UseDecl>,
    items: Vec<Item>,
}

enum Resolved {
    Mod(usize),
    Def(DefEntry),
}

enum PathLookup {
    Found(Resolved),
    StoppedAtDef {
        entry: DefEntry,
        remaining: Vec<String>,
    },
    PrivacyError(String),
    NotFound,
}

struct Resolver {
    nodes: Vec<ModuleNode>,
    has_virtual_root: bool,
    diagnostics: Vec<Diagnostic>,
}

impl Resolver {
    fn new(has_virtual_root: bool) -> Self {
        Self {
            nodes: Vec::new(),
            has_virtual_root,
            diagnostics: Vec::new(),
        }
    }

    fn is_ancestor(&self, ancestor: usize, mut cur: usize) -> bool {
        loop {
            if cur == ancestor {
                return true;
            }
            match self.nodes[cur].parent {
                Some(p) => cur = p,
                None => return false,
            }
        }
    }

    fn can_access(&self, from_mod: usize, owner_mod: usize, is_pub: bool) -> bool {
        is_pub || self.is_ancestor(owner_mod, from_mod)
    }

    fn mangle(&self, mod_id: usize, name: &str, is_extern: bool, is_pub_fn: bool) -> String {
        if is_extern {
            return name.to_string();
        }
        let node = &self.nodes[mod_id];
        if node.path_prefix.is_empty() || (node.is_unit_root && is_pub_fn) {
            name.to_string()
        } else {
            format!("{}__{}", node.path_prefix, name)
        }
    }

    fn add_module(
        &mut self,
        parent: Option<usize>,
        path_prefix: String,
        is_pub: bool,
        is_unit_root: bool,
        raw_items: Vec<Item>,
    ) -> usize {
        let id = self.nodes.len();
        self.nodes.push(ModuleNode {
            parent,
            path_prefix,
            is_pub,
            is_unit_root,
            children: HashMap::new(),
            child_order: Vec::new(),
            defs: HashMap::new(),
            imports: HashMap::new(),
            uses: Vec::new(),
            items: Vec::new(),
        });

        for item in raw_items {
            match item {
                Item::Mod {
                    attrs: _,
                    is_pub: mod_pub,
                    name: mod_name,
                    items: mod_items,
                    span,
                } => {
                    let Some(inner) = mod_items else {
                        self.diagnostics.push(Diagnostic::error(
                            format!("unresolved external module `mod {mod_name};` (only supported when loading files)"),
                            span,
                        ));
                        continue;
                    };
                    if self.nodes[id].children.contains_key(&mod_name) {
                        self.diagnostics.push(Diagnostic::error(
                            format!("duplicate module `{mod_name}`"),
                            span,
                        ));
                        continue;
                    }
                    let child_prefix = if self.nodes[id].path_prefix.is_empty() {
                        mod_name.clone()
                    } else {
                        format!("{}__{}", self.nodes[id].path_prefix, mod_name)
                    };
                    let cid = self.add_module(Some(id), child_prefix, mod_pub, false, inner);
                    self.nodes[id].children.insert(mod_name, cid);
                    self.nodes[id].child_order.push(cid);
                }
                Item::Use(u) => {
                    self.nodes[id].uses.push(u);
                }
                other => {
                    self.register_item_def(id, &other);
                    self.nodes[id].items.push(other);
                }
            }
        }
        id
    }

    fn insert_def(&mut self, mod_id: usize, name: String, kind: DefKind, is_pub: bool) {
        let is_extern = matches!(kind, DefKind::ExternFn | DefKind::ExternStatic);
        let is_pub_fn = is_pub && kind == DefKind::Fn;
        let mangled = self.mangle(mod_id, &name, is_extern, is_pub_fn);
        self.nodes[mod_id].defs.insert(
            name,
            DefEntry {
                def_kind: kind,
                is_pub,
                mangled,
            },
        );
    }

    fn register_item_def(&mut self, mod_id: usize, item: &Item) {
        match item {
            Item::Fn(f) => self.insert_def(mod_id, f.name.clone(), DefKind::Fn, f.is_pub),
            Item::Struct(s) => self.insert_def(mod_id, s.name.clone(), DefKind::Struct, s.is_pub),
            Item::Enum(e) => self.insert_def(mod_id, e.name.clone(), DefKind::Enum, e.is_pub),
            Item::Machine(m) => self.insert_def(mod_id, m.name.clone(), DefKind::Machine, m.is_pub),
            Item::Static(s) => self.insert_def(mod_id, s.name.clone(), DefKind::Static, s.is_pub),
            Item::Const(c) => self.insert_def(mod_id, c.name.clone(), DefKind::Const, c.is_pub),
            Item::ExternAsm(block) => {
                for es in &block.statics {
                    self.insert_def(mod_id, es.name.clone(), DefKind::ExternStatic, true);
                }
                for ef in &block.fns {
                    self.insert_def(mod_id, ef.name.clone(), DefKind::ExternFn, true);
                }
            }
            Item::Target(_) | Item::Include { .. } | Item::Mod { .. } | Item::Use(_) => {}
        }
    }

    fn lookup_in_module(
        &self,
        from_mod: usize,
        cur_mod: usize,
        seg: &str,
        allow_private_imports: bool,
    ) -> Result<Option<Resolved>, String> {
        if let Some(&cid) = self.nodes[cur_mod].children.get(seg) {
            if !self.can_access(from_mod, cur_mod, self.nodes[cid].is_pub) {
                return Err(format!("module `{seg}` is private"));
            }
            return Ok(Some(Resolved::Mod(cid)));
        }
        if let Some(entry) = self.nodes[cur_mod].defs.get(seg) {
            if !self.can_access(from_mod, cur_mod, entry.is_pub) {
                return Err(format!("item `{seg}` is private"));
            }
            return Ok(Some(Resolved::Def(entry.clone())));
        }
        if let Some(imp) = self.nodes[cur_mod].imports.get(seg) {
            let visible = if cur_mod == from_mod && allow_private_imports {
                true
            } else {
                self.can_access(from_mod, cur_mod, imp.import_is_pub())
            };
            if !visible {
                return Err(format!("imported item `{seg}` is private"));
            }
            return Ok(Some(match imp {
                ImportTarget::Def(d) => Resolved::Def(d.clone()),
                ImportTarget::Mod { mod_id, .. } => Resolved::Mod(*mod_id),
            }));
        }
        Ok(None)
    }

    fn resolve_path(&self, from_mod: usize, segs: &[String]) -> PathLookup {
        if segs.is_empty() {
            return PathLookup::NotFound;
        }
        let mut idx = 0usize;
        let mut cur_mod = from_mod;
        let mut explicit_root = false;

        match segs[0].as_str() {
            "crate" => {
                cur_mod = 0;
                idx = 1;
                explicit_root = true;
            }
            "self" => {
                cur_mod = from_mod;
                idx = 1;
                explicit_root = true;
            }
            "super" => {
                let Some(p) = self.nodes[from_mod].parent else {
                    return PathLookup::NotFound;
                };
                cur_mod = p;
                idx = 1;
                explicit_root = true;
            }
            _ => {}
        }

        if idx == segs.len() {
            return PathLookup::Found(Resolved::Mod(cur_mod));
        }

        // First segment for relative path checks current module first, then virtual crate root
        // (for sibling translation units compiled together via `aun build a.aun b.aun`).
        let first = &segs[idx];
        let mut resolved = if explicit_root {
            match self.lookup_in_module(from_mod, cur_mod, first, cur_mod == from_mod) {
                Ok(Some(r)) => r,
                Ok(None) => return PathLookup::NotFound,
                Err(msg) => return PathLookup::PrivacyError(msg),
            }
        } else {
            match self.lookup_in_module(from_mod, cur_mod, first, true) {
                Ok(Some(r)) => r,
                Ok(None) if self.has_virtual_root && cur_mod != 0 => {
                    match self.lookup_in_module(from_mod, 0, first, false) {
                        Ok(Some(r)) => r,
                        Ok(None) => return PathLookup::NotFound,
                        Err(msg) => return PathLookup::PrivacyError(msg),
                    }
                }
                Ok(None) => return PathLookup::NotFound,
                Err(msg) => return PathLookup::PrivacyError(msg),
            }
        };

        idx += 1;
        while idx < segs.len() {
            let next_seg = &segs[idx];
            match resolved {
                Resolved::Mod(mid) => {
                    resolved = match self.lookup_in_module(from_mod, mid, next_seg, mid == from_mod)
                    {
                        Ok(Some(r)) => r,
                        Ok(None) => return PathLookup::NotFound,
                        Err(msg) => return PathLookup::PrivacyError(msg),
                    };
                    idx += 1;
                }
                Resolved::Def(entry) => {
                    return PathLookup::StoppedAtDef {
                        entry,
                        remaining: segs[idx..].to_vec(),
                    };
                }
            }
        }

        PathLookup::Found(resolved)
    }

    fn resolve_use_tree(
        &self,
        from_mod: usize,
        prefix: &[String],
        tree: &UseTree,
        is_pub: bool,
        out: &mut Vec<(String, ImportTarget)>,
        errs: &mut Vec<(String, Span)>,
        span: Span,
    ) {
        match tree {
            UseTree::Path { seg, rest } => {
                let mut next = prefix.to_vec();
                next.push(seg.clone());
                self.resolve_use_tree(from_mod, &next, rest, is_pub, out, errs, span);
            }
            UseTree::Group(items) => {
                for sub in items {
                    self.resolve_use_tree(from_mod, prefix, sub, is_pub, out, errs, span);
                }
            }
            UseTree::Glob => {
                let target_mod = if prefix.is_empty() {
                    from_mod
                } else {
                    match self.resolve_path(from_mod, prefix) {
                        PathLookup::Found(Resolved::Mod(mid)) => mid,
                        PathLookup::PrivacyError(msg) => {
                            errs.push((msg, span));
                            return;
                        }
                        _ => {
                            errs.push((
                                format!("unresolved glob import `{}::*`", prefix.join("::")),
                                span,
                            ));
                            return;
                        }
                    }
                };
                let target = &self.nodes[target_mod];
                for (cname, &cid) in &target.children {
                    if self.can_access(from_mod, target_mod, self.nodes[cid].is_pub) {
                        out.push((
                            cname.clone(),
                            ImportTarget::Mod {
                                mod_id: cid,
                                is_pub,
                            },
                        ));
                    }
                }
                for (dname, entry) in &target.defs {
                    if self.can_access(from_mod, target_mod, entry.is_pub) {
                        let mut imported = entry.clone();
                        imported.is_pub = is_pub;
                        out.push((dname.clone(), ImportTarget::Def(imported)));
                    }
                }
                for (iname, imp) in &target.imports {
                    if self.can_access(from_mod, target_mod, imp.import_is_pub()) {
                        let cloned = match imp {
                            ImportTarget::Def(d) => {
                                let mut d2 = d.clone();
                                d2.is_pub = is_pub;
                                ImportTarget::Def(d2)
                            }
                            ImportTarget::Mod { mod_id, .. } => ImportTarget::Mod {
                                mod_id: *mod_id,
                                is_pub,
                            },
                        };
                        out.push((iname.clone(), cloned));
                    }
                }
            }
            UseTree::Name { name, alias } => {
                let mut full = prefix.to_vec();
                full.push(name.clone());
                let bind_name = if let Some(al) = alias {
                    al.clone()
                } else if name == "self" && !prefix.is_empty() {
                    prefix.last().unwrap().clone()
                } else {
                    name.clone()
                };
                match self.resolve_path(from_mod, &full) {
                    PathLookup::Found(Resolved::Mod(mid)) => {
                        out.push((
                            bind_name,
                            ImportTarget::Mod {
                                mod_id: mid,
                                is_pub,
                            },
                        ));
                    }
                    PathLookup::Found(Resolved::Def(mut entry)) => {
                        entry.is_pub = is_pub;
                        out.push((bind_name, ImportTarget::Def(entry)));
                    }
                    PathLookup::PrivacyError(msg) => errs.push((msg, span)),
                    PathLookup::StoppedAtDef { .. } | PathLookup::NotFound => {
                        errs.push((format!("unresolved import `{}`", full.join("::")), span));
                    }
                }
            }
        }
    }

    fn resolve_all_uses(&mut self) {
        let mut final_errs: Vec<(String, Span)> = Vec::new();
        for pass in 0..16 {
            let mut changed = false;
            let mut errs = Vec::new();
            let mut updates: Vec<(usize, Vec<(String, ImportTarget)>)> = Vec::new();

            for mid in 0..self.nodes.len() {
                let mut bindings = Vec::new();
                for u in &self.nodes[mid].uses {
                    self.resolve_use_tree(mid, &[], &u.tree, u.is_pub, &mut bindings, &mut errs, u.span);
                }
                updates.push((mid, bindings));
            }

            for (mid, bindings) in updates {
                for (name, target) in bindings {
                    let is_new = !self.nodes[mid].imports.contains_key(&name);
                    if is_new {
                        changed = true;
                    }
                    self.nodes[mid].imports.insert(name, target);
                }
            }

            final_errs = errs;
            if !changed || pass == 15 {
                break;
            }
        }

        for (msg, span) in final_errs {
            self.diagnostics.push(Diagnostic::error(msg, span));
        }
    }

    fn lookup_single_def(&self, mod_id: usize, name: &str) -> Option<DefEntry> {
        if let Some(d) = self.nodes[mod_id].defs.get(name) {
            return Some(d.clone());
        }
        if let Some(ImportTarget::Def(d)) = self.nodes[mod_id].imports.get(name) {
            return Some(d.clone());
        }
        None
    }

    fn rewrite_type(&mut self, mod_id: usize, ty: &mut TypeExpr, span: Span) {
        match ty {
            TypeExpr::Unit | TypeExpr::Never => {}
            TypeExpr::Named(name) => {
                if matches!(
                    name.as_str(),
                    "bool" | "u8" | "i8" | "u16" | "i16" | "u32" | "i32" | "q8_8" | "q10_6" | "Self"
                ) {
                    return;
                }
                if name.contains("::") {
                    let segs: Vec<String> = name.split("::").map(|s| s.to_string()).collect();
                    match self.resolve_path(mod_id, &segs) {
                        PathLookup::Found(Resolved::Def(entry)) => {
                            *name = entry.mangled;
                        }
                        PathLookup::PrivacyError(msg) => {
                            self.diagnostics.push(Diagnostic::error(msg, span));
                        }
                        _ => {}
                    }
                } else if let Some(entry) = self.lookup_single_def(mod_id, name) {
                    *name = entry.mangled;
                }
            }
            TypeExpr::Fixed { base, .. } => self.rewrite_type(mod_id, base, span),
            TypeExpr::Ref { inner, .. }
            | TypeExpr::ConstSlice(inner)
            | TypeExpr::Slice { inner, .. }
            | TypeExpr::Handle(inner) => self.rewrite_type(mod_id, inner, span),
            TypeExpr::Array { elem, .. } | TypeExpr::Pool { elem, .. } => {
                self.rewrite_type(mod_id, elem, span);
            }
        }
    }

    fn rewrite_expr(&mut self, mod_id: usize, expr: &mut Expr, locals: &[HashSet<String>]) {
        let span = expr.span;
        match &mut expr.kind {
            ExprKind::IntLit { .. }
            | ExprKind::FixedLit { .. }
            | ExprKind::BoolLit(_)
            | ExprKind::ByteLit(_)
            | ExprKind::StrLit(_)
            | ExprKind::ByteStrLit(_) => {}
            ExprKind::Ident(name) => {
                let shadowed = locals.iter().rev().any(|s| s.contains(name));
                if !shadowed && let Some(entry) = self.lookup_single_def(mod_id, name) {
                    *name = entry.mangled;
                }
            }
            ExprKind::Path(segs) => {
                if segs.len() == 2 && segs[1] == "from_raw" {
                    if !matches!(segs[0].as_str(), "q8_8" | "q10_6")
                        && let Some(entry) = self.lookup_single_def(mod_id, &segs[0])
                    {
                        segs[0] = entry.mangled;
                    }
                    return;
                }
                match self.resolve_path(mod_id, segs) {
                    PathLookup::Found(Resolved::Def(entry)) => {
                        expr.kind = ExprKind::Ident(entry.mangled);
                    }
                    PathLookup::StoppedAtDef { entry, remaining } => {
                        if matches!(entry.def_kind, DefKind::Enum | DefKind::Machine)
                            && remaining.len() == 1
                        {
                            *segs = vec![entry.mangled, remaining[0].clone()];
                        } else if remaining.is_empty() {
                            expr.kind = ExprKind::Ident(entry.mangled);
                        }
                    }
                    PathLookup::PrivacyError(msg) => {
                        self.diagnostics.push(Diagnostic::error(msg, span));
                    }
                    PathLookup::Found(Resolved::Mod(_)) | PathLookup::NotFound => {}
                }
            }
            ExprKind::Unary { expr: inner, .. } | ExprKind::Deref(inner) => {
                self.rewrite_expr(mod_id, inner, locals);
            }
            ExprKind::Ref { expr: inner, .. } => {
                self.rewrite_expr(mod_id, inner, locals);
            }
            ExprKind::Binary { lhs, rhs, .. } => {
                self.rewrite_expr(mod_id, lhs, locals);
                self.rewrite_expr(mod_id, rhs, locals);
            }
            ExprKind::Cast { expr: inner, ty } => {
                self.rewrite_expr(mod_id, inner, locals);
                self.rewrite_type(mod_id, ty, span);
            }
            ExprKind::Call { callee, args } => {
                self.rewrite_expr(mod_id, callee, locals);
                for a in args {
                    self.rewrite_expr(mod_id, a, locals);
                }
            }
            ExprKind::MethodCall { receiver, args, .. } => {
                self.rewrite_expr(mod_id, receiver, locals);
                for a in args {
                    self.rewrite_expr(mod_id, a, locals);
                }
            }
            ExprKind::Field { base, .. } => {
                self.rewrite_expr(mod_id, base, locals);
            }
            ExprKind::Index { base, index } => {
                self.rewrite_expr(mod_id, base, locals);
                self.rewrite_expr(mod_id, index, locals);
            }
            ExprKind::StructLit { name, fields } => {
                if name.contains("::") {
                    let segs: Vec<String> = name.split("::").map(|s| s.to_string()).collect();
                    match self.resolve_path(mod_id, &segs) {
                        PathLookup::Found(Resolved::Def(entry)) => {
                            *name = entry.mangled;
                        }
                        PathLookup::PrivacyError(msg) => {
                            self.diagnostics.push(Diagnostic::error(msg, span));
                        }
                        _ => {}
                    }
                } else if let Some(entry) = self.lookup_single_def(mod_id, name) {
                    *name = entry.mangled;
                }
                for (_, val) in fields {
                    self.rewrite_expr(mod_id, val, locals);
                }
            }
            ExprKind::ArrayLit(elems) => {
                for e in elems {
                    self.rewrite_expr(mod_id, e, locals);
                }
            }
            ExprKind::ArrayRepeat { val, .. } => {
                self.rewrite_expr(mod_id, val, locals);
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => {
                self.rewrite_expr(mod_id, cond, locals);
                let mut sub = locals.to_vec();
                self.rewrite_block(mod_id, then_block, &mut sub);
                if let Some(eb) = else_branch {
                    self.rewrite_expr(mod_id, eb, locals);
                }
            }
            ExprKind::Block(blk) | ExprKind::WithBank { body: blk, .. } => {
                let mut sub = locals.to_vec();
                self.rewrite_block(mod_id, blk, &mut sub);
            }
            ExprKind::Goto { fields, .. } => {
                for (_, val) in fields {
                    self.rewrite_expr(mod_id, val, locals);
                }
            }
            ExprKind::IsState {
                expr: inner,
                machine_name,
                ..
            } => {
                self.rewrite_expr(mod_id, inner, locals);
                if machine_name.contains("::") {
                    let segs: Vec<String> =
                        machine_name.split("::").map(|s| s.to_string()).collect();
                    match self.resolve_path(mod_id, &segs) {
                        PathLookup::Found(Resolved::Def(entry)) => {
                            *machine_name = entry.mangled;
                        }
                        PathLookup::PrivacyError(msg) => {
                            self.diagnostics.push(Diagnostic::error(msg, span));
                        }
                        _ => {}
                    }
                } else if let Some(entry) = self.lookup_single_def(mod_id, machine_name) {
                    *machine_name = entry.mangled;
                }
            }
            ExprKind::Assert { cond } => {
                self.rewrite_expr(mod_id, cond, locals);
            }
            ExprKind::InlineAsm { .. } => {}
        }
    }

    fn rewrite_block(&mut self, mod_id: usize, blk: &mut Block, locals: &mut Vec<HashSet<String>>) {
        locals.push(HashSet::new());
        for stmt in &mut blk.stmts {
            match stmt {
                Stmt::Let {
                    name,
                    ty,
                    init,
                    span,
                    ..
                } => {
                    if let Some(t) = ty {
                        self.rewrite_type(mod_id, t, *span);
                    }
                    self.rewrite_expr(mod_id, init, locals);
                    if let Some(top) = locals.last_mut() {
                        top.insert(name.clone());
                    }
                }
                Stmt::Assign { target, value, .. } => {
                    self.rewrite_expr(mod_id, target, locals);
                    self.rewrite_expr(mod_id, value, locals);
                }
                Stmt::Expr(e) | Stmt::Return(Some(e), _) => {
                    self.rewrite_expr(mod_id, e, locals);
                }
                Stmt::Return(None, _) | Stmt::Break(_) | Stmt::Continue(_) => {}
                Stmt::While { cond, body, .. } => {
                    self.rewrite_expr(mod_id, cond, locals);
                    self.rewrite_block(mod_id, body, locals);
                }
                Stmt::Loop { body, .. } => {
                    self.rewrite_block(mod_id, body, locals);
                }
                Stmt::For {
                    var,
                    start,
                    end,
                    body,
                    ..
                } => {
                    self.rewrite_expr(mod_id, start, locals);
                    self.rewrite_expr(mod_id, end, locals);
                    locals.push(HashSet::from([var.clone()]));
                    self.rewrite_block(mod_id, body, locals);
                    locals.pop();
                }
            }
        }
        if let Some(tail) = &mut blk.tail {
            self.rewrite_expr(mod_id, tail, locals);
        }
        locals.pop();
    }

    fn rewrite_fn(&mut self, mod_id: usize, f: &mut FnDecl, include_self: bool, keep_pub: bool) {
        f.name = self.mangle(mod_id, &f.name, false, f.is_pub);
        f.is_pub = keep_pub && f.is_pub;
        self.rewrite_fn_body_only(mod_id, f, include_self);
    }

    fn flatten_node(
        &mut self,
        mod_id: usize,
        out_items: &mut Vec<Item>,
        seen_extern_fns: &mut HashSet<String>,
        seen_extern_statics: &mut HashSet<String>,
        seen_pub_fns: &mut HashMap<String, Span>,
    ) {
        let mut items = std::mem::take(&mut self.nodes[mod_id].items);
        let is_unit_root = self.nodes[mod_id].is_unit_root;

        for item in &mut items {
            match item {
                Item::Fn(f) => {
                    if is_unit_root
                        && f.is_pub
                        && let Some(prev) = seen_pub_fns.insert(f.name.clone(), f.span)
                    {
                        let _ = prev;
                        self.diagnostics.push(Diagnostic::error(
                            format!("duplicate public function `{}` across modules", f.name),
                            f.span,
                        ));
                    }
                    self.rewrite_fn(mod_id, f, false, is_unit_root);
                }
                Item::Struct(s) => {
                    s.name = self.mangle(mod_id, &s.name, false, false);
                    for (_, fty, fspan) in &mut s.fields {
                        self.rewrite_type(mod_id, fty, *fspan);
                    }
                }
                Item::Enum(e) => {
                    e.name = self.mangle(mod_id, &e.name, false, false);
                }
                Item::Machine(m) => {
                    m.name = self.mangle(mod_id, &m.name, false, false);
                    for (_, fty, fspan) in &mut m.shared_fields {
                        self.rewrite_type(mod_id, fty, *fspan);
                    }
                    for st in &mut m.states {
                        for (_, fty, fspan) in &mut st.fields {
                            self.rewrite_type(mod_id, fty, *fspan);
                        }
                    }
                    for method in &mut m.methods {
                        self.rewrite_fn_body_only(mod_id, method, true);
                    }
                    for ob in &mut m.on_blocks {
                        for method in &mut ob.methods {
                            self.rewrite_fn_body_only(mod_id, method, true);
                        }
                    }
                }
                Item::Static(s) => {
                    s.name = self.mangle(mod_id, &s.name, false, false);
                    self.rewrite_type(mod_id, &mut s.ty, s.span);
                    if let Some(init) = &mut s.init {
                        self.rewrite_expr(mod_id, init, &[]);
                    }
                }
                Item::Const(c) => {
                    c.name = self.mangle(mod_id, &c.name, false, false);
                    self.rewrite_type(mod_id, &mut c.ty, c.span);
                    self.rewrite_expr(mod_id, &mut c.value, &[]);
                }
                Item::ExternAsm(block) => {
                    for es in &mut block.statics {
                        self.rewrite_type(mod_id, &mut es.ty, es.span);
                    }
                    for ef in &mut block.fns {
                        for p in &mut ef.params {
                            self.rewrite_type(mod_id, &mut p.ty, p.span);
                        }
                        if let Some((rty, _)) = &mut ef.ret {
                            self.rewrite_type(mod_id, rty, ef.span);
                        }
                    }
                    block.statics.retain(|es| seen_extern_statics.insert(es.name.clone()));
                    block.fns.retain(|ef| seen_extern_fns.insert(ef.name.clone()));
                }
                Item::Target(_) | Item::Include { .. } | Item::Mod { .. } | Item::Use(_) => {}
            }
        }

        out_items.extend(items);
        let child_order = self.nodes[mod_id].child_order.clone();
        for cid in child_order {
            self.flatten_node(
                cid,
                out_items,
                seen_extern_fns,
                seen_extern_statics,
                seen_pub_fns,
            );
        }
    }

    fn rewrite_fn_body_only(&mut self, mod_id: usize, f: &mut FnDecl, include_self: bool) {
        let mut scope = HashSet::new();
        if include_self {
            scope.insert("self".to_string());
        }
        for p in &mut f.params {
            self.rewrite_type(mod_id, &mut p.ty, p.span);
            scope.insert(p.name.clone());
        }
        if let Some(ret) = &mut f.ret_ty {
            self.rewrite_type(mod_id, ret, f.span);
        }
        let mut locals = vec![scope];
        self.rewrite_block(mod_id, &mut f.body, &mut locals);
    }
}

fn file_has_modules(file: &File) -> bool {
    file.items
        .iter()
        .any(|item| matches!(item, Item::Mod { .. } | Item::Use(_)))
}

/// Resolves inline `mod` trees, `use` imports, qualified paths, and `pub` visibility inside
/// a single parsed `File`, returning a flattened `File` ready for `Checker::check_and_lower`.
pub fn resolve_file(file: File) -> Result<File, Vec<Diagnostic>> {
    if !file_has_modules(&file) {
        return Ok(file);
    }
    let mut resolver = Resolver::new(false);
    resolver.add_module(None, String::new(), true, true, file.items);
    resolver.resolve_all_uses();
    let mut out_items = Vec::new();
    let mut seen_extern_fns = HashSet::new();
    let mut seen_extern_statics = HashSet::new();
    let mut seen_pub_fns = HashMap::new();
    resolver.flatten_node(
        0,
        &mut out_items,
        &mut seen_extern_fns,
        &mut seen_extern_statics,
        &mut seen_pub_fns,
    );
    if resolver.diagnostics.is_empty() {
        Ok(File { items: out_items })
    } else {
        Err(resolver.diagnostics)
    }
}

/// Resolves multiple translation units `(unit_stem, File)` into a single whole-program `File`
/// with private item isolation (`{stem}__{name}`), preserved top-level `pub fn` C ABI exports,
/// cross-unit `{stem}::item` resolution, and deduplicated `extern "asm"` declarations.
pub fn resolve_units(mut units: Vec<(String, File)>) -> Result<File, Vec<Diagnostic>> {
    if units.len() == 1 {
        let (_, file) = units.pop().unwrap();
        return resolve_file(file);
    }
    let mut resolver = Resolver::new(true);
    resolver.nodes.push(ModuleNode {
        parent: None,
        path_prefix: String::new(),
        is_pub: true,
        is_unit_root: false,
        children: HashMap::new(),
        child_order: Vec::new(),
        defs: HashMap::new(),
        imports: HashMap::new(),
        uses: Vec::new(),
        items: Vec::new(),
    });
    let mut used_prefixes: HashSet<String> = HashSet::new();
    for (idx, (stem, file)) in units.into_iter().enumerate() {
        let prefix = if used_prefixes.insert(stem.clone()) {
            stem.clone()
        } else {
            let p = format!("{stem}_{idx}");
            used_prefixes.insert(p.clone());
            p
        };
        let cid = resolver.add_module(Some(0), prefix, true, true, file.items);
        resolver.nodes[0].children.entry(stem).or_insert(cid);
        resolver.nodes[0].child_order.push(cid);
    }
    resolver.resolve_all_uses();
    let mut out_items = Vec::new();
    let mut seen_extern_fns = HashSet::new();
    let mut seen_extern_statics = HashSet::new();
    let mut seen_pub_fns = HashMap::new();
    resolver.flatten_node(
        0,
        &mut out_items,
        &mut seen_extern_fns,
        &mut seen_extern_statics,
        &mut seen_pub_fns,
    );
    if resolver.diagnostics.is_empty() {
        Ok(File { items: out_items })
    } else {
        Err(resolver.diagnostics)
    }
}
