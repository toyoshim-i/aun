//! Embedded agent-native documentation, diagnostic hints, and MCP server powered by `agent-man`.

use agent_man::{CategoryConfig, DocEngine, RenderMode};
use std::io::IsTerminal;

pub fn create_doc_engine() -> DocEngine {
    let mut engine = DocEngine::new("aun")
        .with_version(env!("CARGO_PKG_VERSION"))
        .with_subcommands("guide", "explain")
        .with_scheme("aun")
        .with_categories(vec![
            CategoryConfig::new("guide", "Guides"),
            CategoryConfig::new("rule", "Diagnostic Rules"),
        ]);
    engine.add_pages(include!(concat!(env!("OUT_DIR"), "/agent_man_pages.rs")));
    engine
}

pub fn detect_render_mode(args: &[String]) -> RenderMode {
    for arg in args {
        match arg.as_str() {
            "--json" => return RenderMode::Json,
            "--agent" => return RenderMode::Agent,
            "--terminal" => return RenderMode::Terminal,
            _ => {}
        }
    }
    if std::io::stdout().is_terminal() {
        RenderMode::Terminal
    } else {
        RenderMode::Agent
    }
}

pub fn classify_error_code(msg: &str) -> Option<&'static str> {
    let lower = msg.to_ascii_lowercase();
    if lower.contains("mixed-width")
        || (lower.contains("type mismatch")
            && (lower.contains("u8")
                || lower.contains("i8")
                || lower.contains("u16")
                || lower.contains("i16")
                || lower.contains("u32")
                || lower.contains("i32")
                || lower.contains("fixed")))
    {
        Some("E001")
    } else if lower.contains("second-class")
        || lower.contains("aliasing")
        || lower.contains("mutable borrow")
    {
        Some("E002")
    } else if lower.contains("clobber") || lower.contains("extern") || lower.contains("sdcccall") {
        Some("E003")
    } else if lower.contains("exceeds cycle budget") {
        Some("E008")
    } else if lower.contains("static stack depth")
        || (lower.contains("ram ")
            && (lower.contains("overlaps stack") || lower.contains("exceeds space")))
    {
        Some("E007")
    } else if (lower.contains("overlay `") && lower.contains("exceeds `"))
        || (lower.contains("resident code")
            && (lower.contains("overlaps") || lower.contains("exceeds")))
    {
        Some("E006")
    } else if lower.contains("cross-overlay")
        || lower.contains("state-scoped")
        || lower.contains("conflicting overlay")
        || lower.contains("owning state")
    {
        Some("E005")
    } else if lower.contains("bank")
        || lower.contains("requires di")
        || lower.contains("recursion")
        || lower.contains("strict_transitions")
        || lower.contains("dynamic")
    {
        Some("E004")
    } else {
        None
    }
}

pub fn diagnostic_hint(engine: &DocEngine, msg: &str, mode: RenderMode) -> Option<String> {
    let code = classify_error_code(msg)?;
    engine.hint(code, mode)
}

/// Prints a backend/linker error with its `error[E00x]:` code (if classified) and
/// actionable CLI/MCP guidance (including `--memory-report` for E006/E007 overflows).
pub fn print_backend_error(msg: &str) {
    let engine = create_doc_engine();
    if let Some(code) = classify_error_code(msg) {
        eprintln!("error[{code}]: {msg}");
        match code {
            "E006" | "E007" => {
                eprintln!(
                    "  note: full per-state symbol sizes, Resident/LCA promotion callers, and space budgets are available directly from Aun:"
                );
                eprintln!(
                    "    -> re-run with `--memory-report` (prints full layout even when over budget)"
                );
                eprintln!(
                    "    -> run `aun explain {code}` or `aun guide memory-optimization` before inspecting .asm or rewriting to assembly"
                );
            }
            _ => {
                if let Some(hint) = engine.hint(code, RenderMode::Agent) {
                    eprintln!("  {hint}");
                }
            }
        }
    } else {
        eprintln!("error: {msg}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_man::mcp::HandlerResult;

    #[test]
    fn embedded_docs_have_all_guides_and_rules_without_broken_links() {
        let engine = create_doc_engine();
        let broken = engine.broken_links();
        assert!(broken.is_empty(), "broken doc links: {broken:?}");

        let guides = engine.list_topics(Some("guide"));
        assert_eq!(guides.len(), 9, "expected 9 guide pages: {guides:?}");
        let rules = engine.list_topics(Some("rule"));
        assert_eq!(rules.len(), 9, "expected 9 rule pages: {rules:?}");

        let overview = engine
            .guide(None, RenderMode::Agent)
            .expect("guide index should render");
        assert!(overview.contains("language-reference"));
        assert!(overview.contains("idiomatic-aun"));
        assert!(overview.contains("mcs51-backend"));
        assert!(overview.contains("memory-optimization"));

        let mem_guide = engine
            .guide(Some("memory-optimization"), RenderMode::Agent)
            .expect("memory-optimization should render");
        assert!(mem_guide.contains("--memory-report"));

        let e006 = engine
            .explain("E006", RenderMode::Agent)
            .expect("E006 explain should render");
        assert!(e006.contains("--memory-report"));
        assert!(engine.explain("E007", RenderMode::Agent).is_ok());
        assert!(engine.explain("E008", RenderMode::Agent).is_ok());
        assert!(engine.explain("W001", RenderMode::Agent).is_ok());

        let lang = engine
            .guide(Some("language-reference"), RenderMode::Agent)
            .expect("language-reference should render");
        assert!(lang.contains("Second-Class References"));

        let cookbook = engine
            .guide(Some("idiomatic-aun"), RenderMode::Agent)
            .expect("idiomatic-aun should render");
        assert!(cookbook.contains("swap_boundary"));

        let e001 = engine
            .explain("E001", RenderMode::Agent)
            .expect("E001 explain should render");
        assert!(e001.contains("Mixed-Width"));

        let e005 = engine
            .explain("E005", RenderMode::Agent)
            .expect("E005 explain should render");
        assert!(e005.contains("Overlay"));
    }

    #[test]
    fn diagnostic_hints_map_to_rule_pages() {
        let engine = create_doc_engine();
        let h1 = diagnostic_hint(&engine, "mixed-width arithmetic between u8 and u16", RenderMode::Agent);
        assert_eq!(h1.as_deref(), Some("Help: run 'aun explain E001'"));

        let h2 = diagnostic_hint(
            &engine,
            "second-class references (`&T` / `&mut T`) cannot be stored in structs",
            RenderMode::Agent,
        );
        assert_eq!(h2.as_deref(), Some("Help: run 'aun explain E002'"));

        let h3 = diagnostic_hint(&engine, "invalid register in clobbers()", RenderMode::Agent);
        assert_eq!(h3.as_deref(), Some("Help: run 'aun explain E003'"));

        let h4 = diagnostic_hint(&engine, "callee requires bank(VramAlu)", RenderMode::Agent);
        assert_eq!(h4.as_deref(), Some("Help: run 'aun explain E004'"));

        let h6 = diagnostic_hint(
            &engine,
            "overlay `InGame::Stage3` exceeds `MainRam` budget (0x5E00, 16593 > 8690 bytes, limit 0x7FFF)",
            RenderMode::Agent,
        );
        assert_eq!(h6.as_deref(), Some("Help: run 'aun explain E006'"));

        let h7 = diagnostic_hint(
            &engine,
            "RAM 0xB800..0xBE10 (1553 bytes) overlaps stack `MainStack` (0xBE00..0xBFFF)",
            RenderMode::Agent,
        );
        assert_eq!(h7.as_deref(), Some("Help: run 'aun explain E007'"));

        let h8 = diagnostic_hint(
            &engine,
            "function `isr_vblank` exceeds cycle budget: worst-case 1200 T-states > limit 800 T-states",
            RenderMode::Agent,
        );
        assert_eq!(h8.as_deref(), Some("Help: run 'aun explain E008'"));
    }

    #[test]
    fn mcp_tools_list_exposes_doc_tools() {
        let engine = create_doc_engine();
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#;
        let HandlerResult::Ready(Some(resp)) = engine.handle(req) else {
            panic!("expected synchronous Ready JSON-RPC response");
        };
        assert!(resp.contains("doc_list"), "response: {resp}");
        assert!(resp.contains("doc_read"), "response: {resp}");
    }
}
