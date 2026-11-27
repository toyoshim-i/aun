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

pub fn diagnostic_hint(engine: &DocEngine, msg: &str, mode: RenderMode) -> Option<String> {
    let lower = msg.to_ascii_lowercase();
    let code = if lower.contains("mixed-width")
        || (lower.contains("type mismatch")
            && (lower.contains("u8")
                || lower.contains("i8")
                || lower.contains("u16")
                || lower.contains("i16")
                || lower.contains("u32")
                || lower.contains("i32")
                || lower.contains("fixed")))
    {
        "E001"
    } else if lower.contains("second-class")
        || lower.contains("aliasing")
        || lower.contains("mutable borrow")
    {
        "E002"
    } else if lower.contains("clobber") || lower.contains("extern") || lower.contains("sdcccall") {
        "E003"
    } else if lower.contains("cross-overlay")
        || lower.contains("state-scoped")
        || lower.contains("conflicting overlay")
        || lower.contains("exceeds `")
        || lower.contains("owning state")
    {
        "E005"
    } else if lower.contains("bank")
        || lower.contains("requires di")
        || lower.contains("recursion")
        || lower.contains("strict_transitions")
        || lower.contains("dynamic")
    {
        "E004"
    } else {
        return None;
    };
    engine.hint(code, mode)
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
        assert_eq!(guides.len(), 8, "expected 8 guide pages: {guides:?}");
        let rules = engine.list_topics(Some("rule"));
        assert_eq!(rules.len(), 5, "expected 5 rule pages: {rules:?}");

        let overview = engine
            .guide(None, RenderMode::Agent)
            .expect("guide index should render");
        assert!(overview.contains("language-reference"));
        assert!(overview.contains("idiomatic-aun"));
        assert!(overview.contains("mcs51-backend"));

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
