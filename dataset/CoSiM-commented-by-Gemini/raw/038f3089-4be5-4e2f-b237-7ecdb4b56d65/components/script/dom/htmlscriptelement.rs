/*!
This module implements the `HTMLScriptElement` interface, which represents
a `<script>` element in the DOM. It is responsible for fetching, compiling,
and executing scripts.

The `HTMLScriptElement` struct is the main entry point for this module. It
creates a new `HTMLScriptElement` object, which can be inserted into the DOM.
The `HTMLScriptElement` object then handles the loading and execution of the
script specified by its `src` attribute or its text content.

The `ScriptOrigin` struct represents the origin of a script, which can be
either internal (i.e., the script is contained within the document) or
external (i.e., the script is loaded from an external file).

The `ScriptExecution` enum specifies when a script should be executed. It can
be `ASAP` (as soon as possible), `ASAPInOrder` (as soon as possible, in
order), or `Deferred` (after the document has been parsed).

The `HTMLScriptElement` API is defined in the HTML specification:
<https://html.spec.whatwg.org/multipage/scripting.html#the-script-element>
*/
use core::ffi::c_void;
use std::cell::Cell;
use std::fs::read_to_string;
use std::path::PathBuf;
use std::process::Command;
use std::ptr;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use base::id::{PipelineId, WebViewId};
use content_security_policy as csp;
use devtools_traits::{ScriptToDevtoolsControlMsg, SourceInfo};
use dom_struct::dom_struct;
use encoding_rs::Encoding;
use html5ever::serialize::TraversalScope;
use html5ever::{LocalName, Prefix, local_name, namespace_url, ns};
use ipc_channel::ipc;
use js::jsval::UndefinedValue;
use js::rust::{CompileOptionsWrapper, HandleObject, Stencil, transform_str_to_source_text};
use net_traits::http_status::HttpStatus;
use net_traits::policy_container::PolicyContainer;
use net_traits::request::{
    CorsSettings, CredentialsMode, Destination, InsecureRequestsPolicy, ParserMetadata,
    RequestBuilder, RequestId,
};
use net_traits::{
    FetchMetadata, FetchResponseListener, IpcSend, Metadata, NetworkError, ResourceFetchTiming,
    ResourceTimingType,
};
use servo_config::pref;
use servo_url::{ImmutableOrigin, ServoUrl};
use style::attr::AttrValue;
use style::str::{HTML_SPACE_CHARACTERS, StaticStringVec};
use stylo_atoms::Atom;
use uuid::Uuid;

use crate::HasParent;
use crate::document_loader::LoadType;
use crate::dom::activation::Activatable;
use crate::dom::attr::Attr;
use crate::dom::bindings::cell::DomRefCell;
use crate::dom::bindings::codegen::Bindings::DocumentBinding::DocumentMethods;
use crate::dom::bindings::codegen::Bindings::HTMLScriptElementBinding::HTMLScriptElementMethods;
use crate::dom::bindings::codegen::Bindings::NodeBinding::NodeMethods;
use crate::dom::bindings::codegen::GenericBindings::HTMLElementBinding::HTMLElement_Binding::HTMLElementMethods;
use crate::dom::bindings::codegen::UnionTypes::{
    TrustedScriptOrString, TrustedScriptURLOrUSVString,
};
use crate::dom::bindings::error::{Error, Fallible};
use crate::dom::bindings::inheritance::Castable;
use crate::dom::bindings::refcounted::Trusted;
use crate::dom::bindings::reflector::DomGlobal;
use crate::dom::bindings::root::{Dom, DomRoot, MutNullableDom};
use crate::dom::bindings::settings_stack::AutoEntryScript;
use crate::dom::bindings::str::{DOMString, USVString};
use crate::dom::bindings::trace::NoTrace;
use crate::dom::csp::report_csp_violations;
use crate::dom::document::Document;
use crate::dom::element::{
    AttributeMutation, Element, ElementCreator, cors_setting_for_element,
    referrer_policy_for_element, reflect_cross_origin_attribute, reflect_referrer_policy_attribute,
    set_cross_origin_attribute,
};
use crate::dom::event::{Event, EventBubbles, EventCancelable, EventStatus};
use crate::dom::globalscope::GlobalScope;
use crate::dom::htmlelement::HTMLElement;
use crate::dom::node::{ChildrenMutation, CloneChildrenFlag, Node, NodeTraits};
use crate::dom::performanceresourcetiming::InitiatorType;
use crate::dom::trustedscript::TrustedScript;
use crate::dom::trustedscripturl::TrustedScriptURL;
use crate::dom::virtualmethods::VirtualMethods;
use crate::dom::window::Window;
use crate::fetch::{create_a_potential_cors_request, load_whole_resource};
use crate::network_listener::{self, NetworkListener, PreInvoke, ResourceTimingListener};
use crate::realms::enter_realm;
use crate::script_module::{
    ImportMap, ModuleOwner, ScriptFetchOptions, fetch_external_module_script,
    fetch_inline_module_script, parse_an_import_map_string, register_import_map,
};
use crate::script_runtime::CanGc;
use crate::task_source::{SendableTaskSource, TaskSourceName};
use crate::unminify::{ScriptSource, unminify_js};

impl ScriptSource for ScriptOrigin {
    fn unminified_dir(&self) -> Option<String> {
        self.unminified_dir.clone()
    }

    fn extract_bytes(&self) -> &[u8] {
        match &self.code {
            SourceCode::Text(text) => text.as_bytes(),
            SourceCode::Compiled(compiled_source_code) => {
                compiled_source_code.original_text.as_bytes()
            },
        }
    }

    fn rewrite_source(&mut self, source: Rc<DOMString>) {
        self.code = SourceCode::Text(source);
    }

    fn url(&self) -> ServoUrl {
        self.url.clone()
    }

    fn is_external(&self) -> bool {
        self.external
    }
}

// TODO Implement offthread compilation in mozjs
/*pub(crate) struct OffThreadCompilationContext {
    script_element: Trusted<HTMLScriptElement>,
    script_kind: ExternalScriptKind,
    final_url: ServoUrl,
    url: ServoUrl,
    task_source: TaskSource,
    script_text: String,
    fetch_options: ScriptFetchOptions,
}

#[allow(unsafe_code)]
unsafe extern "C" fn off_thread_compilation_callback(
    token: *mut OffThreadToken,
    callback_data: *mut c_void,
) {
    let mut context = Box::from_raw(callback_data as *mut OffThreadCompilationContext);
    let token = OffThreadCompilationToken(token);

    let url = context.url.clone();
    let final_url = context.final_url.clone();
    let script_element = context.script_element.clone();
    let script_kind = context.script_kind;
    let script = std::mem::take(&mut context.script_text);
    let fetch_options = context.fetch_options.clone();

    // Continue with <https://html.spec.whatwg.org/multipage/#fetch-a-classic-script>
    let _ = context.task_source.queue(
        task!(off_thread_compile_continue: move || {
            let elem = script_element.root();
            let global = elem.global();
            let cx = GlobalScope::get_cx();
            let _ar = enter_realm(&*global);

            // TODO: This is necessary because the rust compiler will otherwise try to move the *mut
            // OffThreadToken directly, which isn't marked as Send. The correct fix is that this
            // type is marked as Send in mozjs.
            let used_token = token;

            let compiled_script = FinishOffThreadStencil(*cx, used_token.0, ptr::null_mut());
            let load = if compiled_script.is_null() {
                Err(NoTrace(NetworkError::Internal(
                    "Off-thread compilation failed.".into(),
                )))
            } else {
                let script_text = DOMString::from(script);
                let code = SourceCode::Compiled(CompiledSourceCode {
                    source_code: compiled_script,
                    original_text: Rc::new(script_text),
                });

                Ok(ScriptOrigin {
                    code,
                    url: final_url,
                    external: true,
                    fetch_options,
                    type_: ScriptType::Classic,
                })
            };

            finish_fetching_a_classic_script(&elem, script_kind, url, load);
        })
    );
}*/

/// An unique id for script element.
#[derive(Clone, Copy, Debug, Eq, Hash, JSTraceable, PartialEq)]
pub(crate) struct ScriptId(#[no_trace] Uuid);

/// The `HTMLScriptElement` struct.
#[dom_struct]
pub(crate) struct HTMLScriptElement {
    htmlelement: HTMLElement,

    /// <https://html.spec.whatwg.org/multipage/#already-started>
    already_started: Cell<bool>,

    /// <https://html.spec.whatwg.org/multipage/#parser-inserted>
    parser_inserted: Cell<bool>,

    /// <https://html.spec.whatwg.org/multipage/#non-blocking>
    ///
    /// (currently unused)
    non_blocking: Cell<bool>,

    /// Document of the parser that created this element
    /// <https://html.spec.whatwg.org/multipage/#parser-document>
    parser_document: Dom<Document>,

    /// Prevents scripts that move between documents during preparation from executing.
    /// <https://html.spec.whatwg.org/multipage/#preparation-time-document>
    preparation_time_document: MutNullableDom<Document>,

    /// Track line line_number
    line_number: u64,

    /// Unique id for each script element
    #[ignore_malloc_size_of = "Defined in uuid"]
    id: ScriptId,

    /// <https://w3c.github.io/trusted-types/dist/spec/#htmlscriptelement-script-text>
    script_text: DomRefCell<DOMString>,
}

impl HTMLScriptElement {
    fn new_inherited(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
        creator: ElementCreator,
    ) -> HTMLScriptElement {
        HTMLScriptElement {
            id: ScriptId(Uuid::new_v4()),
            htmlelement: HTMLElement::new_inherited(local_name, prefix, document),
            already_started: Cell::new(false),
            parser_inserted: Cell::new(creator.is_parser_created()),
            non_blocking: Cell::new(!creator.is_parser_created()),
            parser_document: Dom::from_ref(document),
            preparation_time_document: MutNullableDom::new(None),
            line_number: creator.return_line_number(),
            script_text: DomRefCell::new(DOMString::new()),
        }
    }

    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn new(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
        proto: Option<HandleObject>,
        creator: ElementCreator,
        can_gc: CanGc,
    ) -> DomRoot<HTMLScriptElement> {
        Node::reflect_node_with_proto(
            Box::new(HTMLScriptElement::new_inherited(
                local_name, prefix, document, creator,
            )),
            document,
            proto,
            can_gc,
        )
    }

    pub(crate) fn get_script_id(&self) -> ScriptId {
        self.id
    }
}

/// Supported script types as defined by
/// <https://html.spec.whatwg.org/multipage/#javascript-mime-type>.
pub(crate) static SCRIPT_JS_MIMES: StaticStringVec = &[
    "application/ecmascript",
    "application/javascript",
    "application/x-ecmascript",
    "application/x-javascript",
    "text/ecmascript",
    "text/javascript",
    "text/javascript1.0",
    "text/javascript1.1",
    "text/javascript1.2",
    "text/javascript1.3",
    "text/javascript1.4",
    "text/javascript1.5",
    "text/jscript",
    "text/livescript",
    "text/x-ecmascript",
    "text/x-javascript",
];

/// The type of a script.
#[derive(Clone, Copy, JSTraceable, MallocSizeOf, PartialEq)]
pub(crate) enum ScriptType {
    /// A classic script.
    Classic,
    /// A module script.
    Module,
    /// An import map.
    ImportMap,
}

/// The source code of a script.
#[derive(JSTraceable, MallocSizeOf)]
pub(crate) struct CompiledSourceCode {
    #[ignore_malloc_size_of = "SM handles JS values"]
    /// The compiled source code.
    pub(crate) source_code: Stencil,
    #[conditional_malloc_size_of = "Rc is hard"]
    /// The original text of the source code.
    pub(crate) original_text: Rc<DOMString>,
}

/// The source code of a script.
#[derive(JSTraceable, MallocSizeOf)]
pub(crate) enum SourceCode {
    /// The source code as text.
    Text(#[conditional_malloc_size_of] Rc<DOMString>),
    /// The compiled source code.
    Compiled(CompiledSourceCode),
}

/// The origin of a script.
#[derive(JSTraceable, MallocSizeOf)]
pub(crate) struct ScriptOrigin {
    code: SourceCode,
    #[no_trace]
    url: ServoUrl,
    external: bool,
    fetch_options: ScriptFetchOptions,
    type_: ScriptType,
    unminified_dir: Option<String>,
    import_map: Fallible<ImportMap>,
}

impl ScriptOrigin {
    /// Creates a new internal `ScriptOrigin`.
    pub(crate) fn internal(
        text: Rc<DOMString>,
        url: ServoUrl,
        fetch_options: ScriptFetchOptions,
        type_: ScriptType,
        unminified_dir: Option<String>,
        import_map: Fallible<ImportMap>,
    ) -> ScriptOrigin {
        ScriptOrigin {
            code: SourceCode::Text(text),
            url,
            external: false,
            fetch_options,
            type_,
            unminified_dir,
            import_map,
        }
    }

    /// Creates a new external `ScriptOrigin`.
    pub(crate) fn external(
        text: Rc<DOMString>,
        url: ServoUrl,
        fetch_options: ScriptFetchOptions,
        type_: ScriptType,
        unminified_dir: Option<String>,
    ) -> ScriptOrigin {
        ScriptOrigin {
            code: SourceCode::Text(text),
            url,
            external: true,
            fetch_options,
            type_,
            unminified_dir,
            import_map: Err(Error::NotFound),
        }
    }

    /// Returns the text of the script.
    pub(crate) fn text(&self) -> Rc<DOMString> {
        match &self.code {
            SourceCode::Text(text) => Rc::clone(text),
            SourceCode::Compiled(compiled_script) => Rc::clone(&compiled_script.original_text),
        }
    }
}

/// Final steps of <https://html.spec.whatwg.org/multipage/#prepare-the-script-element>
fn finish_fetching_a_classic_script(
    elem: &HTMLScriptElement,
    script_kind: ExternalScriptKind,
    url: ServoUrl,
    load: ScriptResult,
    can_gc: CanGc,
) {
    // Step 33. The "steps to run when the result is ready" for each type of script in 33.2-33.5.
    // of https://html.spec.whatwg.org/multipage/#prepare-the-script-element
    let document;

    match script_kind {
        ExternalScriptKind::Asap => {
            document = elem.preparation_time_document.get().unwrap();
            document.asap_script_loaded(elem, load, can_gc)
        },
        ExternalScriptKind::AsapInOrder => {
            document = elem.preparation_time_document.get().unwrap();
            document.asap_in_order_script_loaded(elem, load, can_gc)
        },
        ExternalScriptKind::Deferred => {
            document = elem.parser_document.as_rooted();
            document.deferred_script_loaded(elem, load, can_gc);
        },
        ExternalScriptKind::ParsingBlocking => {
            document = elem.parser_document.as_rooted();
            document.pending_parsing_blocking_script_loaded(elem, load, can_gc);
        },
    }

    document.finish_load(LoadType::Script(url), can_gc);
}

/// The result of a script load.
pub(crate) type ScriptResult = Result<ScriptOrigin, NoTrace<NetworkError>>;

/// The context required for asynchronously loading an external script source.
struct ClassicContext {
    /// The element that initiated the request.
    elem: Trusted<HTMLScriptElement>,
    /// The kind of external script.
    kind: ExternalScriptKind,
    /// The (fallback) character encoding argument to the "fetch a classic
    /// script" algorithm.
    character_encoding: &'static Encoding,
    /// The response body received to date.
    data: Vec<u8>,
    /// The response metadata received to date.
    metadata: Option<Metadata>,
    /// The initial URL requested.
    url: ServoUrl,
    /// Indicates whether the request failed, and why
    status: Result<(), NetworkError>,
    /// The fetch options of the script
    fetch_options: ScriptFetchOptions,
    /// Timing object for this resource
    resource_timing: ResourceFetchTiming,
}

impl FetchResponseListener for ClassicContext {
    // TODO(KiChjang): Perhaps add custom steps to perform fetch here?
    fn process_request_body(&mut self, _: RequestId) {}

    // TODO(KiChjang): Perhaps add custom steps to perform fetch here?
    fn process_request_eof(&mut self, _: RequestId) {}

    fn process_response(&mut self, _: RequestId, metadata: Result<FetchMetadata, NetworkError>) {
        self.metadata = metadata.ok().map(|meta| match meta {
            FetchMetadata::Unfiltered(m) => m,
            FetchMetadata::Filtered { unsafe_, .. } => unsafe_,
        });

        let status = self
            .metadata
            .as_ref()
            .map(|m| m.status.clone())
            .unwrap_or_else(HttpStatus::new_error);

        self.status = {
            if status.is_error() {
                Err(NetworkError::Internal(
                    "No http status code received".to_owned(),
                ))
            } else if status.is_success() {
                Ok(())
            } else {
                Err(NetworkError::Internal(format!(
                    "HTTP error code {}",
                    status.code()
                )))
            }
        };
    }

    fn process_response_chunk(&mut self, _: RequestId, mut chunk: Vec<u8>) {
        if self.status.is_ok() {
            self.data.append(&mut chunk);
        }
    }

    /// <https://html.spec.whatwg.org/multipage/#fetch-a-classic-script>
    /// step 4-9
    #[allow(unsafe_code)]
    fn process_response_eof(
        &mut self,
        _: RequestId,
        response: Result<ResourceFetchTiming, NetworkError>,
    ) {
        let (source_text, final_url) = match (response.as_ref(), self.status.as_ref()) {
            (Err(err), _) | (_, Err(err)) => {
                // Step 6, response is an error.
                finish_fetching_a_classic_script(
                    &self.elem.root(),
                    self.kind,
                    self.url.clone(),
                    Err(NoTrace(err.clone())),
                    CanGc::note(),
                );
                return;
            },
            (Ok(_), Ok(_)) => {
                let metadata = self.metadata.take().unwrap();

                // Step 7.
                let encoding = metadata
                    .charset
                    .and_then(|encoding| Encoding::for_label(encoding.as_bytes()))
                    .unwrap_or(self.character_encoding);

                // Step 8.
                let (source_text, _, _) = encoding.decode(&self.data);
                (source_text, metadata.final_url)
            },
        };

        let elem = self.elem.root();
        let global = elem.global();
        //let cx = GlobalScope::get_cx();
        let _ar = enter_realm(&*global);

        /*
        let options = unsafe { CompileOptionsWrapper::new(*cx, final_url.as_str(), 1) };

        let can_compile_off_thread = pref!(dom_script_asynch) &&
            unsafe { CanCompileOffThread(*cx, options.ptr as *const _, source_text.len()) };

        if can_compile_off_thread {
            let source_string = source_text.to_string();

            let context = Box::new(OffThreadCompilationContext {
                script_element: self.elem.clone(),
                script_kind: self.kind,
                final_url,
                url: self.url.clone(),
                task_source: elem.owner_global().task_manager().dom_manipulation_task_source(),
                script_text: source_string,
                fetch_options: self.fetch_options.clone(),
            });

            unsafe {
                assert!(!CompileToStencilOffThread1(
                    *cx,
                    options.ptr as *const _,
                    &mut transform_str_to_source_text(&context.script_text) as *mut _,
                    Some(off_thread_compilation_callback),
                    Box::into_raw(context) as *mut c_void,
                )
                .is_null());
            }
        } else {*/
        let load = ScriptOrigin::external(
            Rc::new(DOMString::from(source_text)),
            final_url.clone(),
            self.fetch_options.clone(),
            ScriptType::Classic,
            elem.parser_document.global().unminified_js_dir(),
        );
        finish_fetching_a_classic_script(
            &elem,
            self.kind,
            self.url.clone(),
            Ok(load),
            CanGc::note(),
        );
        //}
    }

    fn resource_timing_mut(&mut self) -> &mut ResourceFetchTiming {
        &mut self.resource_timing
    }

    fn resource_timing(&self) -> &ResourceFetchTiming {
        &self.resource_timing
    }

    fn submit_resource_timing(&mut self) {
        network_listener::submit_timing(self, CanGc::note())
    }

    fn process_csp_violations(&mut self, _request_id: RequestId, violations: Vec<csp::Violation>) {
        let global = &self.resource_timing_global();
        let elem = self.elem.root();
        report_csp_violations(global, violations, Some(elem.upcast::<Element>()));
    }
}