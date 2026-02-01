/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/// @file htmlcollection.rs
/// @brief This file implements the HTMLCollection interface, which represents a generic
/// live collection of HTML elements. It provides methods for accessing elements by index
/// or by name/ID, and automatically updates itself when the underlying document changes.
/// Functional Utility: Offers a dynamic, iterable view of a filtered subset of DOM elements.

use std::cell::Cell;
use std::cmp::Ordering;

use dom_struct::dom_struct; // Macro for defining DOM structures.
use html5ever::{LocalName, QualName, local_name, namespace_url, ns}; // HTML5 parsing types.
use style::str::split_html_space_chars; // Utility for splitting strings by HTML whitespace characters.
use stylo_atoms::Atom; // Atomized strings for efficiency.

use crate::dom::bindings::codegen::Bindings::HTMLCollectionBinding::HTMLCollectionMethods; // Generated bindings for HTMLCollection methods.
use crate::dom::bindings::inheritance::Castable; // Trait for safe downcasting between DOM types.
use crate::dom::bindings::reflector::{Reflector, reflect_dom_object}; // DOM object reflection.
use crate::dom::bindings::root::{Dom, DomRoot, MutNullableDom}; // Root DOM types.
use crate::dom::bindings::str::DOMString; // DOMString representation.
use crate::dom::bindings::trace::JSTraceable; // Marker trait for JavaScript-traceable types.
use crate::dom::bindings::xmlname::namespace_from_domstring; // Utility for converting DOMString to Namespace.
use crate::dom::element::Element; // Element type.
use crate::dom::node::{Node, NodeTraits}; // Node types and traits.
use crate::dom::window::Window; // Window object.
use crate::script_runtime::CanGc; // Marker trait for types that can be garbage collected.

/// @trait CollectionFilter
/// @brief A trait for defining custom filtering logic for elements within an HTMLCollection.
/// Functional Utility: Allows HTMLCollections to be specialized for different selection criteria
/// (e.g., by tag name, class name, or a custom function).
pub(crate) trait CollectionFilter: JSTraceable {
    /// @brief Determines if an element should be included in the collection.
    /// @param elem The `Element` to evaluate.
    /// @param root The root `Node` of the collection's traversal.
    /// @return `true` if the element matches the filter criteria, `false` otherwise.
    fn filter<'a>(&self, elem: &'a Element, root: &'a Node) -> bool;
}

/// @struct OptionU32
/// @brief Represents an optional `u32` value, using `u32::MAX` as a sentinel for `None`.
/// Functional Utility: A space-optimized alternative to `Option<u32>` to avoid word alignment issues.
#[derive(Clone, Copy, JSTraceable, MallocSizeOf)]
struct OptionU32 {
    bits: u32, //!< The `u32` value, or `u32::MAX` if `None`.
}

impl OptionU32 {
    /// @brief Converts the `OptionU32` to a standard `Option<u32>`.
    /// @return `Some(u32)` if the value is not `u32::MAX`, `None` otherwise.
    fn to_option(self) -> Option<u32> {
        if self.bits == u32::MAX {
            None
        } else {
            Some(self.bits)
        }
    }

    /// @brief Creates an `OptionU32` from a `u32` value.
    /// @param bits The `u32` value. Must not be `u32::MAX`.
    /// @return A new `OptionU32` instance.
    fn some(bits: u32) -> OptionU32 {
        assert_ne!(bits, u32::MAX); // Pre-condition: `bits` must not be the sentinel value.
        OptionU32 { bits }
    }

    /// @brief Creates an `OptionU32` representing `None`.
    /// @return A new `OptionU32` instance with `bits` set to `u32::MAX`.
    fn none() -> OptionU32 {
        OptionU32 { bits: u32::MAX }
    }
}

/// @struct HTMLCollection
/// @brief Implements the HTMLCollection interface, providing a live collection of DOM elements.
/// Functional Utility: Dynamically represents a filtered subset of elements within a DOM tree,
/// updating automatically when the underlying DOM changes. It also incorporates caching
/// mechanisms for efficient element access.
///
/// <https://dom.spec.whatwg.org/#htmlcollection>
#[dom_struct]
pub(crate) struct HTMLCollection {
    reflector_: Reflector, //!< DOM object reflection data.
    root: Dom<Node>, //!< The root node from which elements are collected.
    #[ignore_malloc_size_of = "Trait object (Box<dyn CollectionFilter>) cannot be sized"]
    filter: Box<dyn CollectionFilter + 'static>, //!< The filter applied to select elements.
    // We cache the version of the root node and all its decendents,
    // the length of the collection, and a cursor into the collection.
    // FIXME: make the cached cursor element a weak pointer
    cached_version: Cell<u64>, //!< Cached version of the root node's inclusive descendants for cache invalidation.
    cached_cursor_element: MutNullableDom<Element>, //!< Cached element at the last accessed cursor position.
    cached_cursor_index: Cell<OptionU32>, //!< Cached index of the last accessed cursor position.
    cached_length: Cell<OptionU32>, //!< Cached length of the collection.
}

impl HTMLCollection {
    /// @brief Creates a new `HTMLCollection` instance with inherited properties.
    /// Functional Utility: Constructor for internal use, allowing initialization with a specific filter.
    ///
    /// @param root The root `Node` from which to collect elements.
    /// @param filter The `CollectionFilter` to apply.
    /// @return A new `HTMLCollection` instance.
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn new_inherited(
        root: &Node,
        filter: Box<dyn CollectionFilter + 'static>,
    ) -> HTMLCollection {
        HTMLCollection {
            reflector_: Reflector::new(),
            root: Dom::from_ref(root),
            filter,
            // Default values for the cache
            cached_version: Cell::new(root.inclusive_descendants_version()), // Initialize cache version.
            cached_cursor_element: MutNullableDom::new(None), // Initialize cached cursor element.
            cached_cursor_index: Cell::new(OptionU32::none()), // Initialize cached cursor index.
            cached_length: Cell::new(OptionU32::none()), // Initialize cached length.
        }
    }

    /// @brief Returns a collection which is always empty.
    /// Functional Utility: Provides a singleton-like empty collection, useful for cases
    /// where no elements match (e.g., an empty class list filter).
    ///
    /// @param window The `Window` object.
    /// @param root The root `Node`.
    /// @param can_gc A `CanGc` token.
    /// @return An empty `HTMLCollection`.
    pub(crate) fn always_empty(window: &Window, root: &Node, can_gc: CanGc) -> DomRoot<Self> {
        #[derive(JSTraceable, MallocSizeOf)]
        struct NoFilter;
        impl CollectionFilter for NoFilter {
            /// @brief Filter that always returns `false`, ensuring an empty collection.
            fn filter<'a>(&self, _: &'a Element, _: &'a Node) -> bool {
                false
            }
        }

        Self::new(window, root, Box::new(NoFilter), can_gc)
    }

    /// @brief Creates a new `HTMLCollection` instance.
    /// Functional Utility: Public constructor that reflects the new HTMLCollection object
    /// into the DOM tree for JavaScript accessibility.
    ///
    /// @param window The `Window` object.
    /// @param root The root `Node` from which to collect elements.
    /// @param filter The `CollectionFilter` to apply.
    /// @param can_gc A `CanGc` token.
    /// @return A new `HTMLCollection` instance wrapped in `DomRoot`.
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn new(
        window: &Window,
        root: &Node,
        filter: Box<dyn CollectionFilter + 'static>,
        can_gc: CanGc,
    ) -> DomRoot<Self> {
        reflect_dom_object(Box::new(Self::new_inherited(root, filter)), window, can_gc)
    }

    /// @brief Create a new [`HTMLCollection`] that just filters element using a static function.
    /// Functional Utility: Provides a convenient way to create an `HTMLCollection` with a custom
    /// filtering logic defined by a function pointer.
    ///
    /// @param window The `Window` object.
    /// @param root The root `Node` for the collection.
    /// @param filter_function The static function to use for filtering.
    /// @param can_gc A `CanGc` token.
    /// @return A new `HTMLCollection` instance wrapped in `DomRoot`.
    pub(crate) fn new_with_filter_fn(
        window: &Window,
        root: &Node,
        filter_function: fn(&Element, &Node) -> bool,
        can_gc: CanGc,
    ) -> DomRoot<Self> {
        #[derive(JSTraceable, MallocSizeOf)]
        pub(crate) struct StaticFunctionFilter(
            // The function *must* be static so that it never holds references to DOM objects, which
            // would cause issues with garbage collection -- since it isn't traced.
            #[no_trace]
            #[ignore_malloc_size_of = "Static function pointer"]
            fn(&Element, &Node) -> bool,
        );
        impl CollectionFilter for StaticFunctionFilter {
            /// @brief Filters an element using the stored static function.
            fn filter(&self, element: &Element, root: &Node) -> bool {
                (self.0)(element, root)
            }
        }
        Self::new(
            window,
            root,
            Box::new(StaticFunctionFilter(filter_function)),
            can_gc,
        )
    }

    /// @brief Creates a new `HTMLCollection` instance.
    /// Functional Utility: Convenience method for creating HTMLCollections.
    ///
    /// @param window The `Window` object.
    /// @param root The root `Node` for the collection.
    /// @param filter The `CollectionFilter` to apply.
    /// @param can_gc A `CanGc` token.
    /// @return A new `HTMLCollection` instance wrapped in `DomRoot`.
    pub(crate) fn create(
        window: &Window,
        root: &Node,
        filter: Box<dyn CollectionFilter + 'static>,
        can_gc: CanGc,
    ) -> DomRoot<Self> {
        Self::new(window, root, filter, can_gc)
    }

    /// @brief Validates and invalidates the cache if the underlying DOM has changed.
    /// Functional Utility: Ensures the live nature of HTMLCollection by comparing
    /// the cached DOM version with the current one, clearing the cache if outdated.
    fn validate_cache(&self) {
        // Clear the cache if the root version is different from our cached version
        let cached_version = self.cached_version.get();
        let curr_version = self.root.inclusive_descendants_version();
        // Block Logic: If the current DOM version is different from the cached version, invalidate the cache.
        if curr_version != cached_version {
            // Default values for the cache
            self.cached_version.set(curr_version); // Update cached version.
            self.cached_cursor_element.set(None); // Clear cached cursor element.
            self.cached_length.set(OptionU32::none()); // Clear cached length.
            self.cached_cursor_index.set(OptionU32::none()); // Clear cached cursor index.
        }
    }

    /// @brief Sets the cached cursor position and element.
    /// Functional Utility: Updates the internal cache used for optimizing sequential access
    /// to elements within the collection.
    ///
    /// @param index The index to cache.
    /// @param element The `DomRoot<Element>` to cache.
    /// @return The cached element, or `None` if no element was provided.
    fn set_cached_cursor(
        &self,
        index: u32,
        element: Option<DomRoot<Element>>,
    ) -> Option<DomRoot<Element>> {
        // Block Logic: If an element is provided, update the cached cursor with its index and reference.
        if let Some(element) = element {
            self.cached_cursor_index.set(OptionU32::some(index)); // Set cached index.
            self.cached_cursor_element.set(Some(&element)); // Set cached element.
            Some(element)
        } else {
            None
        }
    }

    /// @brief Creates an `HTMLCollection` that filters elements by qualified name.
    /// Functional Utility: Implements the `getElementsByTagName` behavior, supporting
    /// wildcard tag names and handling HTML document-specific case sensitivity.
    /// <https://dom.spec.whatwg.org/#concept-getelementsbytagname>
    ///
    /// @param window The `Window` object.
    /// @param root The root `Node` for the collection.
    /// @param qualified_name The `LocalName` (tag name) to filter by.
    /// @param can_gc A `CanGc` token.
    /// @return A new `HTMLCollection` instance wrapped in `DomRoot`.
    pub(crate) fn by_qualified_name(
        window: &Window,
        root: &Node,
        qualified_name: LocalName,
        can_gc: CanGc,
    ) -> DomRoot<HTMLCollection> {
        // case 1: If the qualified name is "*", return a collection with no filter (matches all elements).
        if qualified_name == local_name!("*") {
            #[derive(JSTraceable, MallocSizeOf)]
            struct AllFilter;
            impl CollectionFilter for AllFilter {
                /// @brief Filter that always returns `true`, matching all elements.
                fn filter(&self, _elem: &Element, _root: &Node) -> bool {
                    true
                }
            }
            return HTMLCollection::create(window, root, Box::new(AllFilter), can_gc);
        }

        // Block Logic: Define a filter for HTML documents, handling case sensitivity.
        #[derive(JSTraceable, MallocSizeOf)]
        struct HtmlDocumentFilter {
            #[no_trace]
            qualified_name: LocalName, // Original qualified name.
            #[no_trace]
            ascii_lower_qualified_name: LocalName, // Lowercase version for HTML document matching.
        }
        impl CollectionFilter for HtmlDocumentFilter {
            /// @brief Filters an element based on its qualified name and document type.
            fn filter(&self, elem: &Element, root: &Node) -> bool {
                if root.is_in_html_doc() && elem.namespace() == &ns!(html) {
                    // case 2: For HTML documents, match case-insensitively.
                    HTMLCollection::match_element(elem, &self.ascii_lower_qualified_name)
                } else {
                    // case 2 and 3: For non-HTML documents or elements not in HTML namespace, match case-sensitively.
                    HTMLCollection::match_element(elem, &self.qualified_name)
                }
            }
        }

        let filter = HtmlDocumentFilter {
            ascii_lower_qualified_name: qualified_name.to_ascii_lowercase(), // Store lowercase version.
            qualified_name,
        };
        HTMLCollection::create(window, root, Box::new(filter), can_gc)
    }

    /// @brief Matches an element against a qualified name, considering potential XML prefixes.
    /// Functional Utility: Helper for `getElementsByTagName` to correctly compare element names
    /// with the provided qualified name.
    ///
    /// @param elem The `Element` to match.
    /// @param qualified_name The `LocalName` to match against.
    /// @return `true` if the element's name matches the qualified name, `false` otherwise.
    fn match_element(elem: &Element, qualified_name: &LocalName) -> bool {
        match elem.prefix().as_ref() {
            None => elem.local_name() == qualified_name, // No prefix, direct local name comparison.
            Some(prefix) => {
                // If there's a prefix, check if qualified name starts with it and has correct format.
                qualified_name.starts_with(&**prefix) &&
                    qualified_name.find(':') == Some(prefix.len()) &&
                    qualified_name.ends_with(&**elem.local_name())
            },
        }
    }

    /// @brief Creates an `HTMLCollection` that filters elements by tag name and namespace.
    /// Functional Utility: Implements the `getElementsByTagNameNS` behavior, allowing
    /// precise selection of elements across different XML namespaces.
    ///
    /// @param window The `Window` object.
    /// @param root The root `Node` for the collection.
    /// @param tag The tag name as a `DOMString`.
    /// @param maybe_ns An `Option<DOMString>` representing the namespace URI, or `None`.
    /// @param can_gc A `CanGc` token.
    /// @return A new `HTMLCollection` instance wrapped in `DomRoot`.
    pub(crate) fn by_tag_name_ns(
        window: &Window,
        root: &Node,
        tag: DOMString,
        maybe_ns: Option<DOMString>,
        can_gc: CanGc,
    ) -> DomRoot<HTMLCollection> {
        let local = LocalName::from(tag); // Convert tag to LocalName.
        let ns = namespace_from_domstring(maybe_ns); // Convert namespace DOMString to Namespace.
        let qname = QualName::new(None, ns, local); // Create a Qualified Name.
        HTMLCollection::by_qual_tag_name(window, root, qname, can_gc) // Delegate to by_qual_tag_name.
    }

    /// @brief Creates an `HTMLCollection` that filters elements by qualified tag name.
    /// Functional Utility: Provides the core filtering logic for tag name and namespace matching,
    /// supporting wildcard matching for both.
    ///
    /// @param window The `Window` object.
    /// @param root The root `Node` for the collection.
    /// @param qname The `QualName` (qualified name) to filter by.
    /// @param can_gc A `CanGc` token.
    /// @return A new `HTMLCollection` instance wrapped in `DomRoot`.
    pub(crate) fn by_qual_tag_name(
        window: &Window,
        root: &Node,
        qname: QualName,
        can_gc: CanGc,
    ) -> DomRoot<HTMLCollection> {
        #[derive(JSTraceable, MallocSizeOf)]
        struct TagNameNSFilter {
            #[no_trace]
            qname: QualName, // The qualified name to match.
        }
        impl CollectionFilter for TagNameNSFilter {
            /// @brief Filters an element based on its qualified name matching the stored `qname`.
            /// Functional Utility: Handles wildcard matching for both namespace and local name.
            fn filter(&self, elem: &Element, _root: &Node) -> bool {
                // Check if namespace matches (or is wildcard) AND local name matches (or is wildcard).
                ((self.qname.ns == namespace_url!("*")) || (self.qname.ns == *elem.namespace())) &&
                    ((self.qname.local == local_name!("*")) ||
                        (self.qname.local == *elem.local_name()))
            }
        }
        let filter = TagNameNSFilter { qname }; // Create filter with the qualified name.
        HTMLCollection::create(window, root, Box::new(filter), can_gc) // Create HTMLCollection with this filter.
    }

    /// @brief Creates an `HTMLCollection` that filters elements by class name.
    /// Functional Utility: Implements the `getElementsByClassName` behavior,
    /// allowing selection of elements based on their assigned CSS classes.
    ///
    /// @param window The `Window` object.
    /// @param root The root `Node` for the collection.
    /// @param classes The class names as a `DOMString` (space-separated).
    /// @param can_gc A `CanGc` token.
    /// @return A new `HTMLCollection` instance wrapped in `DomRoot`.
    pub(crate) fn by_class_name(
        window: &Window,
        root: &Node,
        classes: DOMString,
        can_gc: CanGc,
    ) -> DomRoot<HTMLCollection> {
        // Split class names by whitespace and atomize them.
        let class_atoms = split_html_space_chars(&classes).map(Atom::from).collect();
        HTMLCollection::by_atomic_class_name(window, root, class_atoms, can_gc) // Delegate to by_atomic_class_name.
    }

    /// @brief Creates an `HTMLCollection` that filters elements by a vector of atomic class names.
    /// Functional Utility: Core filtering logic for class names, considering document's
    /// quirks mode for case sensitivity.
    ///
    /// @param window The `Window` object.
    /// @param root The root `Node` for the collection.
    /// @param classes A `Vec<Atom>` of class names to match.
    /// @param can_gc A `CanGc` token.
    /// @return A new `HTMLCollection` instance wrapped in `DomRoot`.
    pub(crate) fn by_atomic_class_name(
        window: &Window,
        root: &Node,
        classes: Vec<Atom>,
        can_gc: CanGc,
    ) -> DomRoot<HTMLCollection> {
        #[derive(JSTraceable, MallocSizeOf)]
        struct ClassNameFilter {
            #[no_trace]
            classes: Vec<Atom>, // The list of class atoms to match.
        }
        impl CollectionFilter for ClassNameFilter {
            /// @brief Filters an element to check if it has all specified class names.
            /// Functional Utility: Considers the document's quirks mode to determine
            /// case sensitivity for class matching.
            fn filter(&self, elem: &Element, _root: &Node) -> bool {
                let case_sensitivity = elem
                    .owner_document()
                    .quirks_mode()
                    .classes_and_ids_case_sensitivity();

                // All classes must be present on the element.
                self.classes
                    .iter()
                    .all(|class| elem.has_class(class, case_sensitivity))
            }
        }

        // Block Logic: If no classes are specified, return an always-empty collection.
        if classes.is_empty() {
            return HTMLCollection::always_empty(window, root, can_gc);
        }

        let filter = ClassNameFilter { classes }; // Create filter with the class atoms.
        HTMLCollection::create(window, root, Box::new(filter), can_gc) // Create HTMLCollection.
    }

    /// @brief Creates an `HTMLCollection` representing the immediate children of a node.
    /// Functional Utility: Implements the `children` property of a node, returning
    /// a live collection of its child elements.
    ///
    /// @param window The `Window` object.
    /// @param root The parent `Node` whose children are to be collected.
    /// @param can_gc A `CanGc` token.
    /// @return A new `HTMLCollection` instance wrapped in `DomRoot`.
    pub(crate) fn children(window: &Window, root: &Node, can_gc: CanGc) -> DomRoot<HTMLCollection> {
        HTMLCollection::new_with_filter_fn(
            window,
            root,
            // Filter function: element is a child if its parent is the root.
            |element, root| root.is_parent_of(element.upcast()),
            can_gc,
        )
    }

    /// @brief Returns an iterator over elements in the collection starting after a given node.
    /// Functional Utility: Efficiently iterates through the live collection, continuing
    /// from a specified point in the DOM tree.
    ///
    /// @param after The `Node` after which to start iterating.
    /// @return An iterator yielding `DomRoot<Element>` instances.
    pub(crate) fn elements_iter_after<'a>(
        &'a self,
        after: &'a Node,
    ) -> impl Iterator<Item = DomRoot<Element>> + 'a {
        // Iterate forwards from a node.
        after
            .following_nodes(&self.root) // Get all following nodes from the DOM.
            .filter_map(DomRoot::downcast) // Filter for only Element types.
            .filter(move |element| self.filter.filter(element, &self.root)) // Apply the collection's filter.
    }

    /// @brief Returns an iterator over all elements in the collection from the root.
    /// Functional Utility: Provides a convenient way to iterate through all elements
    /// that match the collection's filter criteria.
    /// @return An iterator yielding `DomRoot<Element>` instances.
    pub(crate) fn elements_iter(&self) -> impl Iterator<Item = DomRoot<Element>> + '_ {
        // Iterate forwards from the root.
        self.elements_iter_after(&self.root)
    }

    /// @brief Returns an iterator over elements in the collection preceding a given node.
    /// Functional Utility: Allows iteration through the live collection in reverse order,
    /// starting before a specified node in the DOM tree.
    ///
    /// @param before The `Node` before which to start iterating.
    /// @return An iterator yielding `DomRoot<Element>` instances.
    pub(crate) fn elements_iter_before<'a>(
        &'a self,
        before: &'a Node,
    ) -> impl Iterator<Item = DomRoot<Element>> + 'a {
        // Iterate backwards from a node.
        before
            .preceding_nodes(&self.root) // Get all preceding nodes from the DOM.
            .filter_map(DomRoot::downcast) // Filter for only Element types.
            .filter(move |element| self.filter.filter(element, &self.root)) // Apply the collection's filter.
    }

    /// @brief Returns the root node of this `HTMLCollection`.
    /// Functional Utility: Provides access to the starting point of the DOM traversal
    /// for this collection.
    /// @return The `DomRoot<Node>` representing the root.
    pub(crate) fn root_node(&self) -> DomRoot<Node> {
        DomRoot::from_ref(&self.root)
    }
}

impl HTMLCollectionMethods<crate::DomTypeHolder> for HTMLCollection {
    /// @brief Returns the number of elements in the collection.
    /// Functional Utility: Implements the `length` property of `HTMLCollection`,
    /// providing the current count of matching elements, leveraging caching for performance.
    /// <https://dom.spec.whatwg.org/#dom-htmlcollection-length>
    ///
    /// @return The number of elements (`u32`).
    fn Length(&self) -> u32 {
        self.validate_cache(); // Ensure the cache is up-to-date.

        // Block Logic: Return cached length if available, otherwise calculate and cache it.
        if let Some(cached_length) = self.cached_length.get().to_option() {
            // Cache hit
            cached_length
        } else {
            // Cache miss, calculate the length
            let length = self.elements_iter().count() as u32; // Count elements by iterating.
            self.cached_length.set(OptionU32::some(length)); // Cache the calculated length.
            length
        }
    }

    /// @brief Returns the element at the specified index in the collection.
    /// Functional Utility: Implements the `item()` method of `HTMLCollection`,
    /// providing indexed access to elements, optimized with a cursor-based cache.
    /// <https://dom.spec.whatwg.org/#dom-htmlcollection-item>
    ///
    /// @param index The zero-based index of the element to retrieve.
    /// @return An `Option<DomRoot<Element>>` containing the element if found, or `None`.
    fn Item(&self, index: u32) -> Option<DomRoot<Element>> {
        self.validate_cache(); // Ensure the cache is up-to-date.

        // Block Logic: Use cursor-based caching to efficiently retrieve elements.
        if let Some(element) = self.cached_cursor_element.get() {
            // Cache hit, the cursor element is set
            if let Some(cached_index) = self.cached_cursor_index.get().to_option() {
                match cached_index.cmp(&index) {
                    Ordering::Equal => {
                        // The cursor is the element we're looking for.
                        Some(element)
                    },
                    Ordering::Less => {
                        // The cursor is before the element we're looking for, iterate forwards.
                        let offset = index - (cached_index + 1); // Calculate offset from cursor.
                        let node: DomRoot<Node> = DomRoot::upcast(element); // Upcast cached element to Node.
                        let mut iter = self.elements_iter_after(&node); // Start iterator after cursor.
                        self.set_cached_cursor(index, iter.nth(offset as usize)) // Move iterator to target index.
                    },
                    Ordering::Greater => {
                        // The cursor is after the element we're looking for, iterate backwards.
                        let offset = cached_index - (index + 1); // Calculate offset from cursor.
                        let node: DomRoot<Node> = DomRoot::upcast(element); // Upcast cached element to Node.
                        let mut iter = self.elements_iter_before(&node); // Start iterator before cursor.
                        self.set_cached_cursor(index, iter.nth(offset as usize)) // Move iterator to target index.
                    },
                }
            } else {
                // Cache miss (cursor element exists, but index is unknown)
                // Iterate forwards through all the nodes.
                self.set_cached_cursor(index, self.elements_iter().nth(index as usize))
            }
        } else {
            // Cache miss (no cursor element set)
            // Iterate forwards through all the nodes from the beginning.
            self.set_cached_cursor(index, self.elements_iter().nth(index as usize))
        }
    }

    /// @brief Returns the first element in the collection that matches the given name or ID.
    /// Functional Utility: Implements the `namedItem()` method of `HTMLCollection`,
    /// providing access to elements by their `id` or `name` attributes.
    /// <https://dom.spec.whatwg.org/#dom-htmlcollection-nameditem>
    ///
    /// @param key The `DOMString` representing the name or ID to search for.
    /// @return An `Option<DomRoot<Element>>` containing the matching element, or `None`.
    fn NamedItem(&self, key: DOMString) -> Option<DomRoot<Element>> {
        // Step 1. If key is an empty string, return None.
        if key.is_empty() {
            return None;
        }

        let key = Atom::from(key); // Convert key to an Atom for efficient comparison.

        // Step 2. Iterate through the elements and find the first one with a matching ID or name.
        self.elements_iter().find(|elem| {
            elem.get_id().is_some_and(|id| id == key) || // Check if element's ID matches.
                (elem.namespace() == &ns!(html) && elem.get_name().is_some_and(|id| id == key)) // Check if element is HTML and its name matches.
        })
    }

    // <https://dom.spec.whatwg.org/#dom-htmlcollection-item>
    /// @brief Implements the indexed getter behavior for `HTMLCollection`.
    /// Functional Utility: Allows accessing elements using array-like syntax (e.g., `collection[0]`).
    /// @param index The zero-based index of the element.
    /// @return An `Option<DomRoot<Element>>` containing the element if found, or `None`.
    fn IndexedGetter(&self, index: u32) -> Option<DomRoot<Element>> {
        self.Item(index) // Delegates to the `Item()` method.
    }

    // check-tidy: no specs after this line
    /// @brief Implements the named getter behavior for `HTMLCollection`.
    /// Functional Utility: Allows accessing elements by their `id` or `name` attributes
    /// using property access syntax (e.g., `collection.myElementId`).
    /// @param name The `DOMString` representing the name or ID.
    /// @return An `Option<DomRoot<Element>>` containing the matching element, or `None`.
    fn NamedGetter(&self, name: DOMString) -> Option<DomRoot<Element>> {
        self.NamedItem(name) // Delegates to the `NamedItem()` method.
    }

    /// @brief Returns a list of all supported property names for the collection.
    /// Functional Utility: Implements the `SupportedPropertyNames` method,
    /// returning a list of all unique IDs and names of elements within the collection.
    /// <https://dom.spec.whatwg.org/#interface-htmlcollection>
    ///
    /// @return A `Vec<DOMString>` containing all supported property names.
    fn SupportedPropertyNames(&self) -> Vec<DOMString> {
        // Step 1: Initialize an empty list for results.
        let mut result = vec![];

        // Step 2: Iterate through all elements in the collection to gather their unique IDs and names.
        for elem in self.elements_iter() {
            // Step 2.1: If the element has an ID, add it to the result list if it's not already present.
            if let Some(id_atom) = elem.get_id() {
                let id_str = DOMString::from(&*id_atom);
                if !result.contains(&id_str) {
                    result.push(id_str);
                }
            }
            // Step 2.2: If the element is in the HTML namespace and has a name, add it to the result list if unique.
            if *elem.namespace() == ns!(html) {
                if let Some(name_atom) = elem.get_name() {
                    let name_str = DOMString::from(&*name_atom);
                    if !result.contains(&name_str) {
                        result.push(name_str)
                    }
                }
            }
        }

        // Step 3: Return the collected unique property names.
        result
    }
}