/**
 * @file languageConfiguration.ts
 * @brief Defines interfaces and classes for configuring language-specific editor features.
 *
 * This module provides the foundational structures for describing how a programming
 * language behaves within the editor environment. It includes rules for comments,
 * bracket matching, word patterns, indentation, and auto-closing pairs. These configurations
 * are crucial for enabling intelligent editor features that enhance the user experience.
 * Domain: Text Editor, Language Services, Configuration, User Experience, Auto-completion.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { CharCode } from '../../../base/common/charCode.js';
import { StandardTokenType } from '../encodedTokenAttributes.js';
import { ScopedLineTokens } from './supports.js';

/**
 * @interface CommentRule
 * @brief Describes the commenting syntax and behavior for a specific programming language.
 *
 * Functional Utility: This interface allows defining how line comments and block comments
 *                     are structured and where they should be positioned within the editor.
 *                     It is crucial for features like automatic comment toggling and
 *                     syntax highlighting of comments.
 */
export interface CommentRule {
	/**
	 * @property lineComment
	 * @brief The token used to initiate a single-line comment (e.g., `//` for C++, `#` for Python).
	 * Functional Utility: Specifies the character sequence that designates the remainder of a line as a comment.
	 *                     Used by editor features for toggling line comments.
	 * @type {string | null}
	 */
	lineComment?: string | null;
	/**
	 * @property lineCommentTokenColumn
	 * @brief Specifies the preferred column at which a line comment token should be inserted.
	 * Functional Utility: Guides the editor's auto-commenting features to maintain consistent code style.
	 *                     If `null` or `undefined`, the comment is inserted at the beginning of the line.
	 * @type {number | null}
	 */
	lineCommentTokenColumn?: number | null;
	/**
	 * @property blockComment
	 * @brief The character pair used to delimit multi-line (block) comments (e.g., `/* ... *&#47;` for C++).
	 * Functional Utility: Defines the start and end tokens for block comments, enabling editor features
	 *                     like block comment toggling and proper syntax highlighting across multiple lines.
	 * @type {CharacterPair | null}
	 */
	blockComment?: CharacterPair | null;
}

/**
 * @interface LanguageConfiguration
 * @brief Defines the comprehensive configuration for a programming language within the editor.
 *
 * Functional Utility: This interface serves as the primary contract between language extensions
 *                     and the editor's core features. It enables a wide range of language-specific
 *                     behaviors, including automatic bracket insertion, intelligent indentation,
 *                     word definition, commenting, and code folding.
 * Domain: Text Editor, Language Services, User Experience, Configuration.
 */
export interface LanguageConfiguration {
	/**
	 * @property comments
	 * @brief Configuration for how comments work in the language.
	 * Functional Utility: Enables editor features like toggling line and block comments,
	 *                     and syntax highlighting for comments.
	 * @type {CommentRule}
	 */
	comments?: CommentRule;
	/**
	 * @property brackets
	 * @brief Defines the language's bracket pairs.
	 * Functional Utility: Used for automatic bracket completion, bracket matching,
	 *                     and implicitly influences indentation around these pairs.
	 * @type {CharacterPair[]}
	 */
	brackets?: CharacterPair[];
	/**
	 * @property wordPattern
	 * @brief A regular expression that defines what constitutes a "word" in the language.
	 * Functional Utility: Essential for word-based navigation (e.g., Ctrl+Left/Right),
	 *                     word selection, and features like rename symbol.
	 *                     It is recommended to use exclusion-based regex for Unicode-rich languages.
	 * @type {RegExp}
	 */
	wordPattern?: RegExp;
	/**
	 * @property indentationRules
	 * @brief Rules governing how indentation changes as the user types or formats code.
	 * Functional Utility: Enables smart indentation, ensuring that code blocks are correctly
	 *                     indented/unindented based on language-specific syntax.
	 * @type {IndentationRule}
	 */
	indentationRules?: IndentationRule;
	/**
	 * @property onEnterRules
	 * @brief A set of rules to be applied when the Enter key is pressed.
	 * Functional Utility: Defines custom indentation and text insertion behavior on Enter,
	 *                     allowing for intelligent auto-formatting (e.g., closing braces, new line for comments).
	 * @type {OnEnterRule[]}
	 */
	onEnterRules?: OnEnterRule[];
	/**
	 * @property autoClosingPairs
	 * @brief Defines pairs of characters that should be automatically closed when the opening character is typed.
	 * Functional Utility: Enhances user typing experience by automatically inserting closing characters
	 *                     (e.g., `"` after typing `"`, `)` after typing `(`).
	 *                     If not explicitly set, `brackets` will be used by default.
	 * @type {IAutoClosingPairConditional[]}
	 */
	autoClosingPairs?: IAutoClosingPairConditional[];
	/**
	 * @property surroundingPairs
	 * @brief Defines character pairs that can surround a selected text.
	 * Functional Utility: When an opening character is typed with text selected, the selection
	 *                     is automatically wrapped by the defined open and close characters.
	 *                     If not explicitly set, `autoClosingPairs` will be used by default.
	 * @type {IAutoClosingPair[]}
	 */
	surroundingPairs?: IAutoClosingPair[];
	/**
	 * @property colorizedBracketPairs
	 * @brief A list of bracket pairs that are colorized based on their nesting level.
	 * Functional Utility: Improves code readability by visually distinguishing nested bracket pairs,
	 *                     making it easier to identify scope and structure.
	 *                     If not explicitly set, `brackets` will be used by default.
	 * @type {CharacterPair[]}
	 */
	colorizedBracketPairs?: CharacterPair[];
	/**
	 * @property autoCloseBefore
	 * @brief Specifies characters that must immediately follow the cursor for auto-closing to occur.
	 * Functional Utility: Prevents unwanted auto-closing in contexts where it would be syntactically incorrect,
	 *                     such as before existing closing brackets or non-unary operators.
	 * @type {string}
	 */
	autoCloseBefore?: string;

	/**
	 * @property folding
	 * @brief Rules defining how code regions can be folded in the editor.
	 * Functional Utility: Enables code folding, allowing users to collapse and expand blocks of code
	 *                     to improve readability and navigate large files more easily.
	 * @type {FoldingRules}
	 */
	folding?: FoldingRules;

	/**
	 * @property __electricCharacterSupport
	 * @brief (Deprecated) Support for "electric characters" which trigger auto-indentation or auto-closing.
	 * Functional Utility: Historically used for specific language features that modify indentation
	 *                     or insert characters when certain keys are typed.
	 * @deprecated Will be replaced by a better API soon.
	 * @type {{ docComment?: IDocComment }}
	 */
	__electricCharacterSupport?: {
		docComment?: IDocComment;
	};
}
/**
 * @internal
 */
type OrUndefined<T> = { [P in keyof T]: T[P] | undefined };

/**
 * @internal
 * @typedef ExplicitLanguageConfiguration
 * @brief Represents a language configuration where all properties of `LanguageConfiguration` are explicitly defined as possibly `undefined`.
 * Functional Utility: Ensures type safety and explicitness when working with potentially partial language configurations,
 *                     providing clarity on which properties might be absent.
 */
export type ExplicitLanguageConfiguration = OrUndefined<Required<LanguageConfiguration>>;

/**
 * @interface IndentationRule
 * @brief Defines a set of regular expression patterns to control the automatic indentation behavior of a language.
 *
 * Functional Utility: These rules enable intelligent auto-indentation in the editor,
 *                     ensuring code consistency and improving readability by automatically
 *                     adjusting indentation levels based on structural patterns within the code.
 */
export interface IndentationRule {
	/**
	 * @property decreaseIndentPattern
	 * @brief A regular expression that, if matched by a line, signals that subsequent lines should be unindented.
	 * Functional Utility: Typically used for closing blocks (e.g., `}` in C-like languages, `end` in Ruby).
	 * @type {RegExp}
	 */
	decreaseIndentPattern: RegExp;
	/**
	 * @property increaseIndentPattern
	 * @brief A regular expression that, if matched by a line, signals that subsequent lines should be indented.
	 * Functional Utility: Typically used for opening blocks (e.g., `{` in C-like languages, `do` in Ruby).
	 * @type {RegExp}
	 */
	increaseIndentPattern: RegExp;
	/**
	 * @property indentNextLinePattern
	 * @brief A regular expression that, if matched by a line, signals that *only the immediate next line* should be indented.
	 * Functional Utility: Useful for specific language constructs where only a single subsequent line
	 *                     requires increased indentation (e.g., `return {` where the `{` is on the next line).
	 * @type {RegExp | null}
	 */
	indentNextLinePattern?: RegExp | null;
	/**
	 * @property unIndentedLinePattern
	 * @brief A regular expression that, if matched by a line, prevents its indentation from being changed by other rules.
	 * Functional Utility: Allows specifying exceptions where a line's indentation should remain fixed,
	 *                     even if it might otherwise be affected by `increaseIndentPattern` or `decreaseIndentPattern`.
	 * @type {RegExp | null}
	 */
	unIndentedLinePattern?: RegExp | null;

}
/**
 * @interface FoldingMarkers
 * @brief Defines language-specific regular expressions for identifying custom code folding regions.
 *
 * Functional Utility: Allows languages to define their own explicit markers (e.g., `#region`, `#endregion`
 *                     or `// #region`, `// #endregion`) for code folding, providing more granular control
 *                     over how code blocks can be collapsed and expanded. These regexes must be efficient
 *                     and typically anchored to the start of a line.
 */
export interface FoldingMarkers {
	/**
	 * @property start
	 * @brief The regular expression that matches the start of a custom folding region.
	 * Functional Utility: Identifies lines that begin a collapsible code block. Must start with `^`.
	 * @type {RegExp}
	 */
	start: RegExp;
	/**
	 * @property end
	 * @brief The regular expression that matches the end of a custom folding region.
	 * Functional Utility: Identifies lines that terminate a collapsible code block. Must start with `^`.
	 * @type {RegExp}
	 */
	end: RegExp;
}

/**
 * @interface FoldingRules
 * @brief Defines how code folding should behave for a specific language.
 *
 * Functional Utility: This interface centralizes settings related to code folding,
 *                     allowing for customization of indentation-based folding logic
 *                     and the definition of explicit folding markers. It directly
 *                     impacts the editor's ability to present a collapsible view of code.
 */
export interface FoldingRules {
	/**
	 * @property offSide
	 * @brief Indicates whether the language adheres to the "off-side rule" for block definition.
	 * Functional Utility: Determines how empty lines are treated within indentation-based folding strategies.
	 *                     If `true`, empty lines between blocks are associated with the next block;
	 *                     otherwise, they are associated with the previous block.
	 * @type {boolean}
	 */
	offSide?: boolean;

	/**
	 * @property markers
	 * @brief Specifies language-specific markers that delineate custom folding regions.
	 * Functional Utility: Allows users to define arbitrary collapsible sections of code using
	 *                     explicit start and end patterns (e.g., `#region`/`#endregion`).
	 * @type {FoldingMarkers}
	 */
	markers?: FoldingMarkers;
}

/**
 * @interface OnEnterRule
 * @brief Defines a specific rule that dictates editor behavior when the Enter key is pressed.
 *
 * Functional Utility: These rules enable smart editor actions such as automatic indentation,
 *                     insertion of closing characters, or specific text manipulation
 *                     based on the surrounding text context. They are crucial for
 *                     maintaining code structure and accelerating coding.
 */
export interface OnEnterRule {
	/**
	 * @property beforeText
	 * @brief A regular expression that must match the text immediately preceding the cursor on the current line.
	 * Functional Utility: Specifies the preceding text context required for this rule to be applied.
	 * @type {RegExp}
	 */
	beforeText: RegExp;
	/**
	 * @property afterText
	 * @brief A regular expression that must match the text immediately following the cursor on the current line.
	 * Functional Utility: Specifies the subsequent text context required for this rule to be applied.
	 *                     Often used to prevent auto-actions if an expected closing character already exists.
	 * @type {RegExp}
	 */
	afterText?: RegExp;
	/**
	 * @property previousLineText
	 * @brief A regular expression that must match the text of the line immediately above the current line.
	 * Functional Utility: Allows rules to be context-sensitive to the preceding line's content,
	 *                     useful for multi-line constructs (e.g., doc comments, string literals).
	 * @type {RegExp}
	 */
	previousLineText?: RegExp;
	/**
	 * @property action
	 * @brief The action to perform when this rule is triggered.
	 * Functional Utility: Defines how indentation should be adjusted and if any specific text
	 *                     should be appended or removed on pressing Enter.
	 * @type {EnterAction}
	 */
	action: EnterAction;
}
/**
 * @interface IDocComment
 * @brief Defines the structural properties of documentation comments (doc comments) for a language.
 *
 * Functional Utility: This interface provides a standardized way to configure how doc comments
 *                     are recognized and handled by the editor, which is essential for features
 *                     like auto-completion within doc comments, syntax highlighting, and formatting.
 * Domain: Language Services, Documentation, Editor Features.
 */
export interface IDocComment {
	/**
	 * @property open
	 * @brief The character sequence that marks the beginning of a documentation comment block (e.g., `/**` for Javadoc/JSDoc).
	 * Functional Utility: Used by the editor to identify the start of a doc comment, enabling
	 *                     specialized parsing and formatting within the block.
	 * @type {string}
	 */
	open: string;
	/**
	 * @property close
	 * @brief The character sequence that marks the end of a documentation comment block (e.g., `*&#47;`).
	 * Functional Utility: Used by the editor to identify the end of a doc comment, completing the block.
	 * @type {string}
	 */
	close?: string;
}
/**
 * @typedef CharacterPair
 * @brief Represents a pair of characters, typically an opening and closing delimiter.
 *
 * Functional Utility: Used to define bracket pairs, block comment delimiters,
 *                     or other symmetric character constructs within a language.
 *                     Crucial for features like bracket matching, auto-closing,
 *                     and surrounding selections.
 * @type {[string, string]}
 */
export type CharacterPair = [string, string];

/**
 * @interface IAutoClosingPair
 * @brief Represents a basic pair of opening and closing characters for auto-closing functionality.
 *
 * Functional Utility: Defines the fundamental structure for auto-closing pairs,
 *                     specifying the characters that should trigger the automatic
 *                     insertion of their counterparts.
 */
export interface IAutoClosingPair {
	/**
	 * @property open
	 * @brief The opening character(s) of the pair (e.g., `(` or `"`)
	 * @type {string}
	 */
	open: string;
	/**
	 * @property close
	 * @brief The closing character(s) of the pair (e.g., `)` or `"`)
	 * @type {string}
	 */
	close: string;
}

/**
 * @interface IAutoClosingPairConditional
 * @brief Extends `IAutoClosingPair` to include conditions under which auto-closing should *not* occur.
 *
 * Functional Utility: Provides granular control over auto-closing behavior, allowing auto-closing
 *                     to be disabled in specific token types (e.g., within strings or comments)
 *                     to prevent unwanted or incorrect insertions.
 * @augments IAutoClosingPair
 */
export interface IAutoClosingPairConditional extends IAutoClosingPair {
	/**
	 * @property notIn
	 * @brief An optional array of token types (e.g., 'string', 'comment', 'regex') where auto-closing should be suppressed.
	 * Functional Utility: Prevents auto-closing from happening in contexts where it would be inappropriate or create syntax errors.
	 * @type {string[]}
	 */
	notIn?: string[];
}

/**
 * @enum IndentAction
 * @brief Specifies the type of indentation adjustment to apply when the Enter key is pressed.
 *
 * Functional Utility: These actions are used in conjunction with `OnEnterRule` to precisely
 *                     control the editor's automatic indentation behavior, ensuring code
 *                     formatting remains consistent and intuitive for the user.
 */
export enum IndentAction {
	/**
	 * @member IndentAction.None
	 * @brief Inserts a new line and maintains the indentation level of the previous line.
	 * Functional Utility: Copies the existing indentation to the new line without modification.
	 */
	None = 0,
	/**
	 * @member IndentAction.Indent
	 * @brief Inserts a new line and increases the indentation level once relative to the previous line.
	 * Functional Utility: Automatically indents the new line, typically after an opening brace or block start.
	 */
	Indent = 1,
	/**
	 * @member IndentAction.IndentOutdent
	 * @brief Inserts two new lines; the first is indented and contains the cursor, the second is at the previous indentation level.
	 * Functional Utility: Designed for constructs like `if {}` blocks where the closing brace should align with the opening,
	 *                     and the cursor should be placed inside the block.
	 */
	IndentOutdent = 2,
	/**
	 * @member IndentAction.Outdent
	 * @brief Inserts a new line and decreases the indentation level once relative to the previous line.
	 * Functional Utility: Automatically outdents the new line, typically after a closing brace or block end.
	 */
	Outdent = 3
}

/**
 * @interface EnterAction
 * @brief Defines the specific actions to be performed when the Enter key is pressed in the editor.
 *
 * Functional Utility: This interface orchestrates various automatic behaviors
 *                     upon pressing Enter, such as adjusting indentation, inserting
 *                     boilerplate text, or modifying the current line's indentation.
 *                     It is a core component of intelligent editor functionality
 *                     that adapts to language syntax.
 */
export interface EnterAction {
	/**
	 * @property indentAction
	 * @brief Specifies how the indentation level of the new line should be adjusted.
	 * Functional Utility: Directs the editor to either maintain, increase, decrease,
	 *                     or perform a complex indent/outdent operation.
	 * @type {IndentAction}
	 */
	indentAction: IndentAction;
	/**
	 * @property appendText
	 * @brief Optional text to be appended after the new line and its computed indentation.
	 * Functional Utility: Automatically inserts common boilerplate text, such as closing braces
	 *                     or continuation characters for multi-line strings/comments.
	 * @type {string}
	 */
	appendText?: string;
	/**
	 * @property removeText
	 * @brief Optional number of characters to remove from the new line's automatically computed indentation.
	 * Functional Utility: Allows fine-grained control over indentation, for example, to outdent a closing brace
	 *                     that is part of an `IndentOutdent` action.
	 * @type {number}
	 */
	removeText?: number;
}
/**
 * @interface CompleteEnterAction
 * @internal
 * @brief Represents a fully resolved `EnterAction`, with all optional properties explicitly defined.
 *
 * Functional Utility: This internal interface ensures that all necessary action details
 *                     are available for the editor's Enter key handling logic,
 *                     providing a consistent and complete set of instructions for
 *                     indentation, text insertion, and indentation adjustment.
 */
export interface CompleteEnterAction {
	/**
	 * @property indentAction
	 * @brief The resolved indentation action to perform.
	 * @type {IndentAction}
	 */
	indentAction: IndentAction;
	/**
	 * @property appendText
	 * @brief The text to append after the new line and its indentation. Guaranteed to be a string.
	 * @type {string}
	 */
	appendText: string;
	/**
	 * @property removeText
	 * @brief The number of characters to remove from the new line's indentation. Guaranteed to be a number.
	 * @type {number}
	 */
	removeText: number;
	/**
	 * @property indentation
	 * @brief The computed indentation string for the new line, considering any `removeText` adjustments.
	 * Functional Utility: Provides the final indentation string to be used for the new line,
	 *                     reflecting all rules and adjustments.
	 * @type {string}
	 */
	indentation: string;
}

/**
 * @class StandardAutoClosingPairConditional
 * @internal
 * @brief Represents an auto-closing pair with conditions that determine when it should be active.
 *
 * Functional Utility: This class wraps `IAutoClosingPairConditional` to provide runtime logic
 *                     for checking if an auto-closing pair is allowed given the current token type
 *                     at the cursor position. It also includes functionality to find a "neutral"
 *                     character not present in the pair, which can be useful for certain editor behaviors.
 */
export class StandardAutoClosingPairConditional {
	/**
	 * @property open
	 * @brief The opening character(s) of the pair.
	 * @type {string}
	 */
	readonly open: string;
	/**
	 * @property close
	 * @brief The closing character(s) of the pair.
	 * @type {string}
	 */
	readonly close: string;
	/**
	 * @property _inString
	 * @brief Internal flag indicating if auto-closing is allowed within string tokens.
	 * @private
	 * @type {boolean}
	 */
	private readonly _inString: boolean;
	/**
	 * @property _inComment
	 * @brief Internal flag indicating if auto-closing is allowed within comment tokens.
	 * @private
	 * @type {boolean}
	 */
	private readonly _inComment: boolean;
	/**
	 * @property _inRegEx
	 * @brief Internal flag indicating if auto-closing is allowed within regular expression tokens.
	 * @private
	 * @type {boolean}
	 */
	private readonly _inRegEx: boolean;
	/**
	 * @property _neutralCharacter
	 * @brief Stores a character that is neither the `open` nor `close` character of the pair.
	 * @private
	 * @type {string | null}
	 */
	private _neutralCharacter: string | null = null;
	/**
	 * @property _neutralCharacterSearched
	 * @brief Flag to ensure the search for `_neutralCharacter` is performed only once.
	 * @private
	 * @type {boolean}
	 */
	private _neutralCharacterSearched: boolean = false;

	/**
	 * @brief Constructs a new `StandardAutoClosingPairConditional` instance.
	 * @param source The `IAutoClosingPairConditional` from which to initialize this instance.
	 * Functional Utility: Initializes the auto-closing pair and processes the `notIn` array
	 *                     to set internal flags that control where auto-closing is disabled.
	 * Pre-condition: `source` must be a valid `IAutoClosingPairConditional` object.
	 * Post-condition: Internal flags (`_inString`, `_inComment`, `_inRegEx`) are set based on `source.notIn`.
	 */
	constructor(source: IAutoClosingPairConditional) {
		this.open = source.open;
		this.close = source.close;

		// Block Logic: Initialize auto-closing as allowed in all token types by default.
		// Invariant: These flags will be updated if `source.notIn` specifies exclusions.
		this._inString = true;
		this._inComment = true;
		this._inRegEx = true;

		// Block Logic: Iterates through the `notIn` array to disable auto-closing for specific token types.
		if (Array.isArray(source.notIn)) {
			for (let i = 0, len = source.notIn.length; i < len; i++) {
				const notIn: string = source.notIn[i];
				switch (notIn) {
					case 'string':
						this._inString = false;
						break;
					case 'comment':
						this._inComment = false;
						break;
					case 'regex':
						this._inRegEx = false;
						break;
				}
			}
		}
	}

	/**
	 * @method isOK
	 * @brief Checks if auto-closing is permitted for a given standard token type.
	 * Functional Utility: Determines dynamically whether an auto-closing pair should be active
	 *                     based on the context (e.g., preventing closing quotes inside comments).
	 * @param standardToken The `StandardTokenType` to check against.
	 * @returns `true` if auto-closing is allowed, `false` otherwise.
	 * Pre-condition: `standardToken` is a valid `StandardTokenType`.
	 */
	public isOK(standardToken: StandardTokenType): boolean {
		switch (standardToken) {
			case StandardTokenType.Other:
				return true;
			case StandardTokenType.Comment:
				return this._inComment;
			case StandardTokenType.String:
				return this._inString;
			case StandardTokenType.RegEx:
				return this._inRegEx;
		}
	}

	/**
	 * @method shouldAutoClose
	 * @brief Determines if auto-closing should occur at the given cursor position within the current line context.
	 * Functional Utility: Provides the core logic for deciding when to insert an auto-closing character,
	 *                     considering both empty line scenarios and the token type at the cursor.
	 * @param context The `ScopedLineTokens` representing the tokens on the current line.
	 * @param column The 1-based column number of the cursor position.
	 * @returns `true` if auto-closing should happen, `false` otherwise.
	 * Pre-condition: `context` is a valid `ScopedLineTokens` instance and `column` is a valid 1-based column.
	 */
	public shouldAutoClose(context: ScopedLineTokens, column: number): boolean {
		// Block Logic: Auto-closing is always permitted on an empty line.
		if (context.getTokenCount() === 0) {
			return true;
		}

		// Block Logic: Identifies the token at the cursor position (adjusted by 2 for insertion point).
		const tokenIndex = context.findTokenIndexAtOffset(column - 2);
		// Block Logic: Retrieves the standard token type for the identified token.
		const standardTokenType = context.getStandardTokenType(tokenIndex);
		// Functional Utility: Delegates to `isOK` to determine if auto-closing is allowed for this token type.
		return this.isOK(standardTokenType);
	}

	/**
	 * @method _findNeutralCharacterInRange
	 * @brief Searches for a character within a specified ASCII range that is not part of the opening or closing pair.
	 * @private
	 * @param fromCharCode The ASCII code for the start of the character range (inclusive).
	 * @param toCharCode The ASCII code for the end of the character range (inclusive).
	 * @returns A neutral character as a string, or `null` if no such character is found within the range.
	 * Functional Utility: Aids in finding a character that can serve as a separator or placeholder
	 *                     without conflicting with the auto-closing pair itself.
	 * Pre-condition: `fromCharCode` and `toCharCode` are valid ASCII code points, and `fromCharCode <= toCharCode`.
	 */
	private _findNeutralCharacterInRange(fromCharCode: number, toCharCode: number): string | null {
		// Block Logic: Iterates through characters within the specified ASCII range.
		for (let charCode = fromCharCode; charCode <= toCharCode; charCode++) {
			const character = String.fromCharCode(charCode);
			// Block Logic: Checks if the current character is neither the `open` nor `close` part of the pair.
			if (!this.open.includes(character) && !this.close.includes(character)) {
				return character;
			}
		}
		return null;
	}

	/**
	 * @method findNeutralCharacter
	 * @brief Finds and returns a character from alphanumeric ranges that is not part of the auto-closing pair.
	 * Functional Utility: Provides a character that can be used as a "neutral" separator in scenarios
	 *                     where a character guaranteed not to conflict with the auto-closing pair is needed.
	 *                     This search is optimized to run only once.
	 * @returns A neutral character (digit or letter) or `null` if none found.
	 * Invariant: The search is memoized, ensuring the method's efficiency.
	 */
	public findNeutralCharacter(): string | null {
		// Block Logic: Ensures the search for a neutral character is performed only once per instance.
		if (!this._neutralCharacterSearched) {
			this._neutralCharacterSearched = true;
			// Block Logic: Attempts to find a neutral character within digit range [0-9].
			if (!this._neutralCharacter) {
				this._neutralCharacter = this._findNeutralCharacterInRange(CharCode.Digit0, CharCode.Digit9);
			}
			// Block Logic: If not found in digits, attempts to find a neutral character within lowercase letter range [a-z].
			if (!this._neutralCharacter) {
				this._neutralCharacter = this._findNeutralCharacterInRange(CharCode.a, CharCode.z);
			}
			// Block Logic: If not found in lowercase letters, attempts to find a neutral character within uppercase letter range [A-Z].
			if (!this._neutralCharacter) {
				this._neutralCharacter = this._findNeutralCharacterInRange(CharCode.A, CharCode.Z);
			}
		}
		return this._neutralCharacter;
	}
}

/**
 * @class AutoClosingPairs
 * @internal
 * @brief Manages collections of `StandardAutoClosingPairConditional` objects, indexed for efficient lookup.
 *
 * Functional Utility: This class provides optimized data structures to quickly retrieve
 *                     auto-closing pairs based on their opening or closing characters.
 *                     This is essential for high-performance auto-closing logic in the editor.
 */
export class AutoClosingPairs {
	/**
	 * @property autoClosingPairsOpenByStart
	 * @brief A map where the key is the first character of an opening auto-closing sequence.
	 * Functional Utility: Enables quick lookup of auto-closing pairs when an opening character is typed.
	 * @type {Map<string, StandardAutoClosingPairConditional[]>}
	 */
	public readonly autoClosingPairsOpenByStart: Map<string, StandardAutoClosingPairConditional[]>;
	/**
	 * @property autoClosingPairsOpenByEnd
	 * @brief A map where the key is the last character of an opening auto-closing sequence.
	 * Functional Utility: Useful for scenarios where the end of an opening sequence triggers a check.
	 * @type {Map<string, StandardAutoClosingPairConditional[]>}
	 */
	public readonly autoClosingPairsOpenByEnd: Map<string, StandardAutoClosingPairConditional[]>;
	/**
	 * @property autoClosingPairsCloseByStart
	 * @brief A map where the key is the first character of a closing auto-closing sequence.
	 * Functional Utility: Enables quick lookup of auto-closing pairs when a closing character is typed.
	 * @type {Map<string, StandardAutoClosingPairConditional[]>}
	 */
	public readonly autoClosingPairsCloseByStart: Map<string, StandardAutoClosingPairConditional[]>;
	/**
	 * @property autoClosingPairsCloseByEnd
	 * @brief A map where the key is the last character of a closing auto-closing sequence.
	 * Functional Utility: Useful for complex closing sequences.
	 * @type {Map<string, StandardAutoClosingPairConditional[]>}
	 */
	public readonly autoClosingPairsCloseByEnd: Map<string, StandardAutoClosingPairConditional[]>;
	/**
	 * @property autoClosingPairsCloseSingleChar
	 * @brief A map specifically for single-character auto-closing pairs, keyed by the closing character.
	 * Functional Utility: Provides an optimized lookup for common single-character auto-closing scenarios.
	 * @type {Map<string, StandardAutoClosingPairConditional[]>}
	 */
	public readonly autoClosingPairsCloseSingleChar: Map<string, StandardAutoClosingPairConditional[]>;

	/**
	 * @brief Constructs a new `AutoClosingPairs` instance from a list of conditional auto-closing pairs.
	 * @param autoClosingPairs An array of `StandardAutoClosingPairConditional` objects to manage.
	 * Functional Utility: Populates various internal maps to enable efficient lookups of auto-closing pairs.
	 * Pre-condition: `autoClosingPairs` is an array of valid `StandardAutoClosingPairConditional` instances.
	 * Post-condition: All internal maps are initialized and populated with the provided auto-closing pairs,
	 *                 indexed by their opening and closing characters.
	 */
	constructor(autoClosingPairs: StandardAutoClosingPairConditional[]) {
		this.autoClosingPairsOpenByStart = new Map<string, StandardAutoClosingPairConditional[]>();
		this.autoClosingPairsOpenByEnd = new Map<string, StandardAutoClosingPairConditional[]>();
		this.autoClosingPairsCloseByStart = new Map<string, StandardAutoClosingPairConditional[]>();
		this.autoClosingPairsCloseByEnd = new Map<string, StandardAutoClosingPairConditional[]>();
		this.autoClosingPairsCloseSingleChar = new Map<string, StandardAutoClosingPairConditional[]>();
		// Block Logic: Iterates through each auto-closing pair and adds it to the relevant lookup maps.
		for (const pair of autoClosingPairs) {
			// Functional Utility: Indexes pairs by the first character of their opening sequence.
			appendEntry(this.autoClosingPairsOpenByStart, pair.open.charAt(0), pair);
			// Functional Utility: Indexes pairs by the last character of their opening sequence.
			appendEntry(this.autoClosingPairsOpenByEnd, pair.open.charAt(pair.open.length - 1), pair);
			// Functional Utility: Indexes pairs by the first character of their closing sequence.
			appendEntry(this.autoClosingPairsCloseByStart, pair.close.charAt(0), pair);
			// Functional Utility: Indexes pairs by the last character of their closing sequence.
			appendEntry(this.autoClosingPairsCloseByEnd, pair.close.charAt(pair.close.length - 1), pair);
			// Block Logic: Special handling for single-character open/close pairs for optimized lookup.
			if (pair.close.length === 1 && pair.open.length === 1) {
				appendEntry(this.autoClosingPairsCloseSingleChar, pair.close, pair);
			}
		}
	}
}
/**
 * @function appendEntry
 * @brief Appends a value to an array associated with a key in a Map, creating the array if it doesn't exist.
 * @template K The type of the keys in the Map.
 * @template V The type of the values stored in the arrays within the Map.
 * @param target The Map to which the entry will be appended.
 * @param key The key under which the value should be stored.
 * @param value The value to append to the array.
 * Functional Utility: Simplifies the pattern of managing arrays as values in a Map, ensuring that
 *                     `push` operations are safe even if the key is new.
 * Pre-condition: `target` is a valid Map instance. `key` and `value` are of their respective types.
 * Post-condition: `value` is added to the array associated with `key`. If `key` did not exist, a new
 *                 array containing `value` is created and associated with `key`.
 */
function appendEntry<K, V>(target: Map<K, V[]>, key: K, value: V): void {
	// Block Logic: Checks if the Map already contains an array for the given key.
	if (target.has(key)) {
		// Functional Utility: Appends the value to the existing array.
		target.get(key)!.push(value);
	} else {
		// Functional Utility: Creates a new array with the value and associates it with the key.
		target.set(key, [value]);
	}
}
