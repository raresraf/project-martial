/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
/**
 * @file frontMatterHeader.ts
 * @brief Defines the `FrontMatterHeader` class for representing document front matter.
 *
 * This module introduces the `FrontMatterHeader` class, extending
 * `MarkdownExtensionsToken`. Its primary purpose is to encapsulate and manage
 * the structural and textual components of a front matter block within a document,
 * such as those commonly found in Markdown files. This includes the distinct
 * start and end markers, as well as the content contained within, facilitating
 * structured access and manipulation of document metadata.
 * Domain: Markdown Parsing, Text Editor, Tokenization, Document Structure.
 */

import { Range } from '../../../core/range.js';
import { BaseToken, Text } from '../../baseToken.js';
import { MarkdownExtensionsToken } from './markdownExtensionsToken.js';
import { TSimpleDecoderToken } from '../../simpleCodec/simpleDecoder.js';
import { FrontMatterMarker, TMarkerToken } from './frontMatterMarker.js';

/**
 * @class FrontMatterHeader
 * @augments MarkdownExtensionsToken
 * @brief Token representing a "Front Matter" header in a text document.
 *
 * Functional Utility: This class extends `MarkdownExtensionsToken` to
 * specifically encapsulate and provide structured access to a "Front Matter"
 * block within a text. It serves as a composite token, comprising individual
 * tokens for its `startMarker`, `content`, and `endMarker`. This structure
 * allows for direct manipulation and analysis of the metadata contained
 * within the front matter, which is crucial for parsing and rendering
 * documents that utilize such constructs (e.g., Markdown files with YAML
 * front matter).
 * Invariant: A `FrontMatterHeader` token's `range` always encompasses the ranges of its `startMarker`, `content`, and `endMarker`.
 *            Its `startMarker` and `endMarker` are always `FrontMatterMarker` instances.
 */
export class FrontMatterHeader extends MarkdownExtensionsToken {
	/**
	 * @brief Constructs a new `FrontMatterHeader` token.
	 * @param range The overall range of the entire front matter block in the document.
	 * @param startMarker The `FrontMatterMarker` token representing the opening delimiter.
	 * @param content The `Text` token containing the actual content/metadata of the front matter.
	 * @param endMarker The `FrontMatterMarker` token representing the closing delimiter.
	 * Functional Utility: Assembles a complete `FrontMatterHeader` token from its constituent parts,
	 *                     providing a unified object for interacting with a parsed front matter block.
	 *                     This composite token encapsulates the structural boundaries and the
	 *                     metadata content, enabling cohesive processing.
	 * Pre-condition: `startMarker`, `content`, and `endMarker` must be valid token instances
	 *                with consistent ranges that form a contiguous front matter block.
	 * Invariant: The `range` provided to the constructor must accurately span the collective
	 *            ranges of `startMarker`, `content`, and `endMarker`.
	 */
	constructor(
		range: Range,
		public readonly startMarker: FrontMatterMarker, /**< The token representing the opening marker of the front matter. */
		public readonly content: Text, /**< The token representing the textual content (metadata) of the front matter. */
		public readonly endMarker: FrontMatterMarker, /**< The token representing the closing marker of the front matter. */
	) {
		super(range);
	}

	/**
	 * @brief Gets the complete text representation of the front matter block.
	 * @returns A string concatenating the start marker, content, and end marker.
	 *
	 * Functional Utility: This getter provides the full, contiguous string
	 * representation of the `FrontMatterHeader` token, including its start
	 * marker, content, and end marker. It is useful for operations requiring
	 * the entire textual block of the front matter as a single string.
	 * Invariant: The returned string preserves the original formatting and content, including markers.
	 */
	public get text(): string {
		const text: string[] = [
			this.startMarker.text,
			this.content.text,
			this.endMarker.text,
		];

		return text.join('');
	}

	/**
	 * @brief Gets the range of the content part of the Front Matter header.
	 * @returns A `Range` object specifying the start and end positions of the content.
	 *
	 * Functional Utility: This getter returns the `Range` that exclusively
	 * covers the metadata content within the front matter block, excluding
	 * the start and end markers. This is particularly useful for operations
	 * that need to operate solely on the user-defined metadata, such as
	 * parsing the content for key-value pairs.
	 * Invariant: The returned `Range` is always a sub-range of the overall `FrontMatterHeader`'s `range`.
	 */
	public get contentRange(): Range {
		return this.content.range;
	}

	/**
	 * @brief Gets the content token of the Front Matter header.
	 * @returns The `Text` token that represents the content (metadata) of the front matter.
	 *
	 * Functional Utility: This getter provides direct access to the `Text`
	 * token that holds the raw content of the front matter block. This allows
	 * for deeper inspection or processing of the content token itself, beyond
	 * just its string value or range.
	 * Invariant: The returned `Text` token's `range` is identical to the `contentRange` property.
	 */
	public get contentToken(): Text {
		return this.content;
	}

	/**
	 * @brief Checks if this `FrontMatterHeader` token is equal to another token.
	 * @param other The other token to compare against.
	 * @returns `true` if the tokens are semantically equivalent, `false` otherwise.
	 *
	 * Functional Utility: This method provides a robust comparison mechanism
	 * to determine if two `FrontMatterHeader` tokens represent the same
	 * front matter block. It first verifies if their ranges are identical
	 * and then performs a deep comparison of their concatenated textual
	 * content, ensuring that both position and content match for equality.
	 */
	public override equals<T extends BaseToken>(other: T): boolean {
		// Block Logic: Checks if the tokens occupy the same textual range.
		// Functional Utility: Efficiently prunes non-matching tokens based on their positional information.
		if (!super.sameRange(other.range)) {
			return false;
		}

		// Block Logic: Verifies that the `other` token is specifically an instance of `FrontMatterHeader`.
		// Functional Utility: Ensures type compatibility for a semantic comparison.
		// Invariant: Only two `FrontMatterHeader` tokens can be truly equal.
		if (!(other instanceof FrontMatterHeader)) {
			return false;
		}

		// Block Logic: Optimizes comparison by quickly ruling out tokens with different total text lengths.
		// Functional Utility: Prevents more expensive string content comparison if lengths do not match.
		if (this.text.length !== other.text.length) {
			return false;
		}

		// Functional Utility: Performs a final, comprehensive comparison of the entire textual content
		//                     of both front matter headers to confirm semantic equality.
		return (this.text === other.text);
	}

	/**
	 * @brief Factory method to create a new `FrontMatterHeader` instance from raw tokens.
	 * @param startMarkerTokens An array of tokens forming the start marker.
	 * @param contentTokens An array of tokens forming the content.
	 * @param endMarkerTokens An array of tokens forming the end marker.
	 * @returns A new `FrontMatterHeader` instance.
	 *
	 * Functional Utility: This static method acts as a constructor from a
	 * lower-level token representation. It takes arrays of tokens that
	 * constitute the start marker, content, and end marker of a front matter
	 * block, calculates the overall range, and then constructs a new,
	 * fully-formed `FrontMatterHeader` object. This simplifies the creation
	 * of `FrontMatterHeader` instances during parsing by abstracting away
	 * the details of individual token aggregation.
	 * Pre-condition: `startMarkerTokens`, `contentTokens`, and `endMarkerTokens` must contain valid `TMarkerToken`
	 *                and `TSimpleDecoderToken` instances respectively, and represent a syntactically correct front matter block.
	 * Post-condition: A new `FrontMatterHeader` instance is returned, whose overall `range` is correctly derived
	 *                 from the constituent tokens, and whose internal token references are properly set.
	 */
	public static fromTokens(
		startMarkerTokens: readonly TMarkerToken[],
		contentTokens: readonly TSimpleDecoderToken[],
		endMarkerTokens: readonly TMarkerToken[],
	): FrontMatterHeader {
		const range = BaseToken.fullRange(
			[...startMarkerTokens, ...endMarkerTokens],
		);

		return new FrontMatterHeader(
			range,
			FrontMatterMarker.fromTokens(startMarkerTokens),
			Text.fromTokens(contentTokens),
			FrontMatterMarker.fromTokens(endMarkerTokens),
		);
	}

	/**
	 * @brief Returns a string representation of the `FrontMatterHeader` token.
	 * @returns A string representing the token, including its truncated text and range.
	 *
	 * Functional Utility: This method generates a concise, human-readable
	 * string for debugging and logging purposes. It provides a summary of
	 * the `FrontMatterHeader` token by showing a shortened version of its
	 * textual content and its `Range` information, aiding in quick
	 * identification and analysis of the token's characteristics and location.
	 * Invariant: The string representation accurately reflects the token's type, content (truncated), and location.
	 */
	public override toString(): string {
		return `frontmatter("${this.shortText()}")${this.range}`;
	}
}
