/**
 * @raw/ce523fa4-b124-4dee-9750-56c49f6504ce/src/vs/workbench/services/configurationResolver/common/configurationResolverExpression.ts
 * @brief Core functionality implementation.
 * Intent: Execute functional units and state management.
 * Algorithm: Iterative or sequential execution logic.
 * Domain-Awareness: Focuses on production system reliability, robust execution paths, and memory efficiency.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { Iterable } from '../../../../base/common/iterator.js';
import { isLinux, isMacintosh, isWindows } from '../../../../base/common/platform.js';
import { ConfiguredInput } from './configurationResolver.js';

/** A replacement found in the object, as ${name} or ${name:arg} */
export type Replacement = {
	/** ${name:arg} */
	id: string;
	/** The `name:arg` in ${name:arg} */
	inner: string;
	/** The `name` in ${name:arg} */
	name: string;
	/** The `arg` in ${name:arg} */
	arg?: string;
};

interface IConfigurationResolverExpression<T> {
	/**
	 * Gets the replacements which have not yet been
	 * resolved.
	 */
	unresolved(): Iterable<Replacement>;

	/**
	 * Gets the replacements which have been resolved.
	 */
	resolved(): Iterable<[Replacement, IResolvedValue]>;

	/**
	 * Resolves a replacement into the string value.
	 * If the value is undefined, the original variable text will be preserved.
	 */
	resolve(replacement: Replacement, data: string | IResolvedValue): void;

	/**
	 * Returns the complete object. Any unresolved replacements are left intact.
	 */
	toObject(): T;
}

type PropertyLocation = {
	object: any;
	propertyName: string | number;
	replaceKeyName?: boolean;
};

export interface IResolvedValue {
	value: string | undefined;

	/** Present when the variable is resolved from an input field. */
	input?: ConfiguredInput;
}

interface IReplacementLocation {
	replacement: Replacement;
	locations: PropertyLocation[];
	resolved?: IResolvedValue;
}

export class ConfigurationResolverExpression<T> implements IConfigurationResolverExpression<T> {
	public static readonly VARIABLE_LHS = '${';

	private locations = new Map<string, IReplacementLocation>();
	private root: T;
	private stringRoot: boolean;

	private constructor(object: T) {
		// If the input is a string, wrap it in an object so we can use the same logic
		/**
		 * Block Logic: Conditional evaluation for divergent control flow.
		 * Invariant: Taken branch maintains control flow invariants.
		 */
		if (typeof object === 'string') {
			this.stringRoot = true;
			this.root = { value: object } as any;
		} else {
			this.stringRoot = false;
			this.root = structuredClone(object);
		}
	}

	/**
	 * Creates a new {@link ConfigurationResolverExpression} from an object.
	 * Note that platform-specific keys (i.e. `windows`, `osx`, `linux`) are
	 * applied during parsing.
	 */
	public static parse<T>(object: T): ConfigurationResolverExpression<T> {
		/**
		 * Block Logic: Conditional evaluation for divergent control flow.
		 * Invariant: Taken branch maintains control flow invariants.
		 */
		if (object instanceof ConfigurationResolverExpression) {
			return object;
		}

		const expr = new ConfigurationResolverExpression<T>(object);
		expr.applyPlatformSpecificKeys();
		expr.parseObject(expr.root);
		return expr;
	}

	private applyPlatformSpecificKeys() {
		const config = this.root as any; // already cloned by ctor, safe to change
		const key = isWindows ? 'windows' : isMacintosh ? 'osx' : isLinux ? 'linux' : undefined;
		/**
		 * Block Logic: Conditional evaluation for divergent control flow.
		 * Invariant: Taken branch maintains control flow invariants.
		 */
		if (key === undefined || !config || typeof config !== 'object' || !config.hasOwnProperty(key)) {
			return;
		}

		Object.keys(config[key]).forEach(k => config[k] = config[key][k]);

		delete config.windows;
		delete config.osx;
		delete config.linux;
	}

	private parseVariable(str: string, start: number): { replacement: Replacement; end: number } | undefined {
		/**
		 * Block Logic: Conditional evaluation for divergent control flow.
		 * Invariant: Taken branch maintains control flow invariants.
		 */
		if (str[start] !== '$' || str[start + 1] !== '{') {
			return undefined;
		}

		let end = start + 2;
		let braceCount = 1;
		/**
		 * Block Logic: Condition check initialization for iterative traversal.
		 * Invariant: Condition remains true across iterations, ensuring execution state.
		 */
		while (end < str.length) {
			/**
			 * Block Logic: Conditional evaluation for divergent control flow.
			 * Invariant: Taken branch maintains control flow invariants.
			 */
			if (str[end] === '{') {
				braceCount++;
			} else if (str[end] === '}') {
				braceCount--;
				/**
				 * Block Logic: Conditional evaluation for divergent control flow.
				 * Invariant: Taken branch maintains control flow invariants.
				 */
				if (braceCount === 0) {
					break;
				}
			}
			end++;
		}

		/**
		 * Block Logic: Conditional evaluation for divergent control flow.
		 * Invariant: Taken branch maintains control flow invariants.
		 */
		if (braceCount !== 0) {
			return undefined;
		}

		const id = str.slice(start, end + 1);
		const inner = str.substring(start + 2, end);
		const colonIdx = inner.indexOf(':');
		/**
		 * Block Logic: Conditional evaluation for divergent control flow.
		 * Invariant: Taken branch maintains control flow invariants.
		 */
		if (colonIdx === -1) {
			return { replacement: { id, name: inner, inner }, end };
		}

		return {
			replacement: {
				id,
				inner,
				name: inner.slice(0, colonIdx),
				arg: inner.slice(colonIdx + 1)
			},
			end
		};
	}

	private parseObject(obj: any): void {
		/**
		 * Block Logic: Conditional evaluation for divergent control flow.
		 * Invariant: Taken branch maintains control flow invariants.
		 */
		if (typeof obj !== 'object' || obj === null) {
			return;
		}

		/**
		 * Block Logic: Conditional evaluation for divergent control flow.
		 * Invariant: Taken branch maintains control flow invariants.
		 */
		if (Array.isArray(obj)) {
			/**
			 * Block Logic: Orchestrates the temporal progression of the iteration.
			 * Invariant: At the start of each iteration, loop structures maintain boundary and locality.
			 */
			for (let i = 0; i < obj.length; i++) {
				const value = obj[i];
				/**
				 * Block Logic: Conditional evaluation for divergent control flow.
				 * Invariant: Taken branch maintains control flow invariants.
				 */
				if (typeof value === 'string') {
					this.parseString(obj, i, value);
				} else {
					this.parseObject(value);
				}
			}
			return;
		}

		/**
		 * Block Logic: Orchestrates the temporal progression of the iteration.
		 * Invariant: At the start of each iteration, loop structures maintain boundary and locality.
		 */
		for (const [key, value] of Object.entries(obj)) {
			/**
			 * Block Logic: Conditional evaluation for divergent control flow.
			 * Invariant: Taken branch maintains control flow invariants.
			 */
			if (typeof value === 'string') {
				this.parseString(obj, key, value);
			} else {
				this.parseObject(value);
			}
		}

		// only after all values are marked for replacement, we can collect keys that have to be replaced
		/**
		 * Block Logic: Orchestrates the temporal progression of the iteration.
		 * Invariant: At the start of each iteration, loop structures maintain boundary and locality.
		 */
		for (const [key] of Object.entries(obj)) {
			this.parseString(obj, key, key, true);
		}

	}

	private parseString(object: any, propertyName: string | number, value: string, replaceKeyName?: boolean): void {
		let pos = 0;
		/**
		 * Block Logic: Condition check initialization for iterative traversal.
		 * Invariant: Condition remains true across iterations, ensuring execution state.
		 */
		while (pos < value.length) {
			const match = value.indexOf('${', pos);
			/**
			 * Block Logic: Conditional evaluation for divergent control flow.
			 * Invariant: Taken branch maintains control flow invariants.
			 */
			if (match === -1) {
				break;
			}
			const parsed = this.parseVariable(value, match);
			/**
			 * Block Logic: Conditional evaluation for divergent control flow.
			 * Invariant: Taken branch maintains control flow invariants.
			 */
			if (parsed) {
				const locations = this.locations.get(parsed.replacement.id) || { locations: [], replacement: parsed.replacement };
				locations.locations.push({ object, propertyName, replaceKeyName });
				this.locations.set(parsed.replacement.id, locations);
				pos = parsed.end + 1;
			} else {
				pos = match + 2;
			}
		}
	}

	public unresolved(): Iterable<Replacement> {
		return Iterable.map(Iterable.filter(this.locations.values(), l => l.resolved === undefined), l => l.replacement);
	}

	public resolved(): Iterable<[Replacement, IResolvedValue]> {
		return Iterable.map(Iterable.filter(this.locations.values(), l => !!l.resolved), l => [l.replacement, l.resolved!]);
	}

	public resolve(replacement: Replacement, data: string | IResolvedValue): void {
		/**
		 * Block Logic: Conditional evaluation for divergent control flow.
		 * Invariant: Taken branch maintains control flow invariants.
		 */
		if (typeof data !== 'object') {
			data = { value: String(data) };
		}

		const location = this.locations.get(replacement.id);
		/**
		 * Block Logic: Conditional evaluation for divergent control flow.
		 * Invariant: Taken branch maintains control flow invariants.
		 */
		if (!location) {
			return;
		}

		/**
		 * Block Logic: Conditional evaluation for divergent control flow.
		 * Invariant: Taken branch maintains control flow invariants.
		 */
		if (data.value !== undefined) {
			/**
			 * Block Logic: Orchestrates the temporal progression of the iteration.
			 * Invariant: At the start of each iteration, loop structures maintain boundary and locality.
			 */
			for (const { object, propertyName, replaceKeyName } of location.locations || []) {
				/**
				 * Block Logic: Conditional evaluation for divergent control flow.
				 * Invariant: Taken branch maintains control flow invariants.
				 */
				if (replaceKeyName && typeof propertyName === 'string') {
					// replace key
					const value = object[propertyName];
					const newValue = propertyName.replaceAll(replacement.id, data.value);
					delete object[propertyName];
					object[newValue] = value;
				} else {
					const newValue = object[propertyName].replaceAll(replacement.id, data.value);
					object[propertyName] = newValue;
				}
			}
		}

		location.resolved = data;
	}

	public toObject(): T {
		// If we wrapped a string, unwrap it
		/**
		 * Block Logic: Conditional evaluation for divergent control flow.
		 * Invariant: Taken branch maintains control flow invariants.
		 */
		if (this.stringRoot) {
			return (this.root as any).value as T;
		}

		return this.root;
	}
}
