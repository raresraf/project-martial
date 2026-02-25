/**
 * @file raw/45d7f3eb-3d1c-4486-ab1f-177bb3bea12d/src/vs/base/node/nls.ts
 * @brief Handles the resolution of National Language Support (NLS) configuration
 * in VS Code, including the processing and caching of language packs.
 */

/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import * as path from 'path';
import * as fs from 'fs';
import * as perf from '../common/performance.js';
import type { ILanguagePacks, INLSConfiguration } from '../../nls.js';

/**
 * Defines the context required to resolve the NLS configuration.
 */
export interface IResolveNLSConfigurationContext {

	/**
	 * Location where `nls.messages.json` and `nls.keys.json` are stored.
	 * These files contain the default English strings and their ordering.
	 */
	readonly nlsMetadataPath: string;

	/**
	 * Path to the user data directory. Used as a cache for
	 * language packs converted to the format needed by the application.
	 */
	readonly userDataPath: string;

	/**
	 * Commit hash of the running application. Can be `undefined`
	 * when not running a built version (e.g., in development).
	 */
	readonly commit: string | undefined;

	/**
	 * The locale specified by the user (e.g., from `argv.json` or `app.getLocale()`).
	 */
	readonly userLocale: string;

	/**
	 * The locale determined from the operating system.
	 */
	readonly osLocale: string;
}

/**
 * Resolves the NLS configuration for the application.
 * This function determines the best available language pack, generates the necessary
 * translation files if they are not cached, and returns a configuration object
 * for the NLS system.
 *
 * @param context The context required for resolution.
 * @returns A promise that resolves to the NLS configuration.
 */
export async function resolveNLSConfiguration({ userLocale, osLocale, userDataPath, commit, nlsMetadataPath }: IResolveNLSConfigurationContext): Promise<INLSConfiguration> {
	perf.mark('code/willGenerateNls');

	// Early exit for development, pseudo-locales, or English locales as they don't require language packs.
	if (
		process.env['VSCODE_DEV'] ||
		userLocale === 'pseudo' ||
		userLocale.startsWith('en') ||
		!commit ||
		!userDataPath
	) {
		return defaultNLSConfiguration(userLocale, osLocale, nlsMetadataPath);
	}

	try {
		// 1. Read language pack configurations from `languagepacks.json`.
		const languagePacks = await getLanguagePackConfigurations(userDataPath);
		/**
		 * @block
		 * @description If no language packs are configured, the function returns the default NLS configuration.
		 * This is a common case when no language pack extensions are installed.
		 */
		if (!languagePacks) {
			return defaultNLSConfiguration(userLocale, osLocale, nlsMetadataPath);
		}

		// 2. Resolve the specific language to use (e.g., 'fr' from 'fr-CA').
		const resolvedLanguage = resolveLanguagePackLanguage(languagePacks, userLocale);
		/**
		 * @block
		 * @description If the user's locale cannot be resolved to an available language pack,
		 * the function returns the default NLS configuration.
		 */
		if (!resolvedLanguage) {
			return defaultNLSConfiguration(userLocale, osLocale, nlsMetadataPath);
		}

		const languagePack = languagePacks[resolvedLanguage];
		const mainLanguagePackPath = languagePack?.translations?.['vscode'];
		/**
		 * @block
		 * @description This check validates the integrity of the language pack configuration.
		 * It ensures that the language pack has a valid hash and a path to the main translation file.
		 * If the configuration is invalid, it falls back to the default English configuration.
		 */
		if (
			!languagePack ||
			typeof languagePack.hash !== 'string' ||
			!languagePack.translations ||
			typeof mainLanguagePackPath !== 'string' ||
			!(await exists(mainLanguagePackPath))
		) {
			// The language pack is not valid or installed correctly.
			return defaultNLSConfiguration(userLocale, osLocale, nlsMetadataPath);
		}

		// 3. Set up cache paths. The cache is versioned by language pack hash and commit hash.
		const languagePackId = `${languagePack.hash}.${resolvedLanguage}`;
		const globalLanguagePackCachePath = path.join(userDataPath, 'clp', languagePackId);
		const commitLanguagePackCachePath = path.join(globalLanguagePackCachePath, commit);
		const languagePackMessagesFile = path.join(commitLanguagePackCachePath, 'nls.messages.json');
		const translationsConfigFile = path.join(globalLanguagePackCachePath, 'tcf.json');
		const languagePackCorruptMarkerFile = path.join(globalLanguagePackCachePath, 'corrupted.info');

		// 4. Handle corrupted cache.
		if (await exists(languagePackCorruptMarkerFile)) {
			// Delete the entire cache folder for this language pack if it's marked as corrupt.
			await fs.promises.rm(globalLanguagePackCachePath, { recursive: true, force: true, maxRetries: 3 });
		}

		const result: INLSConfiguration = {
			userLocale,
			osLocale,
			resolvedLanguage,
			defaultMessagesFile: path.join(nlsMetadataPath, 'nls.messages.json'),
			languagePack: {
				translationsConfigFile,
				messagesFile: languagePackMessagesFile,
				corruptMarkerFile: languagePackCorruptMarkerFile
			},

			// Deprecated properties for vscode-nls compatibility.
			locale: userLocale,
			availableLanguages: { '*': resolvedLanguage },
			_languagePackId: languagePackId,
			_languagePackSupport: true,
			_translationsConfigFile: translationsConfigFile,
			_cacheRoot: globalLanguagePackCachePath,
			_resolvedLanguagePackCoreLocation: commitLanguagePackCachePath,
			_corruptedFile: languagePackCorruptMarkerFile
		};

		// 5. Cache hit: If the commit-specific cache exists, we're done.
		if (await exists(commitLanguagePackCachePath)) {
			touch(commitLanguagePackCachePath).catch(() => { }); // Update timestamp for cache management.
			perf.mark('code/didGenerateNls');
			return result;
		}

		// 6. Cache miss: Generate the translation files.
		const [
			,
			nlsDefaultKeys,
			nlsDefaultMessages,
			nlsPackdata
		]:
			[unknown, Array<[string, string[]]>, string[], { contents: Record<string, Record<string, string>> }]
			= await Promise.all([
				fs.promises.mkdir(commitLanguagePackCachePath, { recursive: true }),
				// nls.keys.json: defines the structure and order of keys.
				fs.promises.readFile(path.join(nlsMetadataPath, 'nls.keys.json'), 'utf-8').then(content => JSON.parse(content)),
				// nls.messages.json: contains the default (English) messages in a flat array.
				fs.promises.readFile(path.join(nlsMetadataPath, 'nls.messages.json'), 'utf-8').then(content => JSON.parse(content)),
				// The main language pack file with translations.
				fs.promises.readFile(mainLanguagePackPath, 'utf-8').then(content => JSON.parse(content)),
			]);

		const nlsResult: string[] = [];

		// This process creates a flat array of messages for the target locale.
		// It uses `nls.keys.json` to iterate in the correct order, ensuring that indices match up.
		// For each key, it looks up the translation. If not found, it falls back to the default English message.
		let nlsIndex = 0;
		for (const [moduleId, nlsKeys] of nlsDefaultKeys) {
			const moduleTranslations = nlsPackdata.contents[moduleId];
			for (const nlsKey of nlsKeys) {
				nlsResult.push(moduleTranslations?.[nlsKey] || nlsDefaultMessages[nlsIndex]);
				nlsIndex++;
			}
		}

		// 7. Write the generated files to the cache.
		await Promise.all([
			fs.promises.writeFile(languagePackMessagesFile, JSON.stringify(nlsResult), 'utf-8'),
			fs.promises.writeFile(translationsConfigFile, JSON.stringify(languagePack.translations), 'utf-8')
		]);

		perf.mark('code/didGenerateNls');

		return result;
	} catch (error) {
		console.error('Generating translation files failed.', error);
		// Fallback to default configuration on any error.
		return defaultNLSConfiguration(userLocale, osLocale, nlsMetadataPath);
	}
}

/**
 * Reads the `languagepacks.json` file from the user data directory.
 * This file contains metadata about all installed language packs.
 * @param userDataPath The path to the user data directory.
 * @returns A promise that resolves to the language pack configurations, or undefined if not found.
 */
async function getLanguagePackConfigurations(userDataPath: string): Promise<ILanguagePacks | undefined> {
	const configFile = path.join(userDataPath, 'languagepacks.json');
	try {
		const content = await fs.promises.readFile(configFile, 'utf-8');
		return JSON.parse(content);
	} catch (err) {
		// If the file doesn't exist or is corrupt, we have no language pack config.
		return undefined;
	}
}

/**
 * Resolves a language from a locale.
 * For example, for a locale `fr-CA`, it would first check for `fr-CA` and then `fr`.
 * @param languagePacks The available language packs.
 * @param locale The user's locale.
 * @returns The resolved language that has a language pack available, or undefined.
 */
function resolveLanguagePackLanguage(languagePacks: ILanguagePacks, locale: string | undefined): string | undefined {
	try {
		/**
		 * @block
		 * @description This loop implements the locale fallback strategy.
		 * Invariant: At each iteration, `locale` is a candidate for matching a language pack.
		 * If a match is found, the loop terminates. Otherwise, the locale is shortened by
		 * removing its last segment (e.g., `en-US` becomes `en`).
		 */
		while (locale) {
			if (languagePacks[locale]) {
				return locale; // Exact match found.
			}
			// Fallback to the parent language (e.g., 'fr' from 'fr-CA').
			const index = locale.lastIndexOf('-');
			if (index > 0) {
				locale = locale.substring(0, index);
			} else {
				return undefined; // No more segments to check.
			}
		}
	} catch (error) {
		console.error('Resolving language pack configuration failed.', error);
	}

	return undefined;
}

/**
 * Returns a default NLS configuration for English, which serves as the ultimate fallback.
 * @param userLocale The user's locale.
 * @param osLocale The OS's locale.
 * @param nlsMetadataPath The path to the NLS metadata directory.
 * @returns The default NLS configuration.
 */
function defaultNLSConfiguration(userLocale: string, osLocale: string, nlsMetadataPath: string): INLSConfiguration {
	perf.mark('code/didGenerateNls');

	return {
		userLocale,
		osLocale,
		resolvedLanguage: 'en',
		defaultMessagesFile: path.join(nlsMetadataPath, 'nls.messages.json'),
		// Deprecated properties for vscode-nls compatibility.
		locale: userLocale,
		availableLanguages: {}
	};
}

//#region fs helpers

/**
 * A helper function to check if a file or directory exists.
 * @param path The path to check.
 * @returns A promise that resolves to true if the path exists, false otherwise.
 */
async function exists(path: string): Promise<boolean> {
	try {
		await fs.promises.access(path);
		return true;
	} catch {
		return false;
	}
}

/**
 * A helper function to update the access and modification times of a file or directory.
 * @param path The path to touch.
 * @returns A promise that resolves when the operation is complete.
 */
function touch(path: string): Promise<void> {
	const date = new Date();
	return fs.promises.utimes(path, date, date);
}

//#endregion