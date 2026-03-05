/**
 * @module nls
 * @description This module is responsible for Native Language Support (NLS) configuration
 * resolution within the application. It handles the loading, processing, and caching of
 * language packs based on user and operating system locales to provide localized
 * strings for the user interface.
 *
 * It manages the lifecycle of translation files, including checking for corrupted
 * caches, generating NLS message files from language packs, and providing fallback
 * mechanisms.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import * as path from 'path';
import * as fs from 'fs';
import * as perf from '../common/performance.js';
import type { ILanguagePacks, INLSConfiguration } from '../../nls.js';

export interface IResolveNLSConfigurationContext {
	/**
	 * @property nlsMetadataPath
	 * @description Location where `nls.messages.json` and `nls.keys.json` are stored.
	 */
	readonly nlsMetadataPath: string;

	/**
	 * @property userDataPath
	 * @description Path to the user data directory. Used as a cache for
	 * language packs converted to the format we need.
	 */
	readonly userDataPath: string;

	/**
	 * @property commit
	 * @description Commit of the running application. Can be `undefined`
	 * when not built.
	 */
	readonly commit: string | undefined;

	/**
	 * @property userLocale
	 * @description Locale as defined in `argv.json` or `app.getLocale()`.
	 */
	readonly userLocale: string;

	/**
	 * @property osLocale
	 * @description Locale as defined by the OS (e.g. `app.getPreferredSystemLanguages()`).
	 */
	readonly osLocale: string;
}

export async function resolveNLSConfiguration({ userLocale, osLocale, userDataPath, commit, nlsMetadataPath }: IResolveNLSConfigurationContext): Promise<INLSConfiguration> {
	/**
	 * @function resolveNLSConfiguration
	 * @description Resolves the Native Language Support (NLS) configuration for the application.
	 * This function determines the appropriate language pack to use, manages caching
	 * of translation files, and provides fallback to default English if a specific
	 * language pack cannot be resolved or is corrupted.
	 *
	 * @param {IResolveNLSConfigurationContext} context - The context object containing
	 *   `userLocale`, `osLocale`, `userDataPath`, `commit`, and `nlsMetadataPath`.
	 * @returns {Promise<INLSConfiguration>} A promise that resolves to the NLS configuration.
	 */
	perf.mark('code/willGenerateNls');

	// Pre-condition: Check if NLS resolution should be skipped or use default configuration.
	// Block Logic: Skips NLS resolution for development mode, pseudo localization,
	// English locales, or if essential metadata (commit, userDataPath) is missing.
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
		// Attempt to retrieve installed language pack configurations.
		const languagePacks = await getLanguagePackConfigurations(userDataPath);
		// If no language packs are found, return the default NLS configuration.
		if (!languagePacks) {
			return defaultNLSConfiguration(userLocale, osLocale, nlsMetadataPath);
		}

		// Resolve the specific language to use from the available language packs based on the user's locale.
		const resolvedLanguage = resolveLanguagePackLanguage(languagePacks, userLocale);
		// If no language could be resolved, return the default NLS configuration.
		if (!resolvedLanguage) {
			return defaultNLSConfiguration(userLocale, osLocale, nlsMetadataPath);
		}

		const languagePack = languagePacks[resolvedLanguage];
		const mainLanguagePackPath = languagePack?.translations?.['vscode'];
		// Pre-condition: Validate the integrity and existence of the resolved language pack.
		// Block Logic: Checks for the presence of language pack metadata, hash, translations,
		// and the existence of the main translation file. If any check fails, fall back to default.
		if (
			!languagePack ||
			typeof languagePack.hash !== 'string' ||
			!languagePack.translations ||
			typeof mainLanguagePackPath !== 'string' ||
			!(await exists(mainLanguagePackPath))
		) {
			return defaultNLSConfiguration(userLocale, osLocale, nlsMetadataPath);
		}

		// Construct paths for caching language pack data.
		const languagePackId = `${languagePack.hash}.${resolvedLanguage}`;
		const globalLanguagePackCachePath = path.join(userDataPath, 'clp', languagePackId);
		const commitLanguagePackCachePath = path.join(globalLanguagePackCachePath, commit);
		const languagePackMessagesFile = path.join(commitLanguagePackCachePath, 'nls.messages.json');
		const translationsConfigFile = path.join(globalLanguagePackCachePath, 'tcf.json');
		const languagePackCorruptMarkerFile = path.join(globalLanguagePackCachePath, 'corrupted.info');

		// Pre-condition: Check if the language pack cache is marked as corrupted.
		// Block Logic: If a corrupted marker file exists, delete the entire cache directory
		// to force regeneration of the language pack.
		if (await exists(languagePackCorruptMarkerFile)) {
			await fs.promises.rm(globalLanguagePackCachePath, { recursive: true, force: true, maxRetries: 3 }); // delete corrupted cache folder
		}

		// Construct the NLS configuration object.
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

			// NLS: below properties are a relic from old times only used by vscode-nls and deprecated
			locale: userLocale,
			availableLanguages: { '*': resolvedLanguage },
			_languagePackId: languagePackId,
			_languagePackSupport: true,
			_translationsConfigFile: translationsConfigFile,
			_cacheRoot: globalLanguagePackCachePath,
			_resolvedLanguagePackCoreLocation: commitLanguagePackCachePath,
			_corruptedFile: languagePackCorruptMarkerFile
		};

		// Pre-condition: Check if the language pack cache for the current commit already exists.
		// Block Logic: If the cache exists, simply update its access time and return the result,
		// avoiding regeneration of translation files.
		if (await exists(commitLanguagePackCachePath)) {
			touch(commitLanguagePackCachePath).catch(() => { }); // We don't wait for this. No big harm if we can't touch
			perf.mark('code/didGenerateNls');
			return result;
		}

		// Block Logic: If no cached language pack exists, proceed to generate translation files.
		// This involves creating the cache directory, reading default keys and messages,
		// and merging with language pack data to produce the final NLS messages.
		const [
			,
			nlsDefaultKeys,
			nlsDefaultMessages,
			nlsPackdata
		]:
			[unknown, Array<[string, string[]]>, string[], { contents: Record<string, Record<string, string>> }]
			//               ^moduleId ^nlsKeys                               ^moduleId      ^nlsKey ^nlsValue
			= await Promise.all([
				fs.promises.mkdir(commitLanguagePackCachePath, { recursive: true }),
				fs.promises.readFile(path.join(nlsMetadataPath, 'nls.keys.json'), 'utf-8').then(content => JSON.parse(content)),
				fs.promises.readFile(path.join(nlsMetadataPath, 'nls.messages.json'), 'utf-8').then(content => JSON.parse(content)),
				fs.promises.readFile(mainLanguagePackPath, 'utf-8').then(content => JSON.parse(content)),
			]);

		const nlsResult: string[] = [];

		/**
		 * @brief Block Logic: Merges default NLS keys with language pack translations.
		 *
		 * It constructs a flat array of translated messages, falling back to default
		 * messages if a translation is not available for a given key.
		 * Invariant: `nlsResult` accumulates translated strings in the correct order.
		 */
		let nlsIndex = 0;
		for (const [moduleId, nlsKeys] of nlsDefaultKeys) {
			const moduleTranslations = nlsPackdata.contents[moduleId];
			for (const nlsKey of nlsKeys) {
				nlsResult.push(moduleTranslations?.[nlsKey] || nlsDefaultMessages[nlsIndex]);
				nlsIndex++;
			}
		}

		// Write the generated NLS message file and translations config file to cache.
		await Promise.all([
			fs.promises.writeFile(languagePackMessagesFile, JSON.stringify(nlsResult), 'utf-8'),
			fs.promises.writeFile(translationsConfigFile, JSON.stringify(languagePack.translations), 'utf-8')
		]);

		perf.mark('code/didGenerateNls');

		return result;
	} catch (error) {
		// Block Logic: If an error occurs during NLS configuration resolution or file generation,
		// log the error and fall back to the default NLS configuration to prevent application failure.
		console.error('Generating translation files failed.', error);
	}

	return defaultNLSConfiguration(userLocale, osLocale, nlsMetadataPath);
}

/**
 * @function getLanguagePackConfigurations
 * @description Reads and parses the `languagepacks.json` file from the user data directory.
 * This file contains metadata about all installed language extensions.
 *
 * @param {string} userDataPath - The path to the user data directory.
 * @returns {Promise<ILanguagePacks | undefined>} A promise that resolves to the
 *   language pack configurations, or `undefined` if the file cannot be read or parsed.
 */
async function getLanguagePackConfigurations(userDataPath: string): Promise<ILanguagePacks | undefined> {
	const configFile = path.join(userDataPath, 'languagepacks.json');
	try {
		return JSON.parse(await fs.promises.readFile(configFile, 'utf-8'));
	} catch (err) {
		return undefined; // Do nothing. If we can't read the file we have no language pack config.
	}
}

/**
 * @function resolveLanguagePackLanguage
 * @description Resolves the most specific language string from the available language packs
 * that matches the given locale.
 *
 * This function handles locale fallbacks (e.g., 'en-us' -> 'en').
 *
 * @param {ILanguagePacks} languagePacks - An object containing available language pack configurations.
 * @param {string | undefined} locale - The desired locale string (e.g., 'en-us', 'fr').
 * @returns {string | undefined} The resolved language string from `languagePacks`, or `undefined` if no match is found.
 */
function resolveLanguagePackLanguage(languagePacks: ILanguagePacks, locale: string | undefined): string | undefined {
	try {
		// Invariant: 'locale' is progressively truncated to find a matching language pack.
		while (locale) {
			// If a direct match for the locale is found in the language packs.
			if (languagePacks[locale]) {
				return locale;
			}

			// Block Logic: If no direct match, try a less specific locale (e.g., 'en-us' -> 'en').
			const index = locale.lastIndexOf('-');
			if (index > 0) {
				locale = locale.substring(0, index);
			} else {
				// No more specific locale parts to try.
				return undefined;
			}
		}
	} catch (error) {
		console.error('Resolving language pack configuration failed.', error);
	}

	return undefined;
}

/**
 * @function defaultNLSConfiguration
 * @description Provides a default NLS configuration when a specific language pack
 * cannot be resolved or when NLS resolution is explicitly skipped.
 *
 * @param {string} userLocale - The user's preferred locale.
 * @param {string} osLocale - The operating system's preferred locale.
 * @param {string} nlsMetadataPath - The base path for NLS metadata files.
 * @returns {INLSConfiguration} A default NLS configuration object, typically set to English.
 */
function defaultNLSConfiguration(userLocale: string, osLocale: string, nlsMetadataPath: string): INLSConfiguration {
	perf.mark('code/didGenerateNls');

	return {
		userLocale,
		osLocale,
		resolvedLanguage: 'en',
		defaultMessagesFile: path.join(nlsMetadataPath, 'nls.messages.json'),

		// NLS: below 2 are a relic from old times only used by vscode-nls and deprecated
		locale: userLocale,
		availableLanguages: {}
	};
}

//#region fs helpers

/**
 * @function exists
 * @description Checks if a file or directory exists at the given path.
 *
 * @param {string} path - The path to check.
 * @returns {Promise<boolean>} A promise that resolves to `true` if the path exists, `false` otherwise.
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
 * @function touch
 * @description Updates the access and modification times of a file at the given path.
 * This function does not wait for the operation to complete successfully.
 *
 * @param {string} path - The path of the file to touch.
 * @returns {Promise<void>} A promise that resolves when the operation is initiated.
 */
function touch(path: string): Promise<void> {
	const date = new Date();

	return fs.promises.utimes(path, date, date);
}

//#endregion