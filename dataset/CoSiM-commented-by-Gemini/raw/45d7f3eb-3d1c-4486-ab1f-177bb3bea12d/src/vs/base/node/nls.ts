/**
 * @fileoverview
 * This file implements the logic for resolving the National Language Support (NLS)
 * configuration for the application. It is responsible for determining the appropriate
 * language pack to use based on user locale, OS locale, and available language packs.
 * The module handles the caching of processed language pack data to optimize startup
 * performance. This is a core part of the internationalization (i18n) strategy.
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
 * @interface IResolveNLSConfigurationContext
 * @description Defines the context required to resolve the NLS configuration. This
 * includes paths to necessary metadata and data directories, as well as locale information.
 */
export interface IResolveNLSConfigurationContext {

	/**
	 * @property
	 * @description Location where `nls.messages.json` and `nls.keys.json` are stored.
	 * These files contain the default English strings and the keys used for localization.
	 */
	readonly nlsMetadataPath: string;

	/**
	 * @property
	 * @description Path to the user data directory. This directory is used to cache
	 * language packs that have been converted into an optimized format for consumption.
	 */
	readonly userDataPath: string;

	/**
	 * @property
	 * @description The commit hash of the running application. This is used to ensure
	 * that cached language packs are compatible with the current version of the application.
	 * It can be `undefined` in development environments.
	 */
	readonly commit: string | undefined;

	/**
	 * @property
	 * @description The locale as specified by the user, either through `argv.json` or
	 * via `app.getLocale()`. This is the primary driver for language selection.
	 */
	readonly userLocale: string;

	/**
	 * @property
	 * @description The locale as determined by the operating system's language preferences
	 * (e.g., from `app.getPreferredSystemLanguages()`). This can be used as a fallback
	 * or for secondary locale information.
	 */
	readonly osLocale: string;
}

/**
 * @function resolveNLSConfiguration
 * @description Resolves the NLS configuration by selecting an appropriate language pack
 * based on the provided context. It manages the caching of language packs and generates
 * the necessary configuration for the application to use localized strings.
 * @param {IResolveNLSConfigurationContext} context The context for resolving the NLS configuration.
 * @returns {Promise<INLSConfiguration>} A promise that resolves to the NLS configuration.
 */
export async function resolveNLSConfiguration({ userLocale, osLocale, userDataPath, commit, nlsMetadataPath }: IResolveNLSConfigurationContext): Promise<INLSConfiguration> {
	perf.mark('code/willGenerateNls');

	/**
	 * @block
	 * @description This block serves as a fast path for scenarios where localization is not needed
	 * or not possible. This includes development mode, pseudo-localization, English locales, or when
	 * essential information like commit hash or user data path is missing.
	 */
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
		const languagePacks = await getLanguagePackConfigurations(userDataPath);
		/**
		 * @block
		 * @description If no language packs are configured, the function returns the default NLS configuration.
		 * This is a common case when no language pack extensions are installed.
		 */
		if (!languagePacks) {
			return defaultNLSConfiguration(userLocale, osLocale, nlsMetadataPath);
		}

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
			return defaultNLSConfiguration(userLocale, osLocale, nlsMetadataPath);
		}

		const languagePackId = `${languagePack.hash}.${resolvedLanguage}`;
		const globalLanguagePackCachePath = path.join(userDataPath, 'clp', languagePackId);
		const commitLanguagePackCachePath = path.join(globalLanguagePackCachePath, commit);
		const languagePackMessagesFile = path.join(commitLanguagePackCachePath, 'nls.messages.json');
		const translationsConfigFile = path.join(globalLanguagePackCachePath, 'tcf.json');
		const languagePackCorruptMarkerFile = path.join(globalLanguagePackCachePath, 'corrupted.info');

		/**
		 * @block
		 * @description Pre-condition: Checks for a corruption marker file.
		 * If the marker exists, it indicates that a previous attempt to generate the cache failed.
		 * The corrupted cache is deleted to allow for a clean regeneration attempt.
		 */
		if (await exists(languagePackCorruptMarkerFile)) {
			await fs.promises.rm(globalLanguagePackCachePath, { recursive: true, force: true, maxRetries: 3 }); // delete corrupted cache folder
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

		/**
		 * @block
		 * @description Checks if a cached version of the language pack for the current commit already exists.
		 * If it does, the function can return the configuration immediately, which is a significant
		 * performance optimization. It also touches the directory to update its modification time.
		 */
		if (await exists(commitLanguagePackCachePath)) {
			touch(commitLanguagePackCachePath).catch(() => { }); // We don't wait for this. No big harm if we can't touch
			perf.mark('code/didGenerateNls');
			return result;
		}

		/**
		 * @block
		 * @description This block is responsible for generating the cached NLS data. It reads the default
		 * English messages, the localization keys, and the language pack data. It then creates
		 * the directory structure for the cache.
		 */
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
		 * @block
		 * @description This loop constructs the final array of localized messages.
		 * Invariant: The `nls.keys.json` file provides the canonical ordering of NLS messages.
		 * The loop iterates through this structure, looks up the translation for each key,
		 * and falls back to the default English message if a translation is not found.
		 */
		let nlsIndex = 0;
		for (const [moduleId, nlsKeys] of nlsDefaultKeys) {
			const moduleTranslations = nlsPackdata.contents[moduleId];
			for (const nlsKey of nlsKeys) {
				nlsResult.push(moduleTranslations?.[nlsKey] || nlsDefaultMessages[nlsIndex]);
				nlsIndex++;
			}
		}

		/**
		 * @block
		 * @description After generating the localized messages, this block writes the results
		 * to the cache directory. This includes the `nls.messages.json` file containing the
		 * localized strings and the `tcf.json` file with translation configurations.
		 */
		await Promise.all([
			fs.promises.writeFile(languagePackMessagesFile, JSON.stringify(nlsResult), 'utf-8'),
			fs.promises.writeFile(translationsConfigFile, JSON.stringify(languagePack.translations), 'utf-8')
		]);

		perf.mark('code/didGenerateNls');

		return result;
	} catch (error) {
		console.error('Generating translation files failed.', error);
	}

	return defaultNLSConfiguration(userLocale, osLocale, nlsMetadataPath);
}

/**
 * @function getLanguagePackConfigurations
 * @description Retrieves language pack configurations from `languagepacks.json`.
 * This file acts as a registry for all installed language packs, mapping language
 * identifiers to their corresponding translation files.
 * @param {string} userDataPath The path to the user data directory where `languagepacks.json` is located.
 * @returns {Promise<ILanguagePacks | undefined>} A promise that resolves to the parsed language pack
 * configurations, or `undefined` if the file cannot be read or parsed.
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
 * @description Resolves a locale to a supported language pack language. It uses a fallback
 * strategy, trimming the locale (e.g., from `de-DE` to `de`) until a match is found
 * in the available language packs.
 * @param {ILanguagePacks} languagePacks The available language packs.
 * @param {(string | undefined)} locale The locale to resolve.
 * @returns {(string | undefined)} The resolved language, or `undefined` if no match is found.
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
				return locale;
			}

			const index = locale.lastIndexOf('-');
			if (index > 0) {
				locale = locale.substring(0, index);
			} else {
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
 * @description Returns a default NLS configuration for the English language. This is used
 * as a fallback when no suitable language pack can be found or when localization is
 * explicitly disabled.
 * @param {string} userLocale The user's specified locale.
 * @param {string} osLocale The operating system's locale.
 * @param {string} nlsMetadataPath The path to the NLS metadata directory.
 * @returns {INLSConfiguration} The default NLS configuration.
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
 * @param {string} path The path to check.
 * @returns {Promise<boolean>} A promise that resolves to `true` if the path exists, and `false` otherwise.
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
 * @description Updates the modification and access times of a file or directory.
 * This is used to signal that a cached resource is still in use.
 * @param {string} path The path to the file or directory to touch.
 * @returns {Promise<void>} A promise that resolves when the operation is complete.
 */
function touch(path: string): Promise<void> {
	const date = new Date();

	return fs.promises.utimes(path, date, date);
}

//#endregion