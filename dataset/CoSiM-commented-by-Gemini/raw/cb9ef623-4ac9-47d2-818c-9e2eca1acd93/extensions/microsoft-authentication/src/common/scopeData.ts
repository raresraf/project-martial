/**
 * @file scopeData.ts
 * @brief Manages and processes authentication scopes for Microsoft authentication.
 * @copyright Copyright (c) Microsoft Corporation. All rights reserved.
 * @license MIT
 *
 * This file defines the `ScopeData` class, which is responsible for parsing
 * and managing the scopes used in an authentication request. It handles the
 * extraction of client ID and tenant information from special VS Code-internal
 * scopes, and ensures that the necessary default OIDC scopes are included.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { Uri } from 'vscode';

const DEFAULT_CLIENT_ID = 'aebc6443-996d-45c2-90f0-388ff96faa56';
const DEFAULT_TENANT = 'organizations';

const OIDC_SCOPES = ['openid', 'email', 'profile', 'offline_access'];
const GRAPH_TACK_ON_SCOPE = 'User.Read';

/**
 * @class ScopeData
 * @brief A class to manage and process authentication scopes.
 *
 * This class takes an array of scopes and an optional authorization server URI,
 * and from them, it derives the full set of scopes, the client ID, and the
 * tenant for an authentication request.
 */
export class ScopeData {
	/**
	 * The full list of scopes including:
	 * * the original scopes passed to the constructor
	 * * internal VS Code scopes (e.g. `VSCODE_CLIENT_ID:...`)
	 * * the default scopes (`openid`, `email`, `profile`, `offline_access`)
	 */
	readonly allScopes: string[];

	/**
	 * The full list of scopes as a space-separated string. For logging.
	 */
	readonly scopeStr: string;

	/**
	 * The list of scopes to send to the token endpoint. This is the same as `scopes` but without the internal VS Code scopes.
	 */
	readonly scopesToSend: string[];

	/**
	 * The client ID to use for the token request. This is the value of the `VSCODE_CLIENT_ID:...` scope if present, otherwise the default client ID.
	 */
	readonly clientId: string;

	/**
	 * The tenant ID or `organizations`, `common`, `consumers` to use for the token request. This is the value of the `VSCODE_TENANT:...` scope if present, otherwise it's the default.
	 */
	readonly tenant: string;

	/**
	 * The tenant ID to use for the token request. This will only ever be a GUID if one was specified via the `VSCODE_TENANT:...` scope, otherwise undefined.
	 */
	readonly tenantId: string | undefined;

	constructor(readonly originalScopes: readonly string[] = [], authorizationServer?: Uri) {
		const modifiedScopes = [...originalScopes];
		modifiedScopes.sort();
		this.allScopes = modifiedScopes;
		this.scopeStr = modifiedScopes.join(' ');
		this.scopesToSend = this.getScopesToSend(modifiedScopes);
		this.clientId = this.getClientId(this.allScopes);
		this.tenant = this.getTenant(this.allScopes, authorizationServer);
		this.tenantId = this.getTenantId(this.tenant);
	}

	/**
	 * @brief Extracts the client ID from the scopes.
	 * @param scopes The array of scopes.
	 * @return The client ID, or the default client ID if not found.
	 */
	private getClientId(scopes: string[]): string {
		return scopes.reduce<string | undefined>((prev, current) => {
			if (current.startsWith('VSCODE_CLIENT_ID:')) {
				return current.split('VSCODE_CLIENT_ID:')[1];
			}
			return prev;
		}, undefined) ?? DEFAULT_CLIENT_ID;
	}

	/**
	 * @brief Extracts the tenant from the scopes or the authorization server URI.
	 * @param scopes The array of scopes.
	 * @param authorizationServer The authorization server URI.
	 * @return The tenant, or the default tenant if not found.
	 */
	private getTenant(scopes: string[], authorizationServer?: Uri): string {
		// Invariant: The tenant can be specified either in the path of the
		// authorization server URI or as a special `VSCODE_TENANT:` scope.
		if (authorizationServer?.path) {
			// Get tenant portion of URL
			const tenant = authorizationServer.path.split('/')[1];
			if (tenant) {
				return tenant;
			}
		}
		return scopes.reduce<string | undefined>((prev, current) => {
			if (current.startsWith('VSCODE_TENANT:')) {
				return current.split('VSCODE_TENANT:')[1];
			}
			return prev;
		}, undefined) ?? DEFAULT_TENANT;
	}

	/**
	 * @brief Extracts the tenant ID from the tenant string.
	 * @param tenant The tenant string.
	 * @return The tenant ID, or undefined if the tenant is a common endpoint.
	 */
	private getTenantId(tenant: string): string | undefined {
		switch (tenant) {
			case 'organizations':
			case 'common':
			case 'consumers':
				// These are not valid tenant IDs, so we return undefined
				return undefined;
			default:
				return this.tenant;
		}
	}

	/**
	 * @brief Filters the scopes to be sent to the token endpoint.
	 * @param scopes The array of all scopes.
	 * @return The filtered array of scopes.
	 *
	 * This function removes any VS Code-internal scopes and ensures that if
	 * only OIDC scopes are present, a tack-on scope is added to make the
	 * request valid for the Graph API.
	 */
	private getScopesToSend(scopes: string[]): string[] {
		const scopesToSend = scopes.filter(s => !s.startsWith('VSCODE_'));

		const set = new Set(scopesToSend);
		for (const scope of OIDC_SCOPES) {
			set.delete(scope);
		}

		// If we only had OIDC scopes, we need to add a tack-on scope to make the request valid
		// by forcing Identity into treating this as a Graph token request.
		if (!set.size) {
			scopesToSend.push(GRAPH_TACK_ON_SCOPE);
		}
		return scopesToSend;
	}
}