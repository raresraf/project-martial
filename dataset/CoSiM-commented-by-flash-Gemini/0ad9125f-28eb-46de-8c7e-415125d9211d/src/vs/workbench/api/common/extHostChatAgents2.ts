/**
 * @file extHostChatAgents2.ts
 * @brief Implements the VS Code Extension Host (ExtHost) side management for chat agents (participants).
 *
 * This file provides the core logic for extensions to register and interact
 * with chat participants, handle incoming chat requests, stream responses,
 * provide follow-up actions, and manage related UI elements within VS Code.
 * It acts as an intermediary between VS Code extensions and the Main Process
 * (`MainThreadChatAgentsShape2`).
 *
 * Architecture:
 * - `ExtHostChatAgents2`: The main class managing the lifecycle and interaction
 *   of chat agents registered by extensions. It handles IPC with the Main Process.
 * - `ExtHostChatAgent`: A wrapper class that adapts an extension's `vscode.ChatParticipant`
 *   implementation to the ExtHost's internal management, handling metadata updates
 *   and event dispatching.
 * - `ChatAgentResponseStream`: Manages the streaming of responses from an extension's
 *   chat agent handler back to the Main Process, providing the `vscode.ChatResponseStream`
 *   API object to the extension.
 * - `ExtHostParticipantDetector`: Wraps `vscode.ChatParticipantDetectionProvider` for the ExtHost.
 * - `ExtHostRelatedFilesProvider`: Wraps `vscode.ChatRelatedFilesProvider` for the ExtHost.
 *
 * Core Functionalities:
 * - **Registration:** Extensions register `ChatParticipant`s, `ChatParticipantDetectionProvider`s,
 *   and `ChatRelatedFilesProvider`s with the ExtHost.
 * - **Invocation:** Handles `IChatAgentRequest`s from the Main Process, converting them
 *   into `vscode.ChatRequest` for extensions, and invoking the extension's handler.
 * - **Response Streaming:** Provides an API for extensions to stream various types of
 *   response parts (markdown, codeblocks, references, buttons, progress, etc.) back
 *   to the chat UI.
 * - **Synchronization:** Manages `CancellationToken`s for request cancellation and
 *   coordinates state updates with the Main Process.
 * - **Feedback & Actions:** Dispatches feedback (helpful/unhelpful) and user actions
 *   (button clicks) to the registered chat agents.
 * - **History Management:** Prepares chat history for agent invocation and follow-up
 *   provision.
 * - **Language Model Integration:** Fetches appropriate language models for chat requests.
 */

import type * as vscode from 'vscode'; // Type definitions for VS Code API.
import { coalesce } from '../../../base/common/arrays.js'; // Utility to filter out null/undefined values from an array.
import { raceCancellation } from '../../../base/common/async.js'; // Utility to race a promise against a cancellation token.
import { CancellationToken, CancellationTokenSource } from '../../../base/common/cancellation.js'; // Cancellation token and source for managing cancellable operations.
import { toErrorMessage } from '../../../base/common/errorMessage.js'; // Utility to convert an error object into a human-readable message.
import { Emitter } from '../../../base/common/event.js'; // Event emitter for custom events.
import { Iterable } from '../../../base/common/iterator.js'; // Iterable utilities.
import { Disposable, DisposableMap, DisposableStore, toDisposable } from '../../../base/common/lifecycle.js'; // Utilities for managing disposable objects.
import { revive } from '../../../base/common/marshalling.js'; // Utility to revive data from transferrable format.
import { StopWatch } from '../../../base/common/stopwatch.js'; // Utility for measuring elapsed time.
import { ThemeIcon } from '../../../base/common/themables.js'; // Theme icon definitions.
import { assertType } from '../../../base/common/types.js'; // Type assertion utility.
import { URI } from '../../../base/common/uri.js'; // URI (Uniform Resource Identifier) object.
import { generateUuid } from '../../../base/common/uuid.js'; // Utility for generating UUIDs.
import { Location } from '../../../editor/common/languages.js'; // Location type definition.
import { ExtensionIdentifier, IExtensionDescription, IRelaxedExtensionDescription } from '../../../platform/extensions/common/extensions.js'; // Extension identification and description types.
import { ILogService } from '../../../platform/log/common/log.js'; // Logging service interface.
import { ChatAgentLocation, IChatAgentRequest, IChatAgentResult, IChatAgentResultTimings, IChatWelcomeMessageContent } from '../../contrib/chat/common/chatAgents.js'; // Chat agent related types (Main Process side).
import { ChatAgentVoteDirection, IChatContentReference, IChatFollowup, IChatResponseErrorDetails, IChatUserActionEvent, IChatVoteAction } from '../../contrib/chat/common/chatService.js'; // Chat service related types.
import { checkProposedApiEnabled, isProposedApiEnabled } from '../../services/extensions/common/extensions.js'; // Proposed API enablement checks.
import { Dto } from '../../services/extensions/common/proxyIdentifier.js'; // Data Transfer Object (DTO) type.
import { ExtHostChatAgentsShape2, IChatAgentCompletionItem, IChatAgentHistoryEntryDto, IChatProgressDto, IExtensionChatAgentMetadata, IMainContext, MainContext, MainThreadChatAgentsShape2 } from './extHost.protocol.js'; // IPC protocol shapes.
import { CommandsConverter, ExtHostCommands } from './extHostCommands.js'; // ExtHost Commands and converter.
import { ExtHostDocuments } from './extHostDocuments.js'; // ExtHost Documents.
import { ExtHostLanguageModels } from './extHostLanguageModels.js'; // ExtHost Language Models.
import * as typeConvert from './extHostTypeConverters.js'; // Type conversion utilities between ExtHost and Main Process.
import * as extHostTypes from './extHostTypes.js'; // ExtHost specific types (extension-facing).
import { isChatViewTitleActionContext } from '../../contrib/chat/common/chatActions.js'; // Chat actions related context checking.
import { IChatRelatedFile, IChatRequestDraft } from '../../contrib/chat/common/chatEditingService.js'; // Chat editing related types.
import { ExtHostDiagnostics } from './extHostDiagnostics.js'; // ExtHost Diagnostics.

/**
 * @class ChatAgentResponseStream
 * @brief Manages a single chat agent response stream, enabling extensions to send
 * incremental updates back to the VS Code UI.
 * Functional Utility: Provides the `vscode.ChatResponseStream` API object to extensions,
 * encapsulating the logic for sending various types of chat response parts (markdown,
 * references, commands, etc.) to the Main Process via IPC. It also tracks timings
 * for performance analysis.
 */
class ChatAgentResponseStream {

	private _stopWatch = StopWatch.create(false); // Measures elapsed time for the response.
	private _isClosed: boolean = false; // Flag indicating if the stream has been closed.
	private _firstProgress: number | undefined; // Timestamp of the first meaningful progress update.
	private _apiObject: vscode.ChatResponseStream | undefined; // The API object exposed to extensions.

	/**
	 * @brief Constructs a new `ChatAgentResponseStream`.
	 * @param _extension The extension description.
	 * @param _request The original chat agent request.
	 * @param _proxy The proxy to the MainThreadChatAgents2.
	 * @param _commandsConverter The command converter utility.
	 * @param _sessionDisposables A DisposableStore for session-specific disposables.
	 */
	constructor(
		private readonly _extension: IExtensionDescription,
		private readonly _request: IChatAgentRequest,
		private readonly _proxy: MainThreadChatAgentsShape2,
		private readonly _commandsConverter: CommandsConverter,
		private readonly _sessionDisposables: DisposableStore
	) { }

	/**
	 * @brief Marks the response stream as closed.
	 * Functional Utility: Prevents further writes to the stream after it has been explicitly closed,
	 * ensuring proper state management.
	 */
	close() {
		this._isClosed = true;
	}

	/**
	 * @brief Returns timing information for the response stream.
	 * Functional Utility: Provides metrics such as the time to the first progress update
	 * and the total elapsed time for the response.
	 */
	get timings(): IChatAgentResultTimings {
		return {
			firstProgress: this._firstProgress,
			totalElapsed: this._stopWatch.elapsed()
		};
	}

	/**
	 * @brief Gets the `vscode.ChatResponseStream` API object exposed to extensions.
	 * Functional Utility: This getter constructs and memoizes the API object, which
	 * provides methods for streaming various types of chat response parts. Each method
	 * sends a Data Transfer Object (DTO) to the Main Process via the `_proxy`.
	 */
	get apiObject() {

		if (!this._apiObject) {

			const that = this; // Capture `this` for use in nested functions.
			this._stopWatch.reset(); // Reset stopwatch on first access.

			/**
			 * @brief Throws an error if the response stream is already closed.
			 * @param source The function that called this check, for better stack traces.
			 */
			function throwIfDone(source: Function | undefined) {
				if (that._isClosed) {
					const err = new Error('Response stream has been closed');
					Error.captureStackTrace(err, source);
					throw err;
				}
			}

			/**
			 * @brief Internal helper to report progress chunks to the Main Process.
			 * @param progress The progress DTO to send.
			 * @param task An optional task function for progress with tasks.
			 */
			const _report = (progress: IChatProgressDto, task?: (progress: vscode.Progress<vscode.ChatResponseWarningPart | vscode.ChatResponseReferencePart>) => Thenable<string | void>) => {
				// Measure the time to the first progress update with real markdown content
				// Block Logic: Record the time of the first meaningful progress (markdown content).
				if (typeof this._firstProgress === 'undefined' && (progress.kind === 'markdownContent' || progress.kind === 'markdownVuln')) {
					this._firstProgress = this._stopWatch.elapsed();
				}

				if (task) {
					// Block Logic: Handle progress reports that involve a task, sending task updates.
					const progressReporterPromise = this._proxy.$handleProgressChunk(this._request.requestId, progress);
					const progressReporter = {
						report: (p: vscode.ChatResponseWarningPart | vscode.ChatResponseReferencePart) => {
							progressReporterPromise?.then((handle) => {
								if (handle) {
									if (extHostTypes.MarkdownString.isMarkdownString(p.value)) {
										this._proxy.$handleProgressChunk(this._request.requestId, typeConvert.ChatResponseWarningPart.from(<vscode.ChatResponseWarningPart>p), handle);
									} else {
										this._proxy.$handleProgressChunk(this._request.requestId, typeConvert.ChatResponseReferencePart.from(<vscode.ChatResponseReferencePart>p), handle);
									}
								}
							});
						}
					};

					Promise.all([progressReporterPromise, task?.(progressReporter)]).then(([handle, res]) => {
						if (handle !== undefined) {
							this._proxy.$handleProgressChunk(this._request.requestId, typeConvert.ChatTaskResult.from(res), handle);
						}
					});
				} else {
					// Block Logic: Handle simple progress reports without an associated task.
					this._proxy.$handleProgressChunk(this._request.requestId, progress);
				}
			};

			this._apiObject = Object.freeze<vscode.ChatResponseStream>({
				/**
				 * @brief Appends markdown content to the chat response.
				 * @param value The markdown string.
				 */
				markdown(value) {
					throwIfDone(this.markdown);
					const part = new extHostTypes.ChatResponseMarkdownPart(value);
					const dto = typeConvert.ChatResponseMarkdownPart.from(part);
					_report(dto);
					return this;
				},
				/**
				 * @brief Appends markdown content with associated vulnerability information.
				 * @param value The markdown string.
				 * @param vulnerabilities Optional array of vulnerability information.
				 */
				markdownWithVulnerabilities(value, vulnerabilities) {
					throwIfDone(this.markdown);
					if (vulnerabilities) {
						checkProposedApiEnabled(that._extension, 'chatParticipantAdditions');
					}

					const part = new extHostTypes.ChatResponseMarkdownWithVulnerabilitiesPart(value, vulnerabilities);
					const dto = typeConvert.ChatResponseMarkdownWithVulnerabilitiesPart.from(part);
					_report(dto);
					return this;
				},
				/**
				 * @brief Appends a codeblock URI to the chat response.
				 * @param value The URI of the codeblock.
				 */
				codeblockUri(value) {
					throwIfDone(this.codeblockUri);
					checkProposedApiEnabled(that._extension, 'chatParticipantAdditions');
					const part = new extHostTypes.ChatResponseCodeblockUriPart(value);
					const dto = typeConvert.ChatResponseCodeblockUriPart.from(part);
					_report(dto);
					return this;
				},
				/**
				 * @brief Appends a file tree structure to the chat response.
				 * @param value The file tree data.
				 * @param baseUri The base URI for the file tree.
				 */
				filetree(value, baseUri) {
					throwIfDone(this.filetree);
					const part = new extHostTypes.ChatResponseFileTreePart(value, baseUri);
					const dto = typeConvert.ChatResponseFilesPart.from(part);
					_report(dto);
					return this;
				},
				/**
				 * @brief Appends an anchor (link) to the chat response.
				 * @param value The URI or location of the anchor.
				 * @param title Optional title for the anchor.
				 */
				anchor(value, title?: string) {
					const part = new extHostTypes.ChatResponseAnchorPart(value, title);
					return this.push(part);
				},
				/**
				 * @brief Appends a command button to the chat response.
				 * @param value The command object for the button.
				 */
				button(value) {
					throwIfDone(this.anchor);
					const part = new extHostTypes.ChatResponseCommandButtonPart(value);
					const dto = typeConvert.ChatResponseCommandButtonPart.from(part, that._commandsConverter, that._sessionDisposables);
					_report(dto);
					return this;
				},
				/**
				 * @brief Reports progress with a message and an optional task.
				 * @param value The progress message.
				 * @param task An optional task function to be executed as part of progress.
				 */
				progress(value, task?: ((progress: vscode.Progress<vscode.ChatResponseWarningPart>) => Thenable<string | void>)) {
					throwIfDone(this.progress);
					const part = new extHostTypes.ChatResponseProgressPart2(value, task);
					const dto = task ? typeConvert.ChatTask.from(part) : typeConvert.ChatResponseProgressPart.from(part);
					_report(dto, task);
					return this;
				},
				/**
				 * @brief Appends a warning message to the chat response.
				 * @param value The warning message.
				 */
				warning(value) {
					throwIfDone(this.progress);
					checkProposedApiEnabled(that._extension, 'chatParticipantAdditions');
					const part = new extHostTypes.ChatResponseWarningPart(value);
					const dto = typeConvert.ChatResponseWarningPart.from(part);
					_report(dto);
					return this;
				},
				/**
				 * @brief Appends a reference to the chat response (deprecated, use `reference2`).
				 * @param value The URI or location of the reference.
				 * @param iconPath Optional icon path for the reference.
				 */
				reference(value, iconPath) {
					return this.reference2(value, iconPath);
				},
				/**
				 * @brief Appends a reference to the chat response with more options.
				 * @param value The URI, location, or variable reference.
				 * @param iconPath Optional icon path.
				 * @param options Optional reference options.
				 */
				reference2(value, iconPath, options) {
					throwIfDone(this.reference);

					if (typeof value === 'object' && 'variableName' in value) {
						checkProposedApiEnabled(that._extension, 'chatParticipantAdditions');
					}

					// Block Logic: Handle variable references that might need to pull in existing references.
					if (typeof value === 'object' && 'variableName' in value && !value.value) {
						// The participant used this variable. Does that variable have any references to pull in?
						const matchingVarData = that._request.variables.variables.find(v => v.name === value.variableName);
						if (matchingVarData) {
							let references: Dto<IChatContentReference>[] | undefined;
							if (matchingVarData.references?.length) {
								references = matchingVarData.references.map(r => ({
									kind: 'reference',
									reference: { variableName: value.variableName, value: r.reference as URI | Location }
								} satisfies IChatContentReference));
							} else {
								// Participant sent a variableName reference but the variable produced no references. Show variable reference with no value
								const part = new extHostTypes.ChatResponseReferencePart(value, iconPath, options);
								const dto = typeConvert.ChatResponseReferencePart.from(part);
								references = [dto];
							}

							references.forEach(r => _report(r));
							return this;
						} else {
							// Something went wrong- that variable doesn't actually exist
						}
					} else {
						// Block Logic: Handle direct URI/Location references.
						const part = new extHostTypes.ChatResponseReferencePart(value, iconPath, options);
						const dto = typeConvert.ChatResponseReferencePart.from(part);
						_report(dto);
					}

					return this;
				},
				/**
				 * @brief Appends a code citation to the chat response.
				 * @param value The URI of the cited code.
				 * @param license The license of the cited code.
				 * @param snippet The code snippet.
				 */
				codeCitation(value: vscode.Uri, license: string, snippet: string): void {
					throwIfDone(this.codeCitation);
					checkProposedApiEnabled(that._extension, 'chatParticipantAdditions');

					const part = new extHostTypes.ChatResponseCodeCitationPart(value, license, snippet);
					const dto = typeConvert.ChatResponseCodeCitationPart.from(part);
					_report(dto);
				},
				/**
				 * @brief Appends a text edit suggestion to the chat response.
				 * @param target The target URI for the edit.
				 * @param edits The text edits or `true` if edits are done.
				 */
				textEdit(target, edits) {
					throwIfDone(this.textEdit);
					checkProposedApiEnabled(that._extension, 'chatParticipantAdditions');

					const part = new extHostTypes.ChatResponseTextEditPart(target, edits);
					part.isDone = edits === true ? true : undefined;
					const dto = typeConvert.ChatResponseTextEditPart.from(part);
					_report(dto);
					return this;
				},
				/**
				 * @brief Appends a notebook edit suggestion to the chat response.
				 * @param target The target URI for the notebook edit.
				 * @param edits The notebook edits.
				 */
				notebookEdit(target, edits) {
					throwIfDone(this.notebookEdit);
					checkProposedApiEnabled(that._extension, 'chatParticipantAdditions');

					const part = new extHostTypes.ChatResponseNotebookEditPart(target, edits);
					const dto = typeConvert.ChatResponseNotebookEditPart.from(part);
					_report(dto);
					return this;
				},
				/**
				 * @brief Appends a confirmation prompt to the chat response.
				 * @param title The title of the confirmation.
				 * @param message The message of the confirmation.
				 * @param data Optional data associated with the confirmation.
				 * @param buttons Optional buttons for the confirmation.
				 */
				confirmation(title, message, data, buttons) {
					throwIfDone(this.confirmation);
					checkProposedApiEnabled(that._extension, 'chatParticipantAdditions');

					const part = new extHostTypes.ChatResponseConfirmationPart(title, message, data, buttons);
					const dto = typeConvert.ChatResponseConfirmationPart.from(part);
					_report(dto);
					return this;
				},
				/**
				 * @brief Pushes a generic chat response part.
				 * @param part The chat response part to push.
				 */
				push(part) {
					throwIfDone(this.push);

					// Block Logic: Check proposed API enablement for various part types.
					if (
						part instanceof extHostTypes.ChatResponseTextEditPart ||
						part instanceof extHostTypes.ChatResponseNotebookEditPart ||
						part instanceof extHostTypes.ChatResponseMarkdownWithVulnerabilitiesPart ||
						part instanceof extHostTypes.ChatResponseWarningPart ||
						part instanceof extHostTypes.ChatResponseConfirmationPart ||
						part instanceof extHostTypes.ChatResponseCodeCitationPart ||
						part instanceof extHostTypes.ChatResponseMovePart ||
						part instanceof extHostTypes.ChatResponseProgressPart2
					) {
						checkProposedApiEnabled(that._extension, 'chatParticipantAdditions');
					}

					// Block Logic: Handle specific part types for conversion and reporting.
					if (part instanceof extHostTypes.ChatResponseReferencePart) {
						// Ensure variable reference values get fixed up
						this.reference2(part.value, part.iconPath, part.options);
					} else if (part instanceof extHostTypes.ChatResponseProgressPart2) {
						const dto = part.task ? typeConvert.ChatTask.from(part) : typeConvert.ChatResponseProgressPart.from(part);
						_report(dto, part.task);
					} else if (part instanceof extHostTypes.ChatResponseAnchorPart) {
						const dto = typeConvert.ChatResponseAnchorPart.from(part);

						if (part.resolve) {
							checkProposedApiEnabled(that._extension, 'chatParticipantAdditions');

							dto.resolveId = generateUuid();

							const cts = new CancellationTokenSource();
							part.resolve(cts.token)
								.then(() => {
									const resolvedDto = typeConvert.ChatResponseAnchorPart.from(part);
									that._proxy.$handleAnchorResolve(that._request.requestId, dto.resolveId!, resolvedDto);
								})
								.then(() => cts.dispose(), () => cts.dispose());
							that._sessionDisposables.add(toDisposable(() => cts.dispose(true)));
						}
						_report(dto);
					} else {
						const dto = typeConvert.ChatResponsePart.from(part, that._commandsConverter, that._sessionDisposables);
						_report(dto);
					}

					return this;
				},
			});
		}

		return this._apiObject;
	}
}

/**
 * @interface InFlightChatRequest
 * @brief Represents a chat request that is currently being processed by an agent.
 * Functional Utility: Stores the request ID and the external host request object
 * for tracking purposes, allowing for management of its state (e.g., pausing).
 */
interface InFlightChatRequest {
	requestId: string;
	extRequest: vscode.ChatRequest;
}

/**
 * @class ExtHostChatAgents2
 * @brief The main class in the Extension Host for managing chat agents (participants).
 * Functional Utility: Handles the registration, invocation, and lifecycle management
 * of chat agents provided by extensions. It mediates communication between these
 * extensions and the Main Process.
 */
export class ExtHostChatAgents2 extends Disposable implements ExtHostChatAgentsShape2 {

	private static _idPool = 0; // Static counter for assigning unique handles to chat agents.

	private readonly _agents = new Map<number, ExtHostChatAgent>(); // Map of registered chat agents by their handle.
	private readonly _proxy: MainThreadChatAgentsShape2; // Proxy to communicate with the MainThreadChatAgents2 in the Main Process.

	private static _participantDetectionProviderIdPool = 0; // Static counter for assigning unique handles to participant detection providers.
	private readonly _participantDetectionProviders = new Map<number, ExtHostParticipantDetector>(); // Map of registered participant detection providers.

	private static _relatedFilesProviderIdPool = 0; // Static counter for assigning unique handles to related files providers.
	private readonly _relatedFilesProviders = new Map<number, ExtHostRelatedFilesProvider>(); // Map of registered related files providers.

	private readonly _sessionDisposables: DisposableMap<string, DisposableStore> = this._register(new DisposableMap()); // Disposables associated with chat sessions.
	private readonly _completionDisposables: DisposableMap<number, DisposableStore> = this._register(new DisposableMap()); // Disposables associated with completion providers.

	private readonly _inFlightRequests = new Set<InFlightChatRequest>(); // Set of currently in-flight chat requests.

	/**
	 * @brief Constructs a new `ExtHostChatAgents2` instance.
	 * @param mainContext The main context for IPC communication.
	 * @param _logService The logging service.
	 * @param _commands The ExtHost commands service.
	 * @param _documents The ExtHost documents service.
	 * @param _languageModels The ExtHost language models service.
	 * @param _diagnostics The ExtHost diagnostics service.
	 */
	constructor(
		mainContext: IMainContext,
		private readonly _logService: ILogService,
		private readonly _commands: ExtHostCommands,
		private readonly _documents: ExtHostDocuments,
		private readonly _languageModels: ExtHostLanguageModels,
		private readonly _diagnostics: ExtHostDiagnostics,
	) {
		super();
		this._proxy = mainContext.getProxy(MainContext.MainThreadChatAgents2); // Initialize proxy to Main Process.

		// Functional Utility: Register an argument processor to filter out specific command arguments.
		_commands.registerArgumentProcessor({
			processArgument: (arg) => {
				// Don't send this argument to extension commands
				if (isChatViewTitleActionContext(arg)) {
					return null;
				}

				return arg;
			}
		});
	}

	/**
	 * @brief Transfers the active chat session to a new workspace.
	 * @param newWorkspace The URI of the new workspace.
	 */
	transferActiveChat(newWorkspace: vscode.Uri): void {
		this._proxy.$transferActiveChatSession(newWorkspace);
	}

	/**
	 * @brief Creates a new chat agent (participant).
	 * Functional Utility: Registers a new chat agent with the ExtHost and Main Process,
	 * allowing an extension to contribute a chat participant.
	 * @param extension The extension description.
	 * @param id The unique ID of the chat agent.
	 * @param handler The request handler for the chat agent.
	 * @return The `vscode.ChatParticipant` API object.
	 */
	createChatAgent(extension: IExtensionDescription, id: string, handler: vscode.ChatExtendedRequestHandler): vscode.ChatParticipant {
		const handle = ExtHostChatAgents2._idPool++; // Assign a unique handle.
		const agent = new ExtHostChatAgent(extension, id, this._proxy, handle, handler); // Create ExtHost wrapper.
		this._agents.set(handle, agent); // Store agent.

		this._proxy.$registerAgent(handle, extension.identifier, id, {}, undefined); // Register with Main Process.
		return agent.apiAgent; // Return API object.
	}

	/**
	 * @brief Creates a new dynamic chat agent.
	 * Functional Utility: Registers a dynamic chat agent with additional metadata,
	 * similar to `createChatAgent`. Dynamic agents have specific properties like `isSticky`.
	 * @param extension The extension description.
	 * @param id The unique ID of the chat agent.
	 * @param dynamicProps Dynamic properties for the agent.
	 * @param handler The request handler for the chat agent.
	 * @return The `vscode.ChatParticipant` API object.
	 */
	createDynamicChatAgent(extension: IExtensionDescription, id: string, dynamicProps: vscode.DynamicChatParticipantProps, handler: vscode.ChatExtendedRequestHandler): vscode.ChatParticipant {
		const handle = ExtHostChatAgents2._idPool++;
		const agent = new ExtHostChatAgent(extension, id, this._proxy, handle, handler);
		this._agents.set(handle, agent);

		this._proxy.$registerAgent(handle, extension.identifier, id, { isSticky: true } satisfies IExtensionChatAgentMetadata, dynamicProps); // Register with Main Process with metadata.
		return agent.apiAgent;
	}

	/**
	 * @brief Registers a chat participant detection provider.
	 * Functional Utility: Allows an extension to provide logic for detecting chat participants
	 * based on certain criteria.
	 * @param extension The extension description.
	 * @param provider The chat participant detection provider.
	 * @return A `Disposable` to unregister the provider.
	 */
	registerChatParticipantDetectionProvider(extension: IExtensionDescription, provider: vscode.ChatParticipantDetectionProvider): vscode.Disposable {
		const handle = ExtHostChatAgents2._participantDetectionProviderIdPool++; // Assign unique handle.
		this._participantDetectionProviders.set(handle, new ExtHostParticipantDetector(extension, provider)); // Store provider.
		this._proxy.$registerChatParticipantDetectionProvider(handle); // Register with Main Process.
		return toDisposable(() => { // Return disposable for unregistration.
			this._participantDetectionProviders.delete(handle);
			this._proxy.$unregisterChatParticipantDetectionProvider(handle);
		});
	}

	/**
	 * @brief Registers a related files provider.
	 * Functional Utility: Allows an extension to provide contextually related files
	 * for a chat session.
	 * @param extension The extension description.
	 * @param provider The chat related files provider.
	 * @param metadata Metadata for the provider.
	 * @return A `Disposable` to unregister the provider.
	 */
	registerRelatedFilesProvider(extension: IExtensionDescription, provider: vscode.ChatRelatedFilesProvider, metadata: vscode.ChatRelatedFilesProviderMetadata): vscode.Disposable {
		const handle = ExtHostChatAgents2._relatedFilesProviderIdPool++; // Assign unique handle.
		this._relatedFilesProviders.set(handle, new ExtHostRelatedFilesProvider(extension, provider)); // Store provider.
		this._proxy.$registerRelatedFilesProvider(handle, metadata); // Register with Main Process.
		return toDisposable(() => { // Return disposable for unregistration.
			this._relatedFilesProviders.delete(handle);
			this._proxy.$unregisterRelatedFilesProvider(handle);
		});
	}

	/**
	 * @brief IPC method to provide related files.
	 * @param handle The handle of the related files provider.
	 * @param request The chat request draft.
	 * @param token A cancellation token.
	 * @return An array of `IChatRelatedFile` DTOs, or `undefined`.
	 */
	async $provideRelatedFiles(handle: number, request: IChatRequestDraft, token: CancellationToken): Promise<Dto<IChatRelatedFile>[] | undefined> {
		const provider = this._relatedFilesProviders.get(handle);
		if (!provider) {
			return Promise.resolve([]);
		}

		const extRequestDraft = typeConvert.ChatRequestDraft.to(request); // Convert DTO to ExtHost type.
		return await provider.provider.provideRelatedFiles(extRequestDraft, token) ?? undefined; // Invoke provider and return.
	}

	/**
	 * @brief IPC method to detect chat participants.
	 * @param handle The handle of the participant detection provider.
	 * @param requestDto The chat agent request DTO.
	 * @param context Context for detection, including history.
	 * @param options Options for detection, including location.
	 * @param token A cancellation token.
	 * @return A `vscode.ChatParticipantDetectionResult` or `null`/`undefined`.
	 */
	async $detectChatParticipant(handle: number, requestDto: Dto<IChatAgentRequest>, context: { history: IChatAgentHistoryEntryDto[] }, options: { location: ChatAgentLocation; participants?: vscode.ChatParticipantMetadata[] }, token: CancellationToken): Promise<vscode.ChatParticipantDetectionResult | null | undefined> {
		const detector = this._participantDetectionProviders.get(handle);
		if (!detector) {
			return undefined;
		}

		const { request, location, history } = await this._createRequest(requestDto, context, detector.extension); // Prepare request objects.

		const model = await this.getModelForRequest(request, detector.extension); // Get language model for request.
		const extRequest = typeConvert.ChatAgentRequest.to(request, location, model, this.getDiagnosticsWhenEnabled(detector.extension)); // Convert to ExtHost request.

		return detector.provider.provideParticipantDetection( // Invoke provider.
			extRequest,
			{ history },
			{ participants: options.participants, location: typeConvert.ChatLocation.to(options.location) },
			token
		);
	}

	/**
	 * @brief Internal helper to create `vscode.ChatRequest` and related context objects from DTOs.
	 * Functional Utility: Converts raw DTOs received from the Main Process into API-friendly
	 * objects for extensions, including preparing chat history and location data.
	 * @param requestDto The chat agent request DTO.
	 * @param context Context for the request, including history.
	 * @param extension The extension description.
	 * @return An object containing the converted request, location, and history.
	 */
	private async _createRequest(requestDto: Dto<IChatAgentRequest>, context: { history: IChatAgentHistoryEntryDto[] }, extension: IExtensionDescription) {
		const request = revive<IChatAgentRequest>(requestDto); // Revive DTO into IChatAgentRequest.
		const convertedHistory = await this.prepareHistoryTurns(extension, request.agentId, context); // Prepare chat history.

		// in-place converting for location-data
		let location: vscode.ChatRequestEditorData | vscode.ChatRequestNotebookData | undefined;
		if (request.locationData?.type === ChatAgentLocation.Editor) {
			// Block Logic: Convert editor location data.
			const document = this._documents.getDocument(request.locationData.document);
			location = new extHostTypes.ChatRequestEditorData(document, typeConvert.Selection.to(request.locationData.selection), typeConvert.Range.to(request.locationData.wholeRange));

		} else if (request.locationData?.type === ChatAgentLocation.Notebook) {
			// Block Logic: Convert notebook location data.
			const cell = this._documents.getDocument(request.locationData.sessionInputUri);
			location = new extHostTypes.ChatRequestNotebookData(cell);

		} else if (request.locationData?.type === ChatAgentLocation.Terminal) {
			// TBD: Terminal location data conversion.
		}

		return { request, location, history: convertedHistory };
	}

	/**
	 * @brief Retrieves the appropriate language model for a given chat request.
	 * Functional Utility: Prioritizes a user-selected model; otherwise, falls back
	 * to the default language model. Throws an error if no model is available.
	 * @param request The chat agent request.
	 * @param extension The extension description.
	 * @return A `vscode.LanguageModelChat` object.
	 * @throws Error if no language model is available.
	 */
	private async getModelForRequest(request: IChatAgentRequest, extension: IExtensionDescription): Promise<vscode.LanguageModelChat> {
		let model: vscode.LanguageModelChat | undefined;
		if (request.userSelectedModelId) { // Block Logic: Try user-selected model first.
			model = await this._languageModels.getLanguageModelByIdentifier(extension, request.userSelectedModelId);
		}
		if (!model) { // Block Logic: Fallback to default model.
			model = await this._languageModels.getDefaultLanguageModel(extension);
			if (!model) {
				throw new Error('Language model unavailable');
			}
		}

		return model;
	}

	/**
	 * @brief IPC method to set the paused state of an in-flight chat request.
	 * @param handle The handle of the chat agent.
	 * @param requestId The ID of the request.
	 * @param isPaused The new paused state.
	 */
	async $setRequestPaused(handle: number, requestId: string, isPaused: boolean) {
		const agent = this._agents.get(handle);
		if (!agent) {
			return;
		}

		const inFlight = Iterable.find(this._inFlightRequests, r => r.requestId === requestId);
		if (!inFlight) {
			return;
		}

		agent.setChatRequestPauseState({ request: inFlight.extRequest, isPaused });
	}

	/**
	 * @brief IPC method to invoke a chat agent.
	 * Functional Utility: This is the core method for handling incoming chat requests from
	 * the Main Process. It prepares the request for the extension, creates a response stream,
	 * invokes the extension's chat agent handler, and manages the result and error reporting.
	 * @param handle The handle of the chat agent to invoke.
	 * @param requestDto The chat agent request DTO.
	 * @param context Context for invocation, including history.
	 * @param token A cancellation token.
	 * @return A promise resolving to `IChatAgentResult` or `undefined`.
	 */
	async $invokeAgent(handle: number, requestDto: Dto<IChatAgentRequest>, context: { history: IChatAgentHistoryEntryDto[] }, token: CancellationToken): Promise<IChatAgentResult | undefined> {
		const agent = this._agents.get(handle);
		if (!agent) {
			throw new Error(`[CHAT](${handle}) CANNOT invoke agent because the agent is not registered`);
		}

		let stream: ChatAgentResponseStream | undefined;
		let inFlightRequest: InFlightChatRequest | undefined;

		try {
			const { request, location, history } = await this._createRequest(requestDto, context, agent.extension); // Prepare request objects.

			// Init session disposables
			let sessionDisposables = this._sessionDisposables.get(request.sessionId);
			if (!sessionDisposables) {
				sessionDisposables = new DisposableStore();
				this._sessionDisposables.set(request.sessionId, sessionDisposables);
			}

			stream = new ChatAgentResponseStream(agent.extension, request, this._proxy, this._commands.converter, sessionDisposables); // Create response stream.

			const model = await this.getModelForRequest(request, agent.extension); // Get language model.
			const extRequest = typeConvert.ChatAgentRequest.to(request, location, model, this.getDiagnosticsWhenEnabled(agent.extension)); // Convert to ExtHost request.
			inFlightRequest = { requestId: requestDto.requestId, extRequest };
			this._inFlightRequests.add(inFlightRequest); // Add to in-flight requests.

			const task = agent.invoke( // Invoke the extension's chat agent handler.
				extRequest,
				{ history },
				stream.apiObject,
				token
			);

			// Functional Utility: Race the invocation promise against cancellation.
			return await raceCancellation(Promise.resolve(task).then((result) => {
				// Block Logic: Validate metadata and error details.
				if (result?.metadata) {
					try {
						JSON.stringify(result.metadata);
					} catch (err) {
						const msg = `result.metadata MUST be JSON.stringify-able. Got error: ${err.message}`;
						this._logService.error(`[${agent.extension.identifier.value}] [@${agent.id}] ${msg}`, agent.extension);
						return { errorDetails: { message: msg }, timings: stream?.timings, nextQuestion: result.nextQuestion };
					}
				}
				let errorDetails: IChatResponseErrorDetails | undefined;
				if (result?.errorDetails) {
					errorDetails = {
						...result.errorDetails,
						responseIsIncomplete: true
					};
				}
				if (errorDetails?.responseIsRedacted || errorDetails?.isQuotaExceeded) {
					checkProposedApiEnabled(agent.extension, 'chatParticipantPrivate');
				}

				return { errorDetails, timings: stream?.timings, metadata: result?.metadata, nextQuestion: result?.nextQuestion } satisfies IChatAgentResult;
			}), token);
		} catch (e) {
			this._logService.error(e, agent.extension); // Log errors.

			if (e instanceof extHostTypes.LanguageModelError && e.cause) {
				e = e.cause;
			}

			const isQuotaExceeded = e instanceof Error && e.name === 'ChatQuotaExceeded'; // Check for quota exceeded error.
			return { errorDetails: { message: toErrorMessage(e), responseIsIncomplete: true, isQuotaExceeded } };

		} finally {
			if (inFlightRequest) {
				this._inFlightRequests.delete(inFlightRequest); // Remove from in-flight requests.
			}
			stream?.close(); // Ensure stream is closed.
		}
	}

	/**
	 * @brief Retrieves diagnostics if the `chatReferenceDiagnostic` proposed API is enabled.
	 * @param extension The extension description.
	 * @return An array of diagnostics or an empty array.
	 */
	private getDiagnosticsWhenEnabled(extension: Readonly<IRelaxedExtensionDescription>) {
		if (!isProposedApiEnabled(extension, 'chatReferenceDiagnostic')) {
			return [];
		}
		return this._diagnostics.getDiagnostics();
	}

	/**
	 * @brief Prepares chat history turns for agent invocation.
	 * Functional Utility: Converts raw history DTOs into `vscode.ChatRequestTurn` and
	 * `vscode.ChatResponseTurn` objects, suitable for consumption by extensions.
	 * @param extension The extension description.
	 * @param agentId The ID of the chat agent.
	 * @param context Context including raw history DTOs.
	 * @return An array of `vscode.ChatRequestTurn` or `vscode.ChatResponseTurn`.
	 */
	private async prepareHistoryTurns(extension: Readonly<IRelaxedExtensionDescription>, agentId: string, context: { history: IChatAgentHistoryEntryDto[] }): Promise<(vscode.ChatRequestTurn | vscode.ChatResponseTurn)[]> {
		const res: (vscode.ChatRequestTurn | vscode.ChatResponseTurn)[] = [];

		for (const h of context.history) { // Block Logic: Iterate through each history entry.
			const ehResult = typeConvert.ChatAgentResult.to(h.result);
			const result: vscode.ChatResult = agentId === h.request.agentId ?
				ehResult :
				{ ...ehResult, metadata: undefined }; // Adjust result metadata if not the same agent.

			// REQUEST turn
			const varsWithoutTools = h.request.variables.variables
				.filter(v => !v.isTool)
				.map(v => typeConvert.ChatPromptReference.to(v, this.getDiagnosticsWhenEnabled(extension))); // Convert non-tool variables.
			const toolReferences = h.request.variables.variables
				.filter(v => v.isTool)
				.map(typeConvert.ChatLanguageModelToolReference.to); // Convert tool references.
			const turn = new extHostTypes.ChatRequestTurn(h.request.message, h.request.command, varsWithoutTools, h.request.agentId, toolReferences);
			res.push(turn); // Add request turn.

			// RESPONSE turn
			const parts = coalesce(h.response.map(r => typeConvert.ChatResponsePart.toContent(r, this._commands.converter))); // Convert response parts.
			res.push(new extHostTypes.ChatResponseTurn(parts, result, h.request.agentId, h.request.command)); // Add response turn.
		}

		return res;
	}

	/**
	 * @brief IPC method to release (dispose) resources associated with a chat session.
	 * @param sessionId The ID of the session to release.
	 */
	$releaseSession(sessionId: string): void {
		this._sessionDisposables.deleteAndDispose(sessionId); // Dispose session-specific disposables.
	}

	/**
	 * @brief IPC method to provide follow-up actions for a chat response.
	 * Functional Utility: Invokes the extension's follow-up provider, filters the results,
	 * and converts them into DTOs for the Main Process.
	 * @param requestDto The chat agent request DTO.
	 * @param handle The handle of the chat agent.
	 * @param result The result of the chat agent invocation.
	 * @param context Context including history.
	 * @param token A cancellation token.
	 * @return A promise resolving to an array of `IChatFollowup` DTOs.
	 */
	async $provideFollowups(requestDto: Dto<IChatAgentRequest>, handle: number, result: IChatAgentResult, context: { history: IChatAgentHistoryEntryDto[] }, token: CancellationToken): Promise<IChatFollowup[]> {
		const agent = this._agents.get(handle);
		if (!agent) {
			return Promise.resolve([]);
		}

		const request = revive<IChatAgentRequest>(requestDto);
		const convertedHistory = await this.prepareHistoryTurns(agent.extension, agent.id, context); // Prepare history.

		const ehResult = typeConvert.ChatAgentResult.to(result);
		return (await agent.provideFollowups(ehResult, { history: convertedHistory }, token)) // Invoke provider.
			.filter(f => { // Block Logic: Filter out invalid follow-ups.
				// The followup must refer to a participant that exists from the same extension
				const isValid = !f.participant || Iterable.some(
					this._agents.values(),
					a => a.id === f.participant && ExtensionIdentifier.equals(a.extension.identifier, agent.extension.identifier));
				if (!isValid) {
					this._logService.warn(`[@${agent.id}] ChatFollowup refers to an unknown participant: ${f.participant}`);
				}
				return isValid;
			})
			.map(f => typeConvert.ChatFollowup.from(f, request)); // Convert to DTOs.
	}

	/**
	 * @brief IPC method to accept feedback on a chat response.
	 * Functional Utility: Converts Main Process feedback DTOs into `vscode.ChatResultFeedback`
	 * and dispatches them to the relevant chat agent.
	 * @param handle The handle of the chat agent.
	 * @param result The result of the chat agent invocation.
	 * @param voteAction The vote action (up/down).
	 */
	$acceptFeedback(handle: number, result: IChatAgentResult, voteAction: IChatVoteAction): void {
		const agent = this._agents.get(handle);
		if (!agent) {
			return;
		}

		const ehResult = typeConvert.ChatAgentResult.to(result);
		let kind: extHostTypes.ChatResultFeedbackKind;
		switch (voteAction.direction) { // Block Logic: Map vote direction to feedback kind.
			case ChatAgentVoteDirection.Down:
				kind = extHostTypes.ChatResultFeedbackKind.Unhelpful;
				break;
			case ChatAgentVoteDirection.Up:
				kind = extHostTypes.ChatResultFeedbackKind.Helpful;
				break;
		}

		const feedback: vscode.ChatResultFeedback = { // Construct feedback object.
			result: ehResult,
			kind,
			unhelpfulReason: isProposedApiEnabled(agent.extension, 'chatParticipantAdditions') ? voteAction.reason : undefined,
		};
		agent.acceptFeedback(Object.freeze(feedback)); // Dispatch feedback.
	}

	/**
	 * @brief IPC method to accept user actions on a chat response.
	 * Functional Utility: Converts Main Process user action DTOs into `vscode.ChatUserActionEvent`
	 * and dispatches them to the relevant chat agent. Filters out 'vote' actions as they are
	 * handled by `$acceptFeedback`.
	 * @param handle The handle of the chat agent.
	 * @param result The result of the chat agent invocation.
	 * @param event The user action event.
	 */
	$acceptAction(handle: number, result: IChatAgentResult, event: IChatUserActionEvent): void {
		const agent = this._agents.get(handle);
		if (!agent) {
			return;
		}
		if (event.action.kind === 'vote') {
			// handled by $acceptFeedback
			return;
		}

		const ehAction = typeConvert.ChatAgentUserActionEvent.to(result, event, this._commands.converter); // Convert to ExtHost action.
		if (ehAction) {
			agent.acceptAction(Object.freeze(ehAction)); // Dispatch action.
		}
	}

	/**
	 * @brief IPC method to invoke a completion provider for chat variables.
	 * Functional Utility: Dispatches a request to an agent's variable completion provider,
	 * managing disposables for the completion session.
	 * @param handle The handle of the chat agent.
	 * @param query The query string for completion.
	 * @param token A cancellation token.
	 * @return A promise resolving to an array of `IChatAgentCompletionItem` DTOs.
	 */
	async $invokeCompletionProvider(handle: number, query: string, token: CancellationToken): Promise<IChatAgentCompletionItem[]> {
		const agent = this._agents.get(handle);
		if (!agent) {
			return [];
		}

		let disposables = this._completionDisposables.get(handle);
		if (disposables) {
			// Clear any disposables from the last invocation of this completion provider
			disposables.clear();
		} else {
			disposables = new DisposableStore();
			this._completionDisposables.set(handle, disposables);
		}

		const items = await agent.invokeCompletionProvider(query, token); // Invoke completion provider.

		return items.map((i) => typeConvert.ChatAgentCompletionItem.from(i, this._commands.converter, disposables)); // Convert to DTOs.
	}

	/**
	 * @brief IPC method to provide a welcome message for a chat agent.
	 * @param handle The handle of the chat agent.
	 * @param token A cancellation token.
	 * @return A promise resolving to `IChatWelcomeMessageContent` DTO or `undefined`.
	 */
	async $provideWelcomeMessage(handle: number, token: CancellationToken): Promise<IChatWelcomeMessageContent | undefined> {
		const agent = this._agents.get(handle);
		if (!agent) {
			return;
		}

		const content = await agent.provideWelcomeMessage(token); // Invoke provider.
		const icon = content?.icon; // typescript
		if (!content || !ThemeIcon.isThemeIcon(icon)) { // Block Logic: Validate welcome message content.
			return undefined;
		}

		return { // Convert to DTO.
			...content,
			icon,
			message: typeConvert.MarkdownString.from(content.message),
		};
	}

	/**
	 * @brief IPC method to provide a chat title.
	 * @param handle The handle of the chat agent.
	 * @param context History entries for title generation.
	 * @param token A cancellation token.
	 * @return A promise resolving to the chat title string or `undefined`.
	 */
	async $provideChatTitle(handle: number, context: IChatAgentHistoryEntryDto[], token: CancellationToken): Promise<string | undefined> {
		const agent = this._agents.get(handle);
		if (!agent) {
			return;
		}

		const history = await this.prepareHistoryTurns(agent.extension, agent.id, { history: context }); // Prepare history.
		return await agent.provideTitle({ history }, token); // Invoke title provider.
	}

	/**
	 * @brief IPC method to provide sample questions for a chat agent.
	 * @param handle The handle of the chat agent.
	 * @param location The chat agent location.
	 * @param token A cancellation token.
	 * @return A promise resolving to an array of `IChatFollowup` DTOs or `undefined`.
	 */
	async $provideSampleQuestions(handle: number, location: ChatAgentLocation, token: CancellationToken): Promise<IChatFollowup[] | undefined> {
		const agent = this._agents.get(handle);
		if (!agent) {
			return;
		}

		return (await agent.provideSampleQuestions(typeConvert.ChatLocation.to(location), token)) // Invoke provider.
			.map(f => typeConvert.ChatFollowup.from(f, undefined)); // Convert to DTOs.
	}
}

/**
 * @class ExtHostParticipantDetector
 * @brief Internal wrapper for an extension's `vscode.ChatParticipantDetectionProvider`.
 * Functional Utility: Stores the extension and the provider, used internally by
 * `ExtHostChatAgents2` to manage participant detection.
 */
class ExtHostParticipantDetector {
	/**
	 * @brief Constructs a new `ExtHostParticipantDetector`.
	 * @param extension The extension description.
	 * @param provider The `vscode.ChatParticipantDetectionProvider` instance.
	 */
	constructor(
		public readonly extension: IExtensionDescription,
		public readonly provider: vscode.ChatParticipantDetectionProvider,
	) { }
}

/**
 * @class ExtHostRelatedFilesProvider
 * @brief Internal wrapper for an extension's `vscode.ChatRelatedFilesProvider`.
 * Functional Utility: Stores the extension and the provider, used internally by
 * `ExtHostChatAgents2` to manage related files provision.
 */
class ExtHostRelatedFilesProvider {
	/**
	 * @brief Constructs a new `ExtHostRelatedFilesProvider`.
	 * @param extension The extension description.
	 * @param provider The `vscode.ChatRelatedFilesProvider` instance.
	 */
	constructor(
		public readonly extension: IExtensionDescription,
		public readonly provider: vscode.ChatRelatedFilesProvider,
	) { }
}

/**
 * @class ExtHostChatAgent
 * @brief Internal wrapper for an extension's `vscode.ChatParticipant` implementation.
 * Functional Utility: Adapts the extension's chat participant to the ExtHost's
 * internal management. It handles communication with the Main Process for metadata
 * updates, dispatches feedback and user actions, and provides the API object
 * (`apiAgent`) for the extension to interact with.
 */
class ExtHostChatAgent {

	private _followupProvider: vscode.ChatFollowupProvider | undefined; // Provider for follow-up actions.
	private _iconPath: vscode.Uri | { light: vscode.Uri; dark: vscode.Uri } | vscode.ThemeIcon | undefined; // Icon path for the agent.
	private _helpTextPrefix: string | vscode.MarkdownString | undefined; // Help text prefix.
	private _helpTextVariablesPrefix: string | vscode.MarkdownString | undefined; // Help text variables prefix.
	private _helpTextPostfix: string | vscode.MarkdownString | undefined; // Help text postfix.
	private _isSecondary: boolean | undefined; // Flag indicating if the agent is secondary.
	private _onDidReceiveFeedback = new Emitter<vscode.ChatResultFeedback>(); // Event emitter for feedback reception.
	private _onDidPerformAction = new Emitter<vscode.ChatUserActionEvent>(); // Event emitter for user actions.
	private _supportIssueReporting: boolean | undefined; // Flag for issue reporting support.
	private _agentVariableProvider?: { provider: vscode.ChatParticipantCompletionItemProvider; triggerCharacters: string[] }; // Provider for chat variables completion.
	private _welcomeMessageProvider?: vscode.ChatWelcomeMessageProvider | undefined; // Provider for welcome messages.
	private _titleProvider?: vscode.ChatTitleProvider | undefined; // Provider for chat titles.
	private _requester: vscode.ChatRequesterInformation | undefined; // Information about the requester.
	private _pauseStateEmitter = new Emitter<vscode.ChatParticipantPauseStateEvent>(); // Event emitter for pause state changes.

	/**
	 * @brief Constructs a new `ExtHostChatAgent`.
	 * @param extension The extension description.
	 * @param id The unique ID of the chat agent.
	 * @param _proxy The proxy to the MainThreadChatAgents2.
	 * @param _handle The internal handle of the agent.
	 * @param _requestHandler The extension's `vscode.ChatExtendedRequestHandler`.
	 */
	constructor(
		public readonly extension: IExtensionDescription,
		public readonly id: string,
		private readonly _proxy: MainThreadChatAgentsShape2,
		private readonly _handle: number,
		private _requestHandler: vscode.ChatExtendedRequestHandler,
	) { }

	/**
	 * @brief Accepts feedback on a chat response and fires an event.
	 * @param feedback The feedback received.
	 */
	acceptFeedback(feedback: vscode.ChatResultFeedback) {
		this._onDidReceiveFeedback.fire(feedback);
	}

	/**
	 * @brief Accepts a user action and fires an event.
	 * @param event The user action event.
	 */
	acceptAction(event: vscode.ChatUserActionEvent) {
		this._onDidPerformAction.fire(event);
	}

	/**
	 * @brief Sets the pause state of a chat request and fires an event.
	 * @param pauseState The pause state event.
	 */
	setChatRequestPauseState(pauseState: vscode.ChatParticipantPauseStateEvent) {
		this._pauseStateEmitter.fire(pauseState);
	}

	/**
	 * @brief Invokes the completion provider for chat variables.
	 * @param query The query string.
	 * @param token A cancellation token.
	 * @return A promise resolving to an array of `vscode.ChatCompletionItem`.
	 */
	async invokeCompletionProvider(query: string, token: CancellationToken): Promise<vscode.ChatCompletionItem[]> {
		if (!this._agentVariableProvider) {
			return [];
		}

		return await this._agentVariableProvider.provider.provideCompletionItems(query, token) ?? [];
	}

	/**
	 * @brief Invokes the follow-up provider.
	 * @param result The chat result.
	 * @param context The chat context.
	 * @param token A cancellation token.
	 * @return A promise resolving to an array of `vscode.ChatFollowup`.
	 */
	async provideFollowups(result: vscode.ChatResult, context: vscode.ChatContext, token: CancellationToken): Promise<vscode.ChatFollowup[]> {
		if (!this._followupProvider) {
			return [];
		}

		const followups = await this._followupProvider.provideFollowups(result, context, token);
		if (!followups) {
			return [];
		}
		return followups
			// Block Logic: Filter out deprecated or invalid follow-up types.
			.filter(f => !(f && 'commandId' in f))
			// Filter out followups from older providers before 'message' changed to 'prompt'
			.filter(f => !(f && 'message' in f));
	}

	/**
	 * @brief Invokes the welcome message provider.
	 * @param token A cancellation token.
	 * @return A promise resolving to `IChatWelcomeMessageContent` DTO or `undefined`.
	 */
	async provideWelcomeMessage(token: CancellationToken): Promise<IChatWelcomeMessageContent | undefined> {
		if (!this._welcomeMessageProvider?.provideWelcomeMessage) {
			return undefined;
		}

		const content = await this._welcomeMessageProvider.provideWelcomeMessage(token);
		const icon = content?.icon; // typescript
		if (!content || !ThemeIcon.isThemeIcon(icon)) { // Block Logic: Validate welcome message content.
			return undefined;
		}

		return { // Convert to DTO.
			...content,
			icon,
			message: typeConvert.MarkdownString.from(content.message),
		};
	}

	/**
	 * @brief Invokes the chat title provider.
	 * @param context The chat context.
	 * @param token A cancellation token.
	 * @return A promise resolving to the chat title string or `undefined`.
	 */
	async provideTitle(context: vscode.ChatContext, token: CancellationToken): Promise<string | undefined> {
		if (!this._titleProvider) {
			return;
		}

		return await this._titleProvider.provideChatTitle(context, token) ?? undefined;
	}

	/**
	 * @brief Invokes the sample questions provider.
	 * @param location The chat location.
	 * @param token A cancellation token.
	 * @return A promise resolving to an array of `vscode.ChatFollowup`.
	 */
	async provideSampleQuestions(location: vscode.ChatLocation, token: CancellationToken): Promise<vscode.ChatFollowup[]> {
		if (!this._welcomeMessageProvider || !this._welcomeMessageProvider.provideSampleQuestions) {
			return [];
		}
		const content = await this._welcomeMessageProvider.provideSampleQuestions(location, token);
		if (!content) {
			return [];
		}

		return content;
	}

	/**
	 * @brief Gets the `vscode.ChatParticipant` API object exposed to extensions.
	 * Functional Utility: This getter constructs an API object that allows extensions
	 * to interact with their registered chat participant. It includes properties
	 * that map directly to the `ExtHostChatAgent`'s internal state and
	 * triggers metadata updates to the Main Process when properties are modified.
	 */
	get apiAgent(): vscode.ChatParticipant {
		let disposed = false;
		let updateScheduled = false;
		/**
		 * @brief Schedules an update of the agent's metadata to the Main Process.
		 * Functional Utility: Debounces metadata updates to avoid excessive IPC calls,
		 * ensuring that changes to agent properties are efficiently propagated.
		 */
		const updateMetadataSoon = () => {
			if (disposed) {
				return;
			}
			if (updateScheduled) {
				return;
			}
			updateScheduled = true;
			queueMicrotask(() => { // Block Logic: Queue microtask to send update.
				this._proxy.$updateAgent(this._handle, {
					icon: !this._iconPath ? undefined :
						this._iconPath instanceof URI ? this._iconPath :
							'light' in this._iconPath ? this._iconPath.light :
								undefined,
					iconDark: !this._iconPath ? undefined :
						'dark' in this._iconPath ? this._iconPath.dark :
							undefined,
					themeIcon: this._iconPath instanceof extHostTypes.ThemeIcon ? this._iconPath : undefined,
					hasFollowups: this._followupProvider !== undefined,
					isSecondary: this._isSecondary,
					helpTextPrefix: (!this._helpTextPrefix || typeof this._helpTextPrefix === 'string') ? this._helpTextPrefix : typeConvert.MarkdownString.from(this._helpTextPrefix),
					helpTextVariablesPrefix: (!this._helpTextVariablesPrefix || typeof this._helpTextVariablesPrefix === 'string') ? this._helpTextVariablesPrefix : typeConvert.MarkdownString.from(this._helpTextVariablesPrefix),
					helpTextPostfix: (!this._helpTextPostfix || typeof this._helpTextPostfix === 'string') ? this._helpTextPostfix : typeConvert.MarkdownString.from(this._helpTextPostfix),
					supportIssueReporting: this._supportIssueReporting,
					requester: this._requester,
				});
				updateScheduled = false;
			});
		};

		const that = this; // Capture `this` for use in property getters/setters.
		return {
			get id() {
				return that.id;
			},
			get iconPath() {
				return that._iconPath;
			},
			set iconPath(v) {
				that._iconPath = v;
				updateMetadataSoon();
			},
			get requestHandler() {
				return that._requestHandler;
			},
			set requestHandler(v) {
				assertType(typeof v === 'function', 'Invalid request handler');
				that._requestHandler = v;
			},
			get followupProvider() {
				return that._followupProvider;
			},
			set followupProvider(v) {
				that._followupProvider = v;
				updateMetadataSoon();
			},
			get helpTextPrefix() {
				checkProposedApiEnabled(that.extension, 'defaultChatParticipant');
				return that._helpTextPrefix;
			},
			set helpTextPrefix(v) {
				checkProposedApiEnabled(that.extension, 'defaultChatParticipant');
				that._helpTextPrefix = v;
				updateMetadataSoon();
			},
			get helpTextVariablesPrefix() {
				checkProposedApiEnabled(that.extension, 'defaultChatParticipant');
				return that._helpTextVariablesPrefix;
			},
			set helpTextVariablesPrefix(v) {
				checkProposedApiEnabled(that.extension, 'defaultChatParticipant');
				that._helpTextVariablesPrefix = v;
				updateMetadataSoon();
			},
			get helpTextPostfix() {
				checkProposedApiEnabled(that.extension, 'defaultChatParticipant');
				return that._helpTextPostfix;
			},
			set helpTextPostfix(v) {
				checkProposedApiEnabled(that.extension, 'defaultChatParticipant');
				that._helpTextPostfix = v;
				updateMetadataSoon();
			},
			get isSecondary() {
				checkProposedApiEnabled(that.extension, 'defaultChatParticipant');
				return that._isSecondary;
			},
			set isSecondary(v) {
				checkProposedApiEnabled(that.extension, 'defaultChatParticipant');
				that._isSecondary = v;
				updateMetadataSoon();
			},
			get supportIssueReporting() {
				checkProposedApiEnabled(that.extension, 'chatParticipantPrivate');
				return that._supportIssueReporting;
			},
			set supportIssueReporting(v) {
				checkProposedApiEnabled(that.extension, 'chatParticipantPrivate');
				that._supportIssueReporting = v;
				updateMetadataSoon();
			},
			get onDidReceiveFeedback() {
				return that._onDidReceiveFeedback.event;
			},
			set participantVariableProvider(v) {
				checkProposedApiEnabled(that.extension, 'chatParticipantAdditions');
				that._agentVariableProvider = v;
				if (v) { // Block Logic: If a variable provider is set.
					if (!v.triggerCharacters.length) {
						throw new Error('triggerCharacters are required');
					}

					that._proxy.$registerAgentCompletionsProvider(that._handle, that.id, v.triggerCharacters); // Register completions provider.
				} else { // Block Logic: If variable provider is unset.
					that._proxy.$unregisterAgentCompletionsProvider(that._handle, that.id); // Unregister completions provider.
				}
			},
			get participantVariableProvider() {
				checkProposedApiEnabled(that.extension, 'chatParticipantAdditions');
				return that._agentVariableProvider;
			},
			set welcomeMessageProvider(v) {
				checkProposedApiEnabled(that.extension, 'defaultChatParticipant');
				that._welcomeMessageProvider = v;
				updateMetadataSoon();
			},
			get welcomeMessageProvider() {
				checkProposedApiEnabled(that.extension, 'defaultChatParticipant');
				return that._welcomeMessageProvider;
			},
			set titleProvider(v) {
				checkProposedApiEnabled(that.extension, 'defaultChatParticipant');
				that._titleProvider = v;
				updateMetadataSoon();
			},
			get titleProvider() {
				checkProposedApiEnabled(that.extension, 'defaultChatParticipant');
				return that._titleProvider;
			},
			get onDidChangePauseState() {
				checkProposedApiEnabled(that.extension, 'chatParticipantAdditions');
				return that._pauseStateEmitter.event;
			},
			onDidPerformAction: !isProposedApiEnabled(this.extension, 'chatParticipantAdditions')
				? undefined! // If proposed API not enabled, return undefined.
				: this._onDidPerformAction.event,
			set requester(v) {
				that._requester = v;
				updateMetadataSoon();
			},
			get requester() {
				return that._requester;
			},
			dispose() { // Functional Utility: Dispose method for the API object.
				disposed = true;
				that._followupProvider = undefined; // Clear reference.
				that._onDidReceiveFeedback.dispose(); // Dispose event emitter.
				that._proxy.$unregisterAgent(that._handle); // Unregister agent from Main Process.
			},
		} satisfies vscode.ChatParticipant;
	}

	/**
	 * @brief Invokes the chat agent's request handler.
	 * @param request The chat request.
	 * @param context The chat context.
	 * @param response The chat response stream.
	 * @param token A cancellation token.
	 * @return A promise resolving to `vscode.ChatResult` or `void`.
	 */
	invoke(request: vscode.ChatRequest, context: vscode.ChatContext, response: vscode.ChatResponseStream, token: CancellationToken): vscode.ProviderResult<vscode.ChatResult | void> {
		return this._requestHandler(request, context, response, token); // Call the extension's provided handler.
	}
}