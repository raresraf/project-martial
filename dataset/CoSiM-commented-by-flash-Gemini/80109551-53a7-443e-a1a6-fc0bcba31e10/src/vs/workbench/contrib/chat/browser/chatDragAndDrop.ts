/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

/**
 * @file chatDragAndDrop.ts
 * @brief Implements drag-and-drop functionality for attaching content to a chat input.
 *
 * This module provides the `ChatDragAndDrop` class which manages the UI and logic
 * for drag-and-drop operations within a chat interface, likely in an IDE context
 * such as VS Code. It supports dragging and dropping various types of content,
 * including files (internal/external), folders, images, document symbols, and markers (problems),
 * converting them into structured {@link IChatRequestVariableEntry} objects for the chat.
 *
 * Architectural Intent: To provide a seamless and intuitive user experience for
 * incorporating external content directly into chat requests, thereby enriching
 * the context of AI interactions. It integrates with various platform services
 * (`IFileService`, `IEditorService`, `ITextModelService`, `ISharedWebContentExtractorService`)
 * to resolve and process dropped data.
 *
 * Error Handling: Includes mechanisms for handling unsupported drag types,
 * oversized image files, and logging of fetch errors.
 *
 * Algorithm:
 * 1. Event Listening: Registers `DragAndDropObserver` for `dragover`, `dragleave`, and `drop` events.
 * 2. Type Guessing: Employs `guessDropType` to heuristically determine the content type based on `DataTransfer` items.
 * 3. Data Extraction: Uses platform-specific data transfer utilities (`extractEditorsDropData`, `extractMarkerDropData`, `extractSymbolDropData`, `UriList.parse`).
 * 4. Content Resolution: Asynchronously resolves dropped items into `IChatRequestVariableEntry` objects.
 *    - Files/Folders: Resolved via `IFileService` and `ITextModelService`.
 *    - Images: Handled from file drops or URLs, including resizing and hash generation.
 *    - Symbols/Markers: Converted directly into structured variable entries.
 * 5. UI Feedback: Manages visual overlays and text to guide the user during drag operations.
 *
 * Time Complexity:
 * - DND event handling and type guessing: O(1) in most cases, dependent on the number of `DataTransfer` items.
 * - File/Resource Resolution: Depends on file system operations (I/O), typically O(1) for metadata, O(size of file) for content.
 * - Image Resizing: O(width * height) for image pixel processing.
 * - Symbol/Marker Resolution: O(number of symbols/markers) if processing a list.
 * Space Complexity:
 * - O(1) for DND state and overlays.
 * - O(size of file/image) for buffering dropped content.
 */

import { DataTransfers } from '../../../../base/browser/dnd.js';
import { $, DragAndDropObserver } from '../../../../base/browser/dom.js';
import { renderLabelWithIcons } from '../../../../base/browser/ui/iconLabel/iconLabels.js';
import { coalesce } from '../../../../base/common/arrays.js';
import { CancellationToken } from '../../../../base/common/cancellation.js';
import { Codicon } from '../../../../base/common/codicons.js';
import { UriList } from '../../../../base/common/dataTransfer.js';
import { IDisposable } from '../../../../base/common/lifecycle.js';
import { Mimes } from '../../../../base/common/mime.js';
import { basename } from '../../../../base/common/resources.js';
import { URI } from '../../../../base/common/uri.js';
import { IRange } from '../../../../editor/common/core/range.js';
import { SymbolKinds } from '../../../../editor/common/languages.js';
import { ITextModelService } from '../../../../editor/common/services/resolverService.js';
import { localize } from '../../../../nls.js';
import { IDialogService } from '../../../../platform/dialogs/common/dialogs.js';
import { CodeDataTransfers, containsDragType, DocumentSymbolTransferData, extractEditorsDropData, extractMarkerDropData, extractSymbolDropData, IDraggedResourceEditorInput, MarkerTransferData } from '../../../../platform/dnd/browser/dnd.js';
import { IFileService } from '../../../../platform/files/common/files.js';
import { ILogService } from '../../../../platform/log/common/log.js';
import { MarkerSeverity } from '../../../../platform/markers/common/markers.js';
import { IThemeService, Themable } from '../../../../platform/theme/common/themeService.js';
import { ISharedWebContentExtractorService } from '../../../../platform/webContentExtractor/common/webContentExtractor.js';
import { isUntitledResourceEditorInput } from '../../../common/editor.js';
import { EditorInput } from '../../../common/editor/editorInput.js';
import { IEditorService } from '../../../services/editor/common/editorService.js';
import { IExtensionService, isProposedApiEnabled } from '../../../services/extensions/common/extensions.js';
import { UntitledTextEditorInput } from '../../../services/untitled/common/untitledTextEditorInput.js';
import { IChatRequestVariableEntry, IDiagnosticVariableEntry, IDiagnosticVariableEntryFilterData, ISymbolVariableEntry } from '../common/chatModel.js';
import { IChatWidgetService } from './chat.js';
import { ChatAttachmentModel } from './chatAttachmentModel.js';
import { IChatInputStyles } from './chatInputPart.js';
import { imageToHash } from './chatPasteProviders.js';
import { resizeImage } from './imageUtils.js';

/**
 * @enum ChatDragAndDropType
 * @brief Defines the categorized types of data that can be handled by the chat drag-and-drop feature.
 *
 * This enumeration helps to differentiate between various forms of content
 * that a user might drag into the chat input, enabling specific processing
 * and feedback for each type.
 */
enum ChatDragAndDropType {
	FILE_INTERNAL,  // An internal file URI dragged from within the application.
	FILE_EXTERNAL,  // An external file dropped from the operating system.
	FOLDER,         // A directory dropped.
	IMAGE,          // Image data (from file or URL).
	SYMBOL,         // A programming symbol (e.g., function, variable) from the editor.
	HTML,           // HTML content, potentially containing URLs or embedded data.
	MARKER,         // A diagnostic marker or problem.
}

/**
 * @class ChatDragAndDrop
 * @brief Manages drag-and-drop operations for the chat input, converting dropped content into chat attachments.
 *
 * This class extends `Themable` to allow for theme-aware styling of its drag-and-drop overlays.
 * It observes drag events on specified target elements, provides visual feedback, and processes
 * the dropped data to add it as context to the chat.
 */
export class ChatDragAndDrop extends Themable {

	/**
	 * @brief Stores a map of target HTML elements to their associated DND overlay and disposable listeners.
	 */
	private readonly overlays: Map<HTMLElement, { overlay: HTMLElement; disposable: IDisposable }> = new Map();
	/**
	 * @brief Reference to the HTML element displaying the overlay text feedback.
	 */
	private overlayText?: HTMLElement;
	/**
	 * @brief The background color for the overlay text, derived from theme styles.
	 */
	private overlayTextBackground: string = '';

	/**
	 * @brief Constructs a new ChatDragAndDrop instance.
	 * @param attachmentModel The model responsible for managing chat attachments.
	 * @param styles Styling information for the chat input.
	 * @param themeService The service for theme-related operations.
	 * @param extensionService The service for managing extensions and proposed APIs.
	 * @param fileService The service for file system operations.
	 * @param editorService The service for managing editor instances.
	 * @param dialogService The service for displaying dialogs (e.g., error messages).
	 * @param textModelService The service for creating and resolving text models.
	 * @param webContentExtractorService The service for extracting web content, particularly images.
	 * @param chatWidgetService The service for interacting with the chat widget.
	 * @param logService The service for logging information and errors.
	 */
	constructor(
		private readonly attachmentModel: ChatAttachmentModel,
		private readonly styles: IChatInputStyles,
		@IThemeService themeService: IThemeService,
		@IExtensionService private readonly extensionService: IExtensionService,
		@IFileService private readonly fileService: IFileService,
		@IEditorService private readonly editorService: IEditorService,
		@IDialogService private readonly dialogService: IDialogService,
		@ITextModelService private readonly textModelService: ITextModelService,
		@ISharedWebContentExtractorService private readonly webContentExtractorService: ISharedWebContentExtractorService,
		@IChatWidgetService private readonly chatWidgetService: IChatWidgetService,
		@ILogService private readonly logService: ILogService,
	) {
		super(themeService);

		// Functional Utility: Applies initial styles to the DND overlays.
		this.updateStyles();
	}

	/**
	 * @brief Adds a drag-and-drop overlay to a specified target HTML element.
	 * @param target The HTML element to which the overlay should be added.
	 * @param overlayContainer The container element where the overlay will be appended.
	 *
	 * Logic: If an existing overlay is present for the target, it is removed first.
	 * Then, a new overlay is created and associated with the target.
	 */
	addOverlay(target: HTMLElement, overlayContainer: HTMLElement): void {
		this.removeOverlay(target);

		const { overlay, disposable } = this.createOverlay(target, overlayContainer);
		this.overlays.set(target, { overlay, disposable });
	}

	/**
	 * @brief Removes the drag-and-drop overlay from a specified target HTML element.
	 * @param target The HTML element from which the overlay should be removed.
	 */
	removeOverlay(target: HTMLElement): void {
		// If the target is currently active, clear the active target reference.
		if (this.currentActiveTarget === target) {
			this.currentActiveTarget = undefined;
		}

		// Block Logic: Finds and disposes of the existing overlay and its listeners.
		const existingOverlay = this.overlays.get(target);
		if (existingOverlay) {
			existingOverlay.overlay.remove();
			existingOverlay.disposable.dispose();
			this.overlays.delete(target);
		}
	}

	/**
	 * @brief Tracks the currently active drag-and-drop target element.
	 */
	private currentActiveTarget: HTMLElement | undefined = undefined;

	/**
	 * @brief Creates a new drag-and-drop overlay and sets up event listeners.
	 * @param target The HTML element to observe for drag-and-drop events.
	 * @param overlayContainer The parent element for the overlay.
	 * @returns An object containing the created overlay element and its disposable listeners.
	 *
	 * Logic: Creates an overlay DOM element, applies styling, and registers
	 * `DragAndDropObserver` to handle `dragover`, `dragleave`, and `drop` events.
	 */
	private createOverlay(target: HTMLElement, overlayContainer: HTMLElement): { overlay: HTMLElement; disposable: IDisposable } {
		const overlay = document.createElement('div');
		overlay.classList.add('chat-dnd-overlay');
		// Functional Utility: Applies styling to the overlay based on current theme.
		this.updateOverlayStyles(overlay);
		overlayContainer.appendChild(overlay);

		// Block Logic: Creates a DragAndDropObserver to manage DND events for the target.
		const disposable = new DragAndDropObserver(target, {
			onDragOver: (e) => {
				// Pre-condition: Event is a drag over event.
				// Functional Utility: Prevents default browser drag-and-drop behavior and stops event propagation.
				e.stopPropagation();
				e.preventDefault();

				// If the target is already active, no need to update.
				if (target === this.currentActiveTarget) {
					return;
				}

				// If another target was active, reset its overlay.
				if (this.currentActiveTarget) {
					this.setOverlay(this.currentActiveTarget, undefined);
				}

				// Mark the current target as active.
				this.currentActiveTarget = target;

				// Functional Utility: Updates the visual feedback for the drag over event.
				this.onDragEnter(e, target);

			},
			onDragLeave: (e) => {
				// If the leaving target was the active one, clear it.
				if (target === this.currentActiveTarget) {
					this.currentActiveTarget = undefined;
				}

				// Functional Utility: Clears the visual feedback for the drag leave event.
				this.onDragLeave(e, target);
			},
			onDrop: (e) => {
				// Functional Utility: Prevents default browser drag-and-drop behavior and stops event propagation.
				e.stopPropagation();
				e.preventDefault();

				// If the drop is not on the currently active target, ignore.
				if (target !== this.currentActiveTarget) {
					return;
				}

				// Clear the active target and process the drop.
				this.currentActiveTarget = undefined;
				// Functional Utility: Clears visual feedback and processes the dropped data.
				this.onDrop(e, target);
			},
		});

		return { overlay, disposable };
	}

	/**
	 * @brief Handles the `dragenter` event, providing visual feedback for a potential drop.
	 * @param e The DragEvent.
	 * @param target The HTML element currently being dragged over.
	 */
	private onDragEnter(e: DragEvent, target: HTMLElement): void {
		// Logic: Estimates the type of data being dragged and updates the overlay with appropriate feedback.
		const estimatedDropType = this.guessDropType(e);
		this.updateDropFeedback(e, target, estimatedDropType);
	}

	/**
	 * @brief Handles the `dragleave` event, clearing visual feedback.
	 * @param e The DragEvent.
	 * @param target The HTML element that the drag operation is leaving.
	 */
	private onDragLeave(e: DragEvent, target: HTMLElement): void {
		// Logic: Clears the visual feedback as the drag operation is no longer over the target.
		this.updateDropFeedback(e, target, undefined);
	}

	/**
	 * @brief Handles the `drop` event, processing the dropped data.
	 * @param e The DragEvent.
	 * @param target The HTML element where the data was dropped.
	 */
	private onDrop(e: DragEvent, target: HTMLElement): void {
		// Logic: Clears visual feedback and then initiates the actual processing of the dropped content.
		this.updateDropFeedback(e, target, undefined);
		this.drop(e);
	}

	/**
	 * @brief Asynchronously processes the dropped data and adds it to the chat attachment model.
	 * @param e The DragEvent containing the dropped data.
	 *
	 * Logic: Extracts potential chat contexts from the drag event. If contexts are found,
	 * they are added to the `ChatAttachmentModel`.
	 */
	private async drop(e: DragEvent): Promise<void> {
		const contexts = await this.getAttachContext(e);
		// Block Logic: If any attachment contexts were resolved, add them to the model.
		if (contexts.length === 0) {
			return;
		}

		this.attachmentModel.addContext(...contexts);
	}

	/**
	 * @brief Updates the visual feedback (overlay) displayed during a drag operation.
	 * @param e The DragEvent.
	 * @param target The HTML element currently being dragged over.
	 * @param dropType The estimated type of content being dropped, or undefined to hide feedback.
	 *
	 * Functional Utility: Sets the `dropEffect` on the `dataTransfer` object and updates the overlay's visibility and text.
	 */
	private updateDropFeedback(e: DragEvent, target: HTMLElement, dropType: ChatDragAndDropType | undefined): void {
		const showOverlay = dropType !== undefined;
		// If there's a dataTransfer object, set the drop effect.
		if (e.dataTransfer) {
			e.dataTransfer.dropEffect = showOverlay ? 'copy' : 'none';
		}

		// Update the overlay's appearance based on whether a valid drop type is detected.
		this.setOverlay(target, dropType);
	}

	/**
	 * @brief Estimates the type of data being dragged based on the `DragEvent`'s `dataTransfer` types.
	 * @param e The DragEvent.
	 * @returns A {@link ChatDragAndDropType} enum value or undefined if the type is not recognized/supported.
	 *
	 * Algorithm: Checks the `dataTransfer` for specific MIME types and custom data transfers
	 * to infer the most appropriate `ChatDragAndDropType`. Includes checks for proposed APIs
	 * for binary data.
	 */
	private guessDropType(e: DragEvent): ChatDragAndDropType | undefined {
		// This is an estimation based on the datatransfer types/items
		// Block Logic: Checks if the event indicates an image drag and if the API supports binary data.
		if (this.isImageDnd(e)) {
			return this.extensionService.extensions.some(ext => isProposedApiEnabled(ext, 'chatReferenceBinaryData')) ? ChatDragAndDropType.IMAGE : undefined;
		}
		// Block Logic: Checks for HTML content.
		else if (containsDragType(e, 'text/html')) {
			return ChatDragAndDropType.HTML;
		}
		// Block Logic: Checks for document symbol data.
		else if (containsDragType(e, CodeDataTransfers.SYMBOLS)) {
			return ChatDragAndDropType.SYMBOL;
		}
		// Block Logic: Checks for diagnostic marker data.
		else if (containsDragType(e, CodeDataTransfers.MARKERS)) {
			return ChatDragAndDropType.MARKER;
		}
		// Block Logic: Checks for external file drops.
		else if (containsDragType(e, DataTransfers.FILES)) {
			return ChatDragAndDropType.FILE_EXTERNAL;
		}
		// Block Logic: Checks for internal URI list transfers.
		else if (containsDragType(e, DataTransfers.INTERNAL_URI_LIST)) {
			return ChatDragAndDropType.FILE_INTERNAL;
		}
		// Block Logic: Checks for URI list transfers that might indicate folders or other resources.
		else if (containsDragType(e, Mimes.uriList, CodeDataTransfers.FILES, DataTransfers.RESOURCES)) {
			return ChatDragAndDropType.FOLDER;
		}

		return undefined;
	}

	/**
	 * @brief Checks if the current drag event is supported by the `ChatDragAndDrop` handler.
	 * @param e The DragEvent.
	 * @returns true if the guessed drop type is recognized and supported, false otherwise.
	 */
	private isDragEventSupported(e: DragEvent): boolean {
		// if guessed drop type is undefined, it means the drop is not supported
		const dropType = this.guessDropType(e);
		return dropType !== undefined;
	}

	/**
	 * @brief Returns a human-readable name for a given {@link ChatDragAndDropType}.
	 * @param type The {@link ChatDragAndDropType} enum value.
	 * @returns A localized string representing the type.
	 */
	private getDropTypeName(type: ChatDragAndDropType): string {
		switch (type) {
			case ChatDragAndDropType.FILE_INTERNAL: return localize('file', 'File');
			case ChatDragAndDropType.FILE_EXTERNAL: return localize('file', 'File');
			case ChatDragAndDropType.FOLDER: return localize('folder', 'Folder');
			case ChatDragAndDropType.IMAGE: return localize('image', 'Image');
			case ChatDragAndDropType.SYMBOL: return localize('symbol', 'Symbol');
			case ChatDragAndDropType.MARKER: return localize('problem', 'Problem');
			case ChatDragAndDropType.HTML: return localize('url', 'URL');
		}
	}

	/**
	 * @brief Determines if a drag event specifically involves image data.
	 * @param e The DragEvent.
	 * @returns true if the event contains image data, false otherwise.
	 *
	 * Logic: Checks `dataTransfer` for 'image' type or file types that start with 'image/'.
	 * It aims for no false positives, allowing false negatives.
	 */
	private isImageDnd(e: DragEvent): boolean {
		// Image detection should not have false positives, only false negatives are allowed
		// Inline: Checks if the data transfer directly contains an 'image' type.
		if (containsDragType(e, 'image')) {
			return true;
		}

		// Block Logic: If files are being dragged, inspect their MIME types.
		if (containsDragType(e, DataTransfers.FILES)) {
			const files = e.dataTransfer?.files;
			// Pre-condition: Files array exists and has at least one file.
			if (files && files.length > 0) {
				const file = files[0];
				// Inline: Checks if the file's type starts with 'image/'.
				return file.type.startsWith('image/');
			}

			const items = e.dataTransfer?.items;
			// Pre-condition: DataTransfer items exist and has at least one item.
			if (items && items.length > 0) {
				const item = items[0];
				// Inline: Checks if the item's type starts with 'image/'.
				return item.type.startsWith('image/');
			}
		}

		return false;
	}

	/**
	 * @brief Asynchronously extracts and resolves attachment contexts from a drag event.
	 * @param e The DragEvent.
	 * @returns A promise that resolves to an array of {@link IChatRequestVariableEntry}.
	 *
	 * Algorithm: Prioritizes marker data, then symbol data, then HTML/URL data for images,
	 * and finally generic editor input resources.
	 */
	private async getAttachContext(e: DragEvent): Promise<IChatRequestVariableEntry[]> {
		// Block Logic: If the drag event is not supported, return an empty array.
		// Pre-condition: The drag event has been assessed for supported drop types.
		if (!this.isDragEventSupported(e)) {
			return [];
		}

		// Block Logic: Attempts to extract diagnostic marker data.
		const markerData = extractMarkerDropData(e);
		if (markerData) {
			return this.resolveMarkerAttachContext(markerData);
		}

		// Block Logic: Attempts to extract document symbol data.
		if (containsDragType(e, CodeDataTransfers.SYMBOLS)) {
			const data = extractSymbolDropData(e);
			return this.resolveSymbolsAttachContext(data);
		}

		// Block Logic: Attempts to extract editor input data. If not found, checks for HTML/URI list that might contain images.
		const editorDragData = extractEditorsDropData(e);
		// Pre-condition: No editor drag data, but contains URI list and HTML/text data.
		if (editorDragData.length === 0 && !containsDragType(e, DataTransfers.INTERNAL_URI_LIST) && containsDragType(e, Mimes.uriList) && ((containsDragType(e, Mimes.html) || containsDragType(e, Mimes.text)))) {
			return this.resolveHTMLAttachContext(e);
		}

		// Block Logic: If editor drag data is found, resolve each editor input to an attach context.
		// Functional Utility: `coalesce` filters out any undefined results from the mapping.
		return coalesce(await Promise.all(editorDragData.map(editorInput => {
			return this.resolveAttachContext(editorInput);
		})));
	}

	/**
	 * @brief Resolves an `IDraggedResourceEditorInput` into an {@link IChatRequestVariableEntry}.
	 * @param editorInput The editor input representing the dropped resource.
	 * @returns A promise that resolves to an {@link IChatRequestVariableEntry} or undefined.
	 *
	 * Logic: First attempts to resolve the editor input as an image. If it's not an image
	 * or image handling is not enabled, it then attempts to resolve it as a general file.
	 */
	private async resolveAttachContext(editorInput: IDraggedResourceEditorInput): Promise<IChatRequestVariableEntry | undefined> {
		// Image
		// Block Logic: Attempts to resolve the editor input as an image, checking for proposed API enablement.
		const imageContext = await getImageAttachContext(editorInput, this.fileService, this.dialogService);
		if (imageContext) {
			return this.extensionService.extensions.some(ext => isProposedApiEnabled(ext, 'chatReferenceBinaryData')) ? imageContext : undefined;
		}

		// File
		// Block Logic: If not an image, or if image proposed API is not enabled, resolve as a general file.
		return await this.getEditorAttachContext(editorInput);
	}

	/**
	 * @brief Resolves an editor input (file or untitled) into a chat attachment context.
	 * @param editor The editor input, which can be a standard `EditorInput` or `IDraggedResourceEditorInput`.
	 * @returns A promise that resolves to an {@link IChatRequestVariableEntry} or undefined.
	 *
	 * Logic: Differentiates between untitled editors and regular file resources. For untitled
	 * editors, it may attempt to find existing models. For files, it performs a stat
	 * operation and then creates a resource attach context.
	 */
	private async getEditorAttachContext(editor: EditorInput | IDraggedResourceEditorInput): Promise<IChatRequestVariableEntry | undefined> {

		// Block Logic: Handles untitled editors specifically.
		if (isUntitledResourceEditorInput(editor)) {
			return await this.resolveUntitledAttachContext(editor);
		}

		// Pre-condition: Editor must have a resource to proceed with file system operations.
		if (!editor.resource) {
			return undefined;
		}

		let stat;
		// Block Logic: Attempts to stat the resource to determine if it's a file or directory.
		try {
			stat = await this.fileService.stat(editor.resource);
		} catch {
			// If stat fails, the resource might not exist or be accessible.
			return undefined;
		}

		// Block Logic: If the resource is neither a directory nor a file, it's not attachable.
		if (!stat.isDirectory && !stat.isFile) {
			return undefined;
		}

		// Functional Utility: Creates and returns a resource attachment entry.
		return await getResourceAttachContext(editor.resource, stat.isDirectory, this.textModelService);
	}

	/**
	 * @brief Resolves an untitled editor input into a chat attachment context.
	 * @param editor The untitled resource editor input.
	 * @returns A promise that resolves to an {@link IChatRequestVariableEntry} or undefined.
	 *
	 * Logic: If the untitled editor has a resource, it's directly used. Otherwise, it
	 * attempts to find an already open untitled editor with matching content.
	 */
	private async resolveUntitledAttachContext(editor: IDraggedResourceEditorInput): Promise<IChatRequestVariableEntry | undefined> {
		// Block Logic: If the resource is known, use it directly.
		if (editor.resource) {
			return await getResourceAttachContext(editor.resource, false, this.textModelService);
		}

		// Block Logic: If no resource, search for an existing untitled editor with the same content.
		const openUntitledEditors = this.editorService.editors.filter(editor => editor instanceof UntitledTextEditorInput) as UntitledTextEditorInput[];
		// Invariant: Iterates through open untitled editors to find a content match.
		for (const canidate of openUntitledEditors) {
			const model = await canidate.resolve();
			const contents = model.textEditorModel?.getValue();
			// If content matches, create an attach context from the candidate.
			if (contents === editor.contents) {
				return await getResourceAttachContext(canidate.resource, false, this.textModelService);
			}
		}

		return undefined;
	}

	/**
	 * @brief Converts an array of `DocumentSymbolTransferData` into an array of {@link ISymbolVariableEntry}.
	 * @param symbols An array of symbol transfer data.
	 * @returns An array of {@link ISymbolVariableEntry} objects.
	 */
	private resolveSymbolsAttachContext(symbols: DocumentSymbolTransferData[]): ISymbolVariableEntry[] {
		// Functional Utility: Maps each symbol data object to a symbol variable entry.
		return symbols.map(symbol => {
			const resource = URI.file(symbol.fsPath);
			return {
				kind: 'symbol',
				id: symbolId(resource, symbol.range),
				value: { uri: resource, range: symbol.range },
				symbolKind: symbol.kind,
				fullName: `$(${SymbolKinds.toIcon(symbol.kind).id}) ${symbol.name}`, // Human-readable symbol representation.
				name: symbol.name,
			};
		});
	}

	/**
	 * @brief Asynchronously downloads an image from a given URL and returns its content as a `Uint8Array`.
	 * @param url The URL of the image to download.
	 * @returns A promise that resolves to a `Uint8Array` containing the image data, or undefined if download fails.
	 */
	private async downloadImageAsUint8Array(url: string): Promise<Uint8Array | undefined> {
		// Block Logic: Attempts to read the image using the web content extractor service.
		try {
			const extractedImages = await this.webContentExtractorService.readImage(URI.parse(url), CancellationToken.None);
			if (extractedImages) {
				return extractedImages.buffer;
			}
		} catch (error) {
			// Functional Utility: Logs a warning if fetching the image fails.
			this.logService.warn('Fetch failed:', error);
		}

		// TODO: use dnd provider to insert text @justschen
		// Block Logic: If download fails, and a chat widget is focused, insert the URL as text.
		const selection = this.chatWidgetService.lastFocusedWidget?.inputEditor.getSelection();
		if (selection && this.chatWidgetService.lastFocusedWidget) {
			this.chatWidgetService.lastFocusedWidget.inputEditor.executeEdits('chatInsertUrl', [{ range: selection, text: url }]);
		}

		// Functional Utility: Logs a warning for unsupported image URL formats.
		this.logService.warn(`Image URLs must end in .jpg, .png, .gif, .webp, or .bmp. Failed to fetch image from this URL: ${url}`);
		return undefined;
	}

	/**
	 * @brief Asynchronously resolves HTML drag event data into image attachment contexts.
	 * @param e The DragEvent.
	 * @returns A promise that resolves to an array of {@link IChatRequestVariableEntry}.
	 *
	 * Logic: First attempts to extract an image from `e.dataTransfer.files`. If not found,
	 * it then tries to extract image URLs from `text/uri-list` and downloads/processes them.
	 */
	private async resolveHTMLAttachContext(e: DragEvent): Promise<IChatRequestVariableEntry[]> {
		const displayName = localize('dragAndDroppedImageName', 'Image from URL');
		let finalDisplayName = displayName;

		// Block Logic: Ensures unique display names for multiple attachments.
		for (let appendValue = 2; this.attachmentModel.attachments.some(attachment => attachment.name === finalDisplayName); appendValue++) {
			finalDisplayName = `${displayName} ${appendValue}`;
		}

		// Block Logic: Extracts image data directly from dropped files.
		const dataFromFile = await this.extractImageFromFile(e);
		if (dataFromFile) {
			return [await this.createImageVariable(await resizeImage(dataFromFile), finalDisplayName)];
		}

		// Block Logic: Extracts image URLs from the data transfer and attempts to download and process them.
		const dataFromUrl = await this.extractImageFromUrl(e);
		const variableEntries: IChatRequestVariableEntry[] = [];
		if (dataFromUrl) {
			// Invariant: Each URL in the dataFromUrl list is processed.
			for (const url of dataFromUrl) {
				// Inline: Checks for base64 encoded data URI scheme.
				if (/^data:image\/[a-z]+;base64,/.test(url)) {
					variableEntries.push(await this.createImageVariable(await resizeImage(url), finalDisplayName, URI.parse(url)));
				}
				// Inline: Checks for standard HTTP/HTTPS URLs.
				else if (/^https?:\/\/.+/.test(url)) {
					const imageData = await this.downloadImageAsUint8Array(url);
					if (imageData) {
						variableEntries.push(await this.createImageVariable(await resizeImage(imageData), finalDisplayName, URI.parse(url), url));
					}
				}
			}
		}

		return variableEntries;
	}

	/**
	 * @brief Creates an {@link IChatRequestVariableEntry} for image data.
	 * @param data The image data, either a `Uint8Array` or a base64 string.
	 * @param name The display name for the image.
	 * @param uri Optional: The URI of the image source.
	 * @param id Optional: A unique identifier for the image.
	 * @returns A promise that resolves to an {@link IChatRequestVariableEntry} for the image.
	 */
	private async createImageVariable(data: Uint8Array | string, name: string, uri?: URI, id?: string,): Promise<IChatRequestVariableEntry> {
		return {
			id: id || await imageToHash(data), // Functional Utility: Generates a hash as ID if not provided.
			name: name,
			value: data,
			isImage: true,
			isFile: false,
			isDirectory: false,
			references: uri ? [{ reference: uri, kind: 'reference' }] : [] // Includes URI as a reference if available.
		};
	}

	/**
	 * @brief Converts an array of `MarkerTransferData` into an array of {@link IDiagnosticVariableEntry}.
	 * @param markers An array of marker transfer data.
	 * @returns An array of {@link IDiagnosticVariableEntry} objects.
	 */
	private resolveMarkerAttachContext(markers: MarkerTransferData[]): IDiagnosticVariableEntry[] {
		// Functional Utility: Maps each marker data object to a diagnostic variable entry.
		return markers.map((marker): IDiagnosticVariableEntry => {
			let filter: IDiagnosticVariableEntryFilterData;
			// Block Logic: Determines the filter data based on marker properties.
			if (!('severity' in marker)) {
				filter = { filterUri: URI.revive(marker.uri), filterSeverity: MarkerSeverity.Warning };
			} else {
				filter = IDiagnosticVariableEntryFilterData.fromMarker(marker);
			}

			return IDiagnosticVariableEntryFilterData.toEntry(filter);
		});
	}

	/**
	 * @brief Sets the visibility and content of the drag-and-drop overlay.
	 * @param target The HTML element associated with the overlay.
	 * @param type The estimated type of content being dropped, or undefined to hide the overlay.
	 *
	 * Logic: Clears any existing overlay text, generates new text if a type is provided,
	 * applies styling, and toggles the 'visible' class on the overlay.
	 */
	private setOverlay(target: HTMLElement, type: ChatDragAndDropType | undefined): void {
		// Remove any previous overlay text
		this.overlayText?.remove();
		this.overlayText = undefined;

		const { overlay } = this.overlays.get(target)!;
		// Block Logic: If a drop type is provided, render overlay text.
		if (type !== undefined) {
			// Functional Utility: Renders a label with icons and localized text.
			const iconAndtextElements = renderLabelWithIcons(`$(${Codicon.attach.id}) ${this.getOverlayText(type)}`);
			// Invariant: Converts rendered elements into HTML span elements for display.
			const htmlElements = iconAndtextElements.map(element => {
				if (typeof element === 'string') {
					return $('span.overlay-text', undefined, element);
				}
				return element;
			});

			this.overlayText = $('span.attach-context-overlay-text', undefined, ...htmlElements);
			this.overlayText.style.backgroundColor = this.overlayTextBackground;
			overlay.appendChild(this.overlayText);
		}

		// Functional Utility: Toggles the 'visible' class based on whether a dropType is defined.
		overlay.classList.toggle('visible', type !== undefined);
	}

	/**
	 * @brief Generates localized text for the DND overlay based on the drop type.
	 * @param type The {@link ChatDragAndDropType} enum value.
	 * @returns A localized string indicating what kind of content will be attached.
	 */
	private getOverlayText(type: ChatDragAndDropType): string {
		const typeName = this.getDropTypeName(type);
		return localize('attacAsContext', 'Attach {0} as Context', typeName);
	}

	/**
	 * @brief Applies theme-dependent styles to a given overlay HTML element.
	 * @param overlay The HTML element representing the DND overlay.
	 */
	private updateOverlayStyles(overlay: HTMLElement): void {
		// Functional Utility: Retrieves themed colors for background and foreground.
		overlay.style.backgroundColor = this.getColor(this.styles.overlayBackground) || '';
		overlay.style.color = this.getColor(this.styles.listForeground) || '';
	}

	/**
	 * @brief Overrides the `Themable.updateStyles()` method to ensure overlays are restyled on theme changes.
	 */
	override updateStyles(): void {
		// Invariant: Iterates through all managed overlays and updates their styles.
		this.overlays.forEach(overlay => this.updateOverlayStyles(overlay.overlay));
		// Updates the background color for the overlay text.
		this.overlayTextBackground = this.getColor(this.styles.listBackground) || '';
	}

	/**
	 * @brief Asynchronously extracts image data as a `Uint8Array` from a `DragEvent`'s `files`.
	 * @param e The DragEvent.
	 * @returns A promise that resolves to a `Uint8Array` containing the image data, or undefined.
	 *
	 * Logic: Checks if files are present and if the first file is an image. If so, reads its content.
	 */
	private async extractImageFromFile(e: DragEvent): Promise<Uint8Array | undefined> {
		const files = e.dataTransfer?.files;
		// Block Logic: Checks if files exist and if the first file is an image.
		if (files && files.length > 0) {
			const file = files[0];
			if (file.type.startsWith('image/')) {
				// Functional Utility: Reads the file's content as an ArrayBuffer.
				try {
					const buffer = await file.arrayBuffer();
					return new Uint8Array(buffer);
				} catch (error) {
					// Functional Utility: Logs an error if file reading fails.
					this.logService.error('Error reading file:', error);
					return undefined;
				}
			}
		}

		return undefined;
	}

	/**
	 * @brief Asynchronously extracts a list of URIs from a `DragEvent`'s `text/uri-list` data.
	 * @param e The DragEvent.
	 * @returns A promise that resolves to an array of URI strings, or undefined if parsing fails.
	 *
	 * Logic: Retrieves the 'text/uri-list' data and attempts to parse it into an array of URIs.
	 */
	private async extractImageFromUrl(e: DragEvent): Promise<string[] | undefined> {
		const textUrl = e.dataTransfer?.getData('text/uri-list');
		// Block Logic: If URL text is found, attempt to parse it.
		if (textUrl) {
			// Functional Utility: Parses the URI list string into an array of URIs.
			try {
				const uris = UriList.parse(textUrl);
				if (uris.length > 0) {
					return uris;
				}
			} catch (error) {
				// Functional Utility: Logs an error if URI list parsing fails.
				this.logService.error('Error parsing URI list:', error);
				return undefined;
			}
		}

		return undefined;
	}
}

/**
 * @function getResourceAttachContext
 * @brief Creates an {@link IChatRequestVariableEntry} for a given URI, representing a file or directory.
 * @param resource The URI of the resource.
 * @param isDirectory A boolean indicating if the resource is a directory.
 * @param textModelService The service for creating and resolving text models.
 * @returns A promise that resolves to an {@link IChatRequestVariableEntry} or undefined.
 *
 * Logic: Checks if the resource is a non-directory file and if it can be opened as a text model.
 * Also marks certain file types (e.g., SVG) as omitted if they are not directly supported.
 */
async function getResourceAttachContext(resource: URI, isDirectory: boolean, textModelService: ITextModelService): Promise<IChatRequestVariableEntry | undefined> {
	let isOmitted = false;

	// Block Logic: If the resource is not a directory, perform additional checks.
	if (!isDirectory) {
		// Functional Utility: Tries to create a text model reference for the resource.
		// If it fails, the resource might not be a readable text file, so it's marked as omitted.
		try {
			const createdModel = await textModelService.createModelReference(resource);
			createdModel.dispose();
		} catch {
			isOmitted = true;
		}

		// Inline: Specifically omits SVG files from being attached directly as readable content.
		if (/\.(svg)$/i.test(resource.path)) {
			isOmitted = true;
		}
	}

	return {
		value: resource,
		id: resource.toString(),
		name: basename(resource),
		isFile: !isDirectory,
		isDirectory,
		isOmitted // Indicates if the content was omitted for technical reasons.
	};
}

/**
 * @function getImageAttachContext
 * @brief Creates an {@link IChatRequestVariableEntry} for image data from an editor input.
 * @param editor The editor input, potentially containing an image resource.
 * @param fileService The service for file system operations.
 * @param dialogService The service for displaying dialogs.
 * @returns A promise that resolves to an {@link IChatRequestVariableEntry} for the image, or undefined.
 *
 * Logic: Checks if the editor resource is a recognized image file type. If so, reads its content,
 * performs a size check (30MB limit), resizes the image, and then creates an attachment entry.
 */
async function getImageAttachContext(editor: EditorInput | IDraggedResourceEditorInput, fileService: IFileService, dialogService: IDialogService): Promise<IChatRequestVariableEntry | undefined> {
	// Pre-condition: The editor must have a resource.
	if (!editor.resource) {
		return undefined;
	}

	// Block Logic: Checks if the resource path matches common image file extensions.
	if (/\.(png|jpg|jpeg|gif|webp)$/i.test(editor.resource.path)) {
		const fileName = basename(editor.resource);
		// Functional Utility: Reads the content of the image file.
		const readFile = await fileService.readFile(editor.resource);
		// Error Handling: Checks if the image file size exceeds a predefined limit.
		if (readFile.size > 30 * 1024 * 1024) { // 30 MB
			dialogService.error(localize('imageTooLarge', 'Image is too large'), localize('imageTooLargeMessage', 'The image {0} is too large to be attached.', fileName));
			throw new Error('Image is too large');
		}
		// Functional Utility: Resizes the image buffer.
		const resizedImage = await resizeImage(readFile.value.buffer);
		return {
			id: editor.resource.toString(),
			name: fileName,
			fullName: editor.resource.path,
			value: resizedImage,
			icon: Codicon.fileMedia, // Assigns a media file icon for visual representation.
			isImage: true,
			isFile: false,
			references: [{ reference: editor.resource, kind: 'reference' }] // Includes the resource as a reference.
		};
	}

	return undefined;
}

/**
 * @function symbolId
 * @brief Generates a unique string identifier for a given document symbol.
 * @param resource The URI of the document containing the symbol.
 * @param range Optional: The {@link IRange} of the symbol within the document.
 * @returns A string identifier for the symbol.
 *
 * Logic: Combines the file system path of the resource with the symbol's
 * line numbers (if available) to create a unique identifier.
 */
function symbolId(resource: URI, range?: IRange): string {
	let rangePart = '';
	// Block Logic: If a range is provided, append line number information to the ID.
	if (range) {
		// Inline: Appends the start line number.
		rangePart = `:${range.startLineNumber}`;
		// Inline: If the symbol spans multiple lines, append the end line number.
		if (range.startLineNumber !== range.endLineNumber) {
			rangePart += `-${range.endLineNumber}`;
		}
	}
	// Combines the file path and optional range part for the unique ID.
	return resource.fsPath + rangePart;
}
