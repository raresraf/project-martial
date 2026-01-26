/**
 * @file chatDragAndDrop.ts
 * @brief Handles drag and drop functionality for the chat widget.
 * @details This file implements the logic for dragging and dropping various types of content
 * into the chat input area. It supports attaching files, folders, images, symbols, and problems
 * (markers) as context to the chat. It manages the UI feedback for the drag and drop operation
 * and processes the dropped data to be added to the chat model.
 */
/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

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
 * @enum {number} ChatDragAndDropType
 * @brief Enumerates the different types of content that can be dragged and dropped.
 */
enum ChatDragAndDropType {
	FILE_INTERNAL,
	FILE_EXTERNAL,
	FOLDER,
	IMAGE,
	SYMBOL,
	HTML,
	MARKER,
}

/**
 * @class ChatDragAndDrop
 * @brief Manages the drag and drop functionality for the chat input.
 * @details This class is responsible for creating and managing drag and drop overlays,
 * detecting the type of content being dragged, providing visual feedback, and handling
 * the drop event to attach the content to the chat.
 */
export class ChatDragAndDrop extends Themable {

	private readonly overlays: Map<HTMLElement, { overlay: HTMLElement; disposable: IDisposable }> = new Map();
	private overlayText?: HTMLElement;
	private overlayTextBackground: string = '';

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

		this.updateStyles();
	}

	/**
	 * @brief Adds a drag and drop overlay to a target element.
	 * @param target The HTML element to which the overlay is added.
	 * @param overlayContainer The container element for the overlay.
	 */
	addOverlay(target: HTMLElement, overlayContainer: HTMLElement): void {
		this.removeOverlay(target);

		const { overlay, disposable } = this.createOverlay(target, overlayContainer);
		this.overlays.set(target, { overlay, disposable });
	}

	/**
	 * @brief Removes the drag and drop overlay from a target element.
	 * @param target The HTML element from which the overlay is removed.
	 */
	removeOverlay(target: HTMLElement): void {
		if (this.currentActiveTarget === target) {
			this.currentActiveTarget = undefined;
		}

		const existingOverlay = this.overlays.get(target);
		if (existingOverlay) {
			existingOverlay.overlay.remove();
			existingOverlay.disposable.dispose();
			this.overlays.delete(target);
		}
	}

	private currentActiveTarget: HTMLElement | undefined = undefined;

	/**
	 * @brief Creates and initializes the drag and drop overlay and its event listeners.
	 * @param target The target HTML element for drag and drop events.
	 * @param overlayContainer The container for the overlay element.
	 * @returns An object containing the overlay element and a disposable for its event listeners.
	 */
	private createOverlay(target: HTMLElement, overlayContainer: HTMLElement): { overlay: HTMLElement; disposable: IDisposable } {
		const overlay = document.createElement('div');
		overlay.classList.add('chat-dnd-overlay');
		this.updateOverlayStyles(overlay);
		overlayContainer.appendChild(overlay);

		const disposable = new DragAndDropObserver(target, {
			onDragOver: (e) => {
				e.stopPropagation();
				e.preventDefault();

				if (target === this.currentActiveTarget) {
					return;
				}

				if (this.currentActiveTarget) {
					this.setOverlay(this.currentActiveTarget, undefined);
				}

				this.currentActiveTarget = target;

				this.onDragEnter(e, target);

			},
			onDragLeave: (e) => {
				if (target === this.currentActiveTarget) {
					this.currentActiveTarget = undefined;
				}

				this.onDragLeave(e, target);
			},
			onDrop: (e) => {
				e.stopPropagation();
				e.preventDefault();

				if (target !== this.currentActiveTarget) {
					return;
				}

				this.currentActiveTarget = undefined;
				this.onDrop(e, target);
			},
		});

		return { overlay, disposable };
	}

	/**
	 * @brief Handles the drag enter event.
	 * @param e The drag event.
	 * @param target The target HTML element.
	 */
	private onDragEnter(e: DragEvent, target: HTMLElement): void {
		const estimatedDropType = this.guessDropType(e);
		this.updateDropFeedback(e, target, estimatedDropType);
	}

	/**
	 * @brief Handles the drag leave event.
	 * @param e The drag event.
	 * @param target The target HTML element.
	 */
	private onDragLeave(e: DragEvent, target: HTMLElement): void {
		this.updateDropFeedback(e, target, undefined);
	}

	/**
	 * @brief Handles the drop event.
	 * @param e The drag event.
	 * @param target The target HTML element.
	 */
	private onDrop(e: DragEvent, target: HTMLElement): void {
		this.updateDropFeedback(e, target, undefined);
		this.drop(e);
	}

	/**
	 * @brief Processes the dropped content and attaches it to the chat.
	 * @param e The drag event.
	 */
	private async drop(e: DragEvent): Promise<void> {
		const contexts = await this.getAttachContext(e);
		if (contexts.length === 0) {
			return;
		}

		this.attachmentModel.addContext(...contexts);
	}

	/**
	 * @brief Updates the visual feedback for the drag and drop operation.
	 * @param e The drag event.
	 * @param target The target HTML element.
	 * @param dropType The estimated type of the dropped content.
	 */
	private updateDropFeedback(e: DragEvent, target: HTMLElement, dropType: ChatDragAndDropType | undefined): void {
		const showOverlay = dropType !== undefined;
		if (e.dataTransfer) {
			e.dataTransfer.dropEffect = showOverlay ? 'copy' : 'none';
		}

		this.setOverlay(target, dropType);
	}

	/**
	 * @brief Guesses the type of content being dragged based on the data transfer object.
	 * @param e The drag event.
	 * @returns The estimated drop type, or undefined if not supported.
	 */
	private guessDropType(e: DragEvent): ChatDragAndDropType | undefined {
		// This is an esstimation based on the datatransfer types/items
		if (this.isImageDnd(e)) {
			return this.extensionService.extensions.some(ext => isProposedApiEnabled(ext, 'chatReferenceBinaryData')) ? ChatDragAndDropType.IMAGE : undefined;
		} else if (containsDragType(e, 'text/html')) {
			return ChatDragAndDropType.HTML;
		} else if (containsDragType(e, CodeDataTransfers.SYMBOLS)) {
			return ChatDragAndDropType.SYMBOL;
		} else if (containsDragType(e, CodeDataTransfers.MARKERS)) {
			return ChatDragAndDropType.MARKER;
		} else if (containsDragType(e, DataTransfers.FILES)) {
			return ChatDragAndDropType.FILE_EXTERNAL;
		} else if (containsDragType(e, DataTransfers.INTERNAL_URI_LIST)) {
			return ChatDragAndDropType.FILE_INTERNAL;
		} else if (containsDragType(e, Mimes.uriList, CodeDataTransfers.FILES, DataTransfers.RESOURCES)) {
			return ChatDragAndDropType.FOLDER;
		}

		return undefined;
	}

	/**
	 * @brief Checks if the drag event is for a supported content type.
	 * @param e The drag event.
	 * @returns True if the drop is supported, false otherwise.
	 */
	private isDragEventSupported(e: DragEvent): boolean {
		// if guessed drop type is undefined, it means the drop is not supported
		const dropType = this.guessDropType(e);
		return dropType !== undefined;
	}

	/**
	 * @brief Gets a human-readable name for a drop type.
	 * @param type The drop type.
	 * @returns The name of the drop type.
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
	 * @brief Checks if the drag event is for an image.
	 * @param e The drag event.
	 * @returns True if the dragged content is an image, false otherwise.
	 */
	private isImageDnd(e: DragEvent): boolean {
		// Image detection should not have false positives, only false negatives are allowed
		if (containsDragType(e, 'image')) {
			return true;
		}

		if (containsDragType(e, DataTransfers.FILES)) {
			const files = e.dataTransfer?.files;
			if (files && files.length > 0) {
				const file = files[0];
				return file.type.startsWith('image/');
			}

			const items = e.dataTransfer?.items;
			if (items && items.length > 0) {
				const item = items[0];
				return item.type.startsWith('image/');
			}
		}

		return false;
	}

	/**
	 * @brief Resolves the attach context from a drag event.
	 * @param e The drag event.
	 * @returns A promise that resolves to an array of chat request variable entries.
	 */
	private async getAttachContext(e: DragEvent): Promise<IChatRequestVariableEntry[]> {
		if (!this.isDragEventSupported(e)) {
			return [];
		}

		const markerData = extractMarkerDropData(e);
		if (markerData) {
			return this.resolveMarkerAttachContext(markerData);
		}

		if (containsDragType(e, CodeDataTransfers.SYMBOLS)) {
			const data = extractSymbolDropData(e);
			return this.resolveSymbolsAttachContext(data);
		}

		const editorDragData = extractEditorsDropData(e);
		if (editorDragData.length === 0 && !containsDragType(e, DataTransfers.INTERNAL_URI_LIST) && containsDragType(e, Mimes.uriList) && ((containsDragType(e, Mimes.html) || containsDragType(e, Mimes.text)))) {
			return this.resolveHTMLAttachContext(e);
		}

		return coalesce(await Promise.all(editorDragData.map(editorInput => {
			return this.resolveAttachContext(editorInput);
		})));
	}

	/**
	 * @brief Resolves the attach context for a single editor input.
	 * @param editorInput The dragged editor input.
	 * @returns A promise that resolves to a chat request variable entry, or undefined.
	 */
	private async resolveAttachContext(editorInput: IDraggedResourceEditorInput): Promise<IChatRequestVariableEntry | undefined> {
		// Image
		const imageContext = await getImageAttachContext(editorInput, this.fileService, this.dialogService);
		if (imageContext) {
			return this.extensionService.extensions.some(ext => isProposedApiEnabled(ext, 'chatReferenceBinaryData')) ? imageContext : undefined;
		}

		// File
		return await this.getEditorAttachContext(editorInput);
	}

	/**
	 * @brief Resolves the attach context for an editor input.
	 * @param editor The editor input.
	 * @returns A promise that resolves to a chat request variable entry, or undefined.
	 */
	private async getEditorAttachContext(editor: EditorInput | IDraggedResourceEditorInput): Promise<IChatRequestVariableEntry | undefined> {

		// untitled editor
		if (isUntitledResourceEditorInput(editor)) {
			return await this.resolveUntitledAttachContext(editor);
		}

		if (!editor.resource) {
			return undefined;
		}

		let stat;
		try {
			stat = await this.fileService.stat(editor.resource);
		} catch {
			return undefined;
		}

		if (!stat.isDirectory && !stat.isFile) {
			return undefined;
		}

		return await getResourceAttachContext(editor.resource, stat.isDirectory, this.textModelService);
	}

	/**
	 * @brief Resolves the attach context for an untitled editor input.
	 * @param editor The untitled editor input.
	 * @returns A promise that resolves to a chat request variable entry, or undefined.
	 */
	private async resolveUntitledAttachContext(editor: IDraggedResourceEditorInput): Promise<IChatRequestVariableEntry | undefined> {
		// If the resource is known, we can use it directly
		if (editor.resource) {
			return await getResourceAttachContext(editor.resource, false, this.textModelService);
		}

		// Otherwise, we need to check if the contents are already open in another editor
		const openUntitledEditors = this.editorService.editors.filter(editor => editor instanceof UntitledTextEditorInput) as UntitledTextEditorInput[];
		for (const canidate of openUntitledEditors) {
			const model = await canidate.resolve();
			const contents = model.textEditorModel?.getValue();
			if (contents === editor.contents) {
				return await getResourceAttachContext(canidate.resource, false, this.textModelService);
			}
		}

		return undefined;
	}

	/**
	 * @brief Resolves the attach context for document symbols.
	 * @param symbols An array of document symbol transfer data.
	 * @returns An array of symbol variable entries.
	 */
	private resolveSymbolsAttachContext(symbols: DocumentSymbolTransferData[]): ISymbolVariableEntry[] {
		return symbols.map(symbol => {
			const resource = URI.file(symbol.fsPath);
			return {
				kind: 'symbol',
				id: symbolId(resource, symbol.range),
				value: { uri: resource, range: symbol.range },
				symbolKind: symbol.kind,
				fullName: `$(${SymbolKinds.toIcon(symbol.kind).id}) ${symbol.name}`,
				name: symbol.name,
			};
		});
	}

	/**
	 * @brief Downloads an image from a URL as a Uint8Array.
	 * @param url The URL of the image.
	 * @returns A promise that resolves to a Uint8Array, or undefined if the download fails.
	 */
	private async downloadImageAsUint8Array(url: string): Promise<Uint8Array | undefined> {
		try {
			const extractedImages = await this.webContentExtractorService.readImage(URI.parse(url), CancellationToken.None);
			if (extractedImages) {
				return extractedImages.buffer;
			}
		} catch (error) {
			this.logService.warn('Fetch failed:', error);
		}

		// TODO: use dnd provider to insert text @justschen
		const selection = this.chatWidgetService.lastFocusedWidget?.inputEditor.getSelection();
		if (selection && this.chatWidgetService.lastFocusedWidget) {
			this.chatWidgetService.lastFocusedWidget.inputEditor.executeEdits('chatInsertUrl', [{ range: selection, text: url }]);
		}

		this.logService.warn(`Image URLs must end in .jpg, .png, .gif, .webp, or .bmp. Failed to fetch image from this URL: ${url}`);
		return undefined;
	}

	/**
	 * @brief Resolves the attach context for HTML content, which may contain image URLs.
	 * @param e The drag event.
	 * @returns A promise that resolves to an array of chat request variable entries.
	 */
	private async resolveHTMLAttachContext(e: DragEvent): Promise<IChatRequestVariableEntry[]> {
		const displayName = localize('dragAndDroppedImageName', 'Image from URL');
		let finalDisplayName = displayName;

		for (let appendValue = 2; this.attachmentModel.attachments.some(attachment => attachment.name === finalDisplayName); appendValue++) {
			finalDisplayName = `${displayName} ${appendValue}`;
		}

		const dataFromFile = await this.extractImageFromFile(e);
		if (dataFromFile) {
			return [await this.createImageVariable(await resizeImage(dataFromFile), finalDisplayName)];
		}

		const dataFromUrl = await this.extractImageFromUrl(e);
		const variableEntries: IChatRequestVariableEntry[] = [];
		if (dataFromUrl) {
			for (const url of dataFromUrl) {
				if (/^data:image\/[a-z]+;base64,/.test(url)) {
					variableEntries.push(await this.createImageVariable(await resizeImage(url), finalDisplayName, URI.parse(url)));
				} else if (/^https?:\/\/.+/.test(url)) {
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
	 * @brief Creates an image variable entry for the chat model.
	 * @param data The image data as a Uint8Array.
	 * @param name The name of the image.
	 * @param uri The URI of the image, if available.
	 * @param id The ID of the image, if available.
	 * @returns A promise that resolves to a chat request variable entry for the image.
	 */
	private async createImageVariable(data: Uint8Array, name: string, uri?: URI, id?: string,): Promise<IChatRequestVariableEntry> {
		return {
			id: id || await imageToHash(data),
			name: name,
			value: data,
			isImage: true,
			isFile: false,
			isDirectory: false,
			references: uri ? [{ reference: uri, kind: 'reference' }] : []
		};
	}

	/**
	 * @brief Resolves the attach context for markers (problems).
	 * @param markers An array of marker transfer data.
	 * @returns An array of diagnostic variable entries.
	 */
	private resolveMarkerAttachContext(markers: MarkerTransferData[]): IDiagnosticVariableEntry[] {
		return markers.map((marker): IDiagnosticVariableEntry => {
			let filter: IDiagnosticVariableEntryFilterData;
			if (!('severity' in marker)) {
				filter = { filterUri: URI.revive(marker.uri), filterSeverity: MarkerSeverity.Warning };
			} else {
				filter = IDiagnosticVariableEntryFilterData.fromMarker(marker);
			}

			return IDiagnosticVariableEntryFilterData.toEntry(filter);
		});
	}

	/**
	 * @brief Sets the content and visibility of the drag and drop overlay.
	 * @param target The target HTML element.
	 * @param type The type of content being dragged.
	 */
	private setOverlay(target: HTMLElement, type: ChatDragAndDropType | undefined): void {
		// Remove any previous overlay text
		this.overlayText?.remove();
		this.overlayText = undefined;

		const { overlay } = this.overlays.get(target)!;
		if (type !== undefined) {
			// Render the overlay text

			const iconAndtextElements = renderLabelWithIcons(`$(${Codicon.attach.id}) ${this.getOverlayText(type)}`);
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

		overlay.classList.toggle('visible', type !== undefined);
	}

	/**
	 * @brief Gets the text to display in the overlay.
	 * @param type The type of content being dragged.
	 * @returns The overlay text.
	 */
	private getOverlayText(type: ChatDragAndDropType): string {
		const typeName = this.getDropTypeName(type);
		return localize('attacAsContext', 'Attach {0} as Context', typeName);
	}

	/**
	 * @brief Updates the styles of the overlay based on the current theme.
	 * @param overlay The overlay element.
	 */
	private updateOverlayStyles(overlay: HTMLElement): void {
		overlay.style.backgroundColor = this.getColor(this.styles.overlayBackground) || '';
		overlay.style.color = this.getColor(this.styles.listForeground) || '';
	}

	/**
	 * @brief Updates the styles of all overlays and the overlay text background.
	 */
	override updateStyles(): void {
		this.overlays.forEach(overlay => this.updateOverlayStyles(overlay.overlay));
		this.overlayTextBackground = this.getColor(this.styles.listBackground) || '';
	}



	/**
	 * @brief Extracts image data from a file in a drag event.
	 * @param e The drag event.
	 * @returns A promise that resolves to a Uint8Array, or undefined if no image file is found.
	 */
	private async extractImageFromFile(e: DragEvent): Promise<Uint8Array | undefined> {
		const files = e.dataTransfer?.files;
		if (files && files.length > 0) {
			const file = files[0];
			if (file.type.startsWith('image/')) {
				try {
					const buffer = await file.arrayBuffer();
					return new Uint8Array(buffer);
				} catch (error) {
					this.logService.error('Error reading file:', error);
					return undefined;
				}
			}
		}

		return undefined;
	}

	/**
	 * @brief Extracts image URLs from a drag event.
	 * @param e The drag event.
	 * @returns A promise that resolves to an array of image URLs, or undefined.
	 */
	private async extractImageFromUrl(e: DragEvent): Promise<string[] | undefined> {
		const textUrl = e.dataTransfer?.getData('text/uri-list');
		if (textUrl) {
			try {
				const uris = UriList.parse(textUrl);
				if (uris.length > 0) {
					return uris;
				}
			} catch (error) {
				this.logService.error('Error parsing URI list:', error);
				return undefined;
			}
		}

		return undefined;
	}


}

/**
 * @brief Gets the attach context for a resource.
 * @param resource The URI of the resource.
 * @param isDirectory Whether the resource is a directory.
 * @param textModelService The text model service.
 * @returns A promise that resolves to a chat request variable entry, or undefined.
 */
async function getResourceAttachContext(resource: URI, isDirectory: boolean, textModelService: ITextModelService): Promise<IChatRequestVariableEntry | undefined> {
	let isOmitted = false;

	if (!isDirectory) {
		try {
			const createdModel = await textModelService.createModelReference(resource);
			createdModel.dispose();
		} catch {
			isOmitted = true;
		}

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
		isOmitted
	};
}

/**
 * @brief Gets the attach context for an image.
 * @param editor The editor input.
 * @param fileService The file service.
 * @param dialogService The dialog service.
 * @returns A promise that resolves to a chat request variable entry, or undefined.
 */
async function getImageAttachContext(editor: EditorInput | IDraggedResourceEditorInput, fileService: IFileService, dialogService: IDialogService): Promise<IChatRequestVariableEntry | undefined> {
	if (!editor.resource) {
		return undefined;
	}

	if (/\.(png|jpg|jpeg|gif|webp)$/i.test(editor.resource.path)) {
		const fileName = basename(editor.resource);
		const readFile = await fileService.readFile(editor.resource);
		if (readFile.size > 30 * 1024 * 1024) { // 30 MB
			dialogService.error(localize('imageTooLarge', 'Image is too large'), localize('imageTooLargeMessage', 'The image {0} is too large to be attached.', fileName));
			throw new Error('Image is too large');
		}
		const resizedImage = await resizeImage(readFile.value.buffer);
		return {
			id: editor.resource.toString(),
			name: fileName,
			fullName: editor.resource.path,
			value: resizedImage,
			icon: Codicon.fileMedia,
			isImage: true,
			isFile: false,
			references: [{ reference: editor.resource, kind: 'reference' }]
		};
	}

	return undefined;
}

/**
 * @brief Generates a unique ID for a symbol.
 * @param resource The URI of the resource containing the symbol.
 * @param range The range of the symbol.
 * @returns A unique ID for the symbol.
 */
function symbolId(resource: URI, range?: IRange): string {
	let rangePart = '';
	if (range) {
		rangePart = `:${range.startLineNumber}`;
		if (range.startLineNumber !== range.endLineNumber) {
			rangePart += `-${range.endLineNumber}`;
		}
	}
	return resource.fsPath + rangePart;
}
