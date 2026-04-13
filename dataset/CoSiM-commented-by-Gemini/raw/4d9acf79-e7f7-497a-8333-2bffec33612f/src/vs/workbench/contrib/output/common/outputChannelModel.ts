/**
 * @fileoverview
 * @module vs/workbench/contrib/output/common/outputChannelModel
 * @description
 * This module defines the data models for output channels in Visual Studio Code. It provides interfaces
 * and implementations for handling output content, including file-backed output channels that can be
 * displayed in the editor. This allows for features like appending, clearing, and replacing content
 * in a way that is decoupled from the UI representation.
 */

import { IInstantiationService } from '../../../../platform/instantiation/common/instantiation.js';
import * as resources from '../../../../base/common/resources.js';
import { ITextModel } from '../../../../editor/common/model.js';
import { IEditorWorkerService } from '../../../../editor/common/services/editorWorker.js';
import { Emitter, Event } from '../../../../base/common/event.js';
import { URI } from '../../../../base/common/uri.js';
import { ThrottledDelayer } from '../../../../base/common/async.js';
import { FileOperationResult, IFileService, toFileOperationResult } from '../../../../platform/files/common/files.js';
import { IModelService } from '../../../../editor/common/services/model.js';
import { ILanguageSelection } from '../../../../editor/common/languages/language.js';
import { Disposable, toDisposable, IDisposable, MutableDisposable, DisposableStore } from '../../../../base/common/lifecycle.js';
import { isNumber } from '../../../../base/common/types.js';
import { EditOperation, ISingleEditOperation } from '../../../../editor/common/core/editOperation.js';
import { Position } from '../../../../editor/common/core/position.js';
import { Range } from '../../../../editor/common/core/range.js';
import { VSBuffer } from '../../../../base/common/buffer.js';
import { ILogger, ILoggerService, ILogService } from '../../../../platform/log/common/log.js';
import { CancellationToken, CancellationTokenSource } from '../../../../base/common/cancellation.js';
import { OutputChannelUpdateMode } from '../../../services/output/common/output.js';
import { isCancellationError } from '../../../../base/common/errors.js';

/**
 * @interface IOutputChannelModel
 * @description Represents the data model for an output channel. It provides methods for manipulating
 * the output content, such as appending, clearing, and replacing text. This interface decouples the
 * data handling from the UI, allowing for different underlying storage mechanisms (e.g., in-memory, file-based).
 */
export interface IOutputChannelModel extends IDisposable {
	/**
	 * @event onDispose
	 * @description An event that is fired when the model is disposed.
	 */
	readonly onDispose: Event<void>;
	/**
	 * @method append
	 * @description Appends a string of text to the output channel.
	 * @param {string} output The text to append.
	 */
	append(output: string): void;
	/**
	 * @method update
	 * @description Updates the output channel based on the given mode. This can be used to append,
	 * clear, or replace the content.
	 * @param {OutputChannelUpdateMode} mode The update mode to apply.
	 * @param {number | undefined} till An optional offset to use for operations like 'replace'.
	 * @param {boolean} immediate Whether the update should be performed immediately or throttled.
	 */
	update(mode: OutputChannelUpdateMode, till: number | undefined, immediate: boolean): void;
	/**
	 * @method loadModel
	 * @description Loads the underlying text model for the output channel. This is typically used
	 * to display the output in an editor.
	 * @returns {Promise<ITextModel>} A promise that resolves with the text model.
	 */
	loadModel(): Promise<ITextModel>;
	/**
	 * @method clear
	 * @description Clears all content from the output channel.
	 */
	clear(): void;
	/**
	 * @method replace
	 * @description Replaces the entire content of the output channel with a new value.
	 * @param {string} value The new content for the output channel.
	 */
	replace(value: string): void;
}

/**
 * @interface IContentProvider
 * @description An interface for providing content to an output channel model. This allows for different
 * sources of content, such as files or in-memory buffers.
 */
interface IContentProvider {
	/**
	 * @event onDidAppend
	 * @description An event that is fired when new content has been appended to the source.
	 */
	readonly onDidAppend: Event<void>;
	/**
	 * @event onDidReset
	 * @description An event that is fired when the content source has been reset.
	 */
	readonly onDidReset: Event<void>;
	/**
	 * @method reset
	 * @description Resets the content provider, optionally to a specific offset.
	 */
	reset(): void;
	/**
	 * @method watch
	 * @description Starts watching the content source for changes.
	 * @returns {IDisposable} A disposable to stop watching.
	 */
	watch(): IDisposable;
	/**
	 * @method getContent
	 * @description Retrieves content from the source.
	 * @returns {Promise<{ readonly content: string; readonly consume: () => void }>} A promise that resolves with the content and a function to consume it.
	 */
	getContent(): Promise<{ readonly content: string; readonly consume: () => void }>;
}

/**
 * @class FileContentProvider
 * @description Provides content for an output channel from a file. It polls the file for changes
 * and notifies listeners when new content is available.
 */
class FileContentProvider extends Disposable implements IContentProvider {

	private readonly _onDidAppend = new Emitter<void>();
	readonly onDidAppend = this._onDidAppend.event;

	private readonly _onDidReset = new Emitter<void>();
	readonly onDidReset = this._onDidReset.event;

	private watching: boolean = false;
	private syncDelayer: ThrottledDelayer<void>;
	private etag: string | undefined = '';

	private startOffset: number = 0;
	private endOffset: number = 0;

	constructor(
		private readonly file: URI,
		@IFileService private readonly fileService: IFileService,
		@ILogService private readonly logService: ILogService,
	) {
		super();

		this.syncDelayer = new ThrottledDelayer<void>(500);
		this._register(toDisposable(() => this.unwatch()));
	}

	/**
	 * @method reset
	 * @description Resets the read offsets to a specific position or the current start.
	 * @param {number} [offset] - The offset to reset to.
	 */
	reset(offset?: number): void {
		this.endOffset = this.startOffset = offset ?? this.startOffset;
	}

	/**
	 * @method resetToEnd
	 * @description Resets the start offset to the current end offset, effectively skipping past content that has already been read.
	 */
	resetToEnd(): void {
		this.startOffset = this.endOffset;
	}

	/**
	 * @method watch
	 * @description Begins polling the file for changes.
	 * @returns {IDisposable} A disposable that can be used to stop polling.
	 */
	watch(): IDisposable {
		if (!this.watching) {
			this.logService.trace('Started polling', this.file.toString());
			this.poll(true);
			this.watching = true;
		}
		return toDisposable(() => this.unwatch());
	}

	private unwatch(): void {
		if (this.watching) {
			this.syncDelayer.cancel();
			this.watching = false;
			this.logService.trace('Stopped polling', this.file.toString());
		}
	}

	/**
	 * @method poll
	 * @description Schedules a polling operation to check for file changes.
	 * @param {boolean} [immediate] - If true, the poll is scheduled with no delay.
	 */
	private poll(immediate?: boolean): void {
		const loop = () => this.doWatch().then(() => this.poll());
		this.syncDelayer.trigger(loop, immediate ? 0 : undefined).catch(error => {
			if (!isCancellationError(error)) {
				throw error;
			}
		});
	}

	/**
	 * @method doWatch
	 * @description Performs a single polling operation to check for file modifications. It compares the file's etag to detect changes.
	 */
	private async doWatch(): Promise<void> {
		try {
			const stat = await this.fileService.stat(this.file);
			if (stat.etag !== this.etag) {
				this.etag = stat.etag;
				if (isNumber(stat.size) && this.endOffset > stat.size) {
					// The file has been truncated, so reset.
					this.reset(0);
					this._onDidReset.fire();
				} else {
					// The file has been appended to.
					this._onDidAppend.fire();
				}
			}
		} catch (error) {
			// Ignore file not found errors, as the file may be created later.
			if (toFileOperationResult(error) !== FileOperationResult.FILE_NOT_FOUND) {
				throw error;
			}
		}
	}

	/**
	 * @method getContent
	 * @description Reads content from the file starting from the current end offset.
	 * @returns {Promise<{ readonly content: string; readonly consume: () => void }>} The content read and a function to call to mark the content as consumed, which updates the internal offsets.
	 */
	async getContent(): Promise<{ readonly content: string; readonly consume: () => void }> {
		try {
			const content = await this.fileService.readFile(this.file, { position: this.endOffset });
			let consumed = false;
			return {
				content: content.value.toString(),
				consume: () => {
					if (!consumed) {
						consumed = true;
						this.endOffset += content.value.byteLength;
						this.etag = content.etag;
					}
				}
			};
		} catch (error) {
			if (toFileOperationResult(error) !== FileOperationResult.FILE_NOT_FOUND) {
				throw error;
			}
			return {
				content: '',
				consume: () => { /* No Op */ }
			};
		}
	}
}

/**
 * @class AbstractFileOutputChannelModel
 * @description An abstract base class for output channel models that are backed by a file. It handles the logic for creating and updating an ITextModel from a content provider.
 */
export abstract class AbstractFileOutputChannelModel extends Disposable implements IOutputChannelModel {

	private readonly _onDispose = this._register(new Emitter<void>());
	readonly onDispose: Event<void> = this._onDispose.event;

	private readonly modelDisposable = this._register(new MutableDisposable<DisposableStore>());
	protected model: ITextModel | null = null;
	private modelUpdateInProgress: boolean = false;
	private readonly modelUpdateCancellationSource = this._register(new MutableDisposable<CancellationTokenSource>());
	private readonly appendThrottler = this._register(new ThrottledDelayer(300));
	private replacePromise: Promise<void> | undefined;

	constructor(
		private readonly modelUri: URI,
		private readonly language: ILanguageSelection,
		private readonly outputContentProvider: IContentProvider,
		@IModelService protected readonly modelService: IModelService,
		@IEditorWorkerService private readonly editorWorkerService: IEditorWorkerService,
	) {
		super();
	}

	/**
	 * @method loadModel
	 * @description Creates and loads the underlying `ITextModel` for the output channel's content.
	 * If the model already exists, it returns the existing instance.
	 * @returns {Promise<ITextModel>} A promise that resolves to the text model.
	 */
	async loadModel(): Promise<ITextModel> {
		if (!this.model) {
			this.modelDisposable.value = new DisposableStore();
			this.model = this.modelService.createModel('', this.language, this.modelUri);
			this.outputContentProvider.getContent()
				.then(({ content, consume }) => {
					if (!this.model || !this.modelDisposable.value) {
						return;
					}
					// Initialize model with current content from provider.
					this.doAppendContent(this.model, content);
					consume();
					// Listen for content changes from the provider.
					this.modelDisposable.value.add(this.outputContentProvider.onDidReset(() => this.onDidContentChange(true, true)));
					this.modelDisposable.value.add(this.outputContentProvider.onDidAppend(() => this.onDidContentChange(false, false)));
					this.modelDisposable.value.add(this.outputContentProvider.watch());
				});
			// When the model is disposed, clean up associated resources.
			this.modelDisposable.value.add(this.model.onWillDispose(() => {
				this.outputContentProvider.reset();
				this.modelDisposable.value = undefined;
				this.cancelModelUpdate();
				this.model = null;
			}));
		}
		return this.model;
	}

	/**
	 * @method onDidContentChange
	 * @description Handles content change events from the content provider, triggering a model update.
	 * @param {boolean} reset - Indicates if the content was reset (e.g., cleared).
	 * @param {boolean} appendImmediately - If true, appending is done without delay.
	 */
	private onDidContentChange(reset: boolean, appendImmediately: boolean): void {
		if (reset && !this.modelUpdateInProgress) {
			this.doUpdate(OutputChannelUpdateMode.Clear, true);
		}
		this.doUpdate(OutputChannelUpdateMode.Append, appendImmediately);
	}

	/**
	 * @method doUpdate
	 * @description Central logic for updating the model based on the specified mode.
	 * @param {OutputChannelUpdateMode} mode - The type of update to perform (Clear, Replace, Append).
	 * @param {boolean} immediate - If true, the update is performed without throttling.
	 */
	protected doUpdate(mode: OutputChannelUpdateMode, immediate: boolean): void {
		if (mode === OutputChannelUpdateMode.Clear || mode === OutputChannelUpdateMode.Replace) {
			this.cancelModelUpdate();
		}
		if (!this.model) {
			return;
		}

		this.modelUpdateInProgress = true;
		if (!this.modelUpdateCancellationSource.value) {
			this.modelUpdateCancellationSource.value = new CancellationTokenSource();
		}
		const token = this.modelUpdateCancellationSource.value.token;

		if (mode === OutputChannelUpdateMode.Clear) {
			this.clearContent(this.model);
		}

		else if (mode === OutputChannelUpdateMode.Replace) {
			this.replacePromise = this.replaceContent(this.model, token).finally(() => this.replacePromise = undefined);
		}

		else {
			this.appendContent(this.model, immediate, token);
		}
	}

	/**
	 * @method clearContent
	 * @description Clears all text from the given text model.
	 * @param {ITextModel} model - The model to clear.
	 */
	private clearContent(model: ITextModel): void {
		model.applyEdits([EditOperation.delete(model.getFullModelRange())]);
		this.modelUpdateInProgress = false;
	}

	/**
	 * @method appendContent
	 * @description Appends new content to the model, with throttling to batch updates.
	 * @param {ITextModel} model - The model to append to.
	 * @param {boolean} immediate - If true, appends immediately without delay.
	 * @param {CancellationToken} token - A cancellation token to abort the operation.
	 */
	private appendContent(model: ITextModel, immediate: boolean, token: CancellationToken): void {
		this.appendThrottler.trigger(async () => {
			/* Abort if operation is cancelled */
			if (token.isCancellationRequested) {
				return;
			}

			/* Wait for replace to finish */
			if (this.replacePromise) {
				try { await this.replacePromise; } catch (e) { /* Ignore */ }
				/* Abort if operation is cancelled */
				if (token.isCancellationRequested) {
					return;
				}
			}

			/* Get content to append */
			const { content, consume } = await this.outputContentProvider.getContent();
			/* Abort if operation is cancelled */
			if (token.isCancellationRequested) {
				return;
			}

			/* Appned Content */
			this.doAppendContent(model, content);
			consume();
			this.modelUpdateInProgress = false;
		}, immediate ? 0 : undefined).catch(error => {
			if (!isCancellationError(error)) {
				throw error;
			}
		});
	}

	/**
	 * @method doAppendContent
	 * @description Performs the actual append operation on the text model.
	 * @param {ITextModel} model - The model to append to.
	 * @param {string} content - The content to append.
	 */
	private doAppendContent(model: ITextModel, content: string): void {
		const lastLine = model.getLineCount();
		const lastLineMaxColumn = model.getLineMaxColumn(lastLine);
		model.applyEdits([EditOperation.insert(new Position(lastLine, lastLineMaxColumn), content)]);
	}

	/**
	 * @method replaceContent
	 * @description Replaces the model's content with new content, calculating minimal edits for efficiency.
	 * @param {ITextModel} model - The model to replace content in.
	 * @param {CancellationToken} token - A cancellation token to abort the operation.
	 */
	private async replaceContent(model: ITextModel, token: CancellationToken): Promise<void> {
		/* Get content to replace */
		const { content, consume } = await this.outputContentProvider.getContent();
		/* Abort if operation is cancelled */
		if (token.isCancellationRequested) {
			return;
		}

		/* Compute Edits */
		const edits = await this.getReplaceEdits(model, content.toString());
		/* Abort if operation is cancelled */
		if (token.isCancellationRequested) {
			return;
		}

		if (edits.length) {
			/* Apply Edits */
			model.applyEdits(edits);
		}
		consume();
		this.modelUpdateInProgress = false;
	}

	/**
	 * @method getReplaceEdits
	 * @description Computes the minimal edits required to transition the model's content to the new content.
	 * @param {ITextModel} model - The current text model.
	 * @param {string} contentToReplace - The new content.
	 * @returns {Promise<ISingleEditOperation[]>} A promise that resolves to an array of edits.
	 */
	private async getReplaceEdits(model: ITextModel, contentToReplace: string): Promise<ISingleEditOperation[]> {
		if (!contentToReplace) {
			return [EditOperation.delete(model.getFullModelRange())];
		}
		if (contentToReplace !== model.getValue()) {
			// Use the editor worker service to compute minimal edits for performance.
			const edits = await this.editorWorkerService.computeMoreMinimalEdits(model.uri, [{ text: contentToReplace.toString(), range: model.getFullModelRange() }]);
			if (edits?.length) {
				return edits.map(edit => EditOperation.replace(Range.lift(edit.range), edit.text));
			}
		}
		return [];
	}

	/**
	 * @method cancelModelUpdate
	 * @description Cancels any in-progress model updates.
	 */
	protected cancelModelUpdate(): void {
		this.modelUpdateCancellationSource.value?.cancel();
		this.modelUpdateCancellationSource.value = undefined;
		this.appendThrottler.cancel();
		this.replacePromise = undefined;
		this.modelUpdateInProgress = false;
	}

	/**
	 * @method isVisible
	 * @description Checks if the output channel model is currently visible (i.e., has an active text model).
	 * @returns {boolean} True if the model is visible, false otherwise.
	 */
	protected isVisible(): boolean {
		return !!this.model;
	}

	override dispose(): void {
		this._onDispose.fire();
		super.dispose();
	}

	append(message: string): void { throw new Error('Not supported'); }
	replace(message: string): void { throw new Error('Not supported'); }

	abstract clear(): void;
	abstract update(mode: OutputChannelUpdateMode, till: number | undefined, immediate: boolean): void;
}

/**
 * @class FileOutputChannelModel
 * @description A concrete implementation of an output channel model that is directly backed by a file content provider.
 */
export class FileOutputChannelModel extends AbstractFileOutputChannelModel implements IOutputChannelModel {

	private readonly fileOutput: FileContentProvider;

	constructor(
		modelUri: URI,
		language: ILanguageSelection,
		file: URI,
		@IFileService fileService: IFileService,
		@IModelService modelService: IModelService,
		@ILogService logService: ILogService,
		@IEditorWorkerService editorWorkerService: IEditorWorkerService,
	) {
		const fileOutput = new FileContentProvider(file, fileService, logService);
		super(modelUri, language, fileOutput, modelService, editorWorkerService);
		this.fileOutput = this._register(fileOutput);
	}

	/**
	 * @method clear
	 * @description Clears the content of the output channel by triggering a clear update.
	 */
	override clear(): void {
		this.update(OutputChannelUpdateMode.Clear, undefined, true);
	}

	/**
	 * @method update
	 * @description Overrides the base update to handle resetting the file provider's position.
	 */
	override update(mode: OutputChannelUpdateMode, till: number | undefined, immediate: boolean): void {
		if (mode === OutputChannelUpdateMode.Clear || mode === OutputChannelUpdateMode.Replace) {
			if (isNumber(till)) {
				this.fileOutput.reset(till);
			} else {
				this.fileOutput.resetToEnd();
			}
		}
		this.doUpdate(mode, immediate);
	}

}

/**
 * @class OutputChannelBackedByFile
 * @description An output channel model that writes messages to a file via a logger and also displays them in the editor.
 */
class OutputChannelBackedByFile extends FileOutputChannelModel implements IOutputChannelModel {

	private logger: ILogger;
	private _offset: number;

	constructor(
		id: string,
		modelUri: URI,
		language: ILanguageSelection,
		file: URI,
		@IFileService fileService: IFileService,
		@IModelService modelService: IModelService,
		@ILoggerService loggerService: ILoggerService,
		@ILogService logService: ILogService,
		@IEditorWorkerService editorWorkerService: IEditorWorkerService
	) {
		super(modelUri, language, file, fileService, modelService, logService, editorWorkerService);

		// Use a logger to write to the file without rotation to ensure content stability.
		this.logger = loggerService.createLogger(file, { logLevel: 'always', donotRotate: true, donotUseFormatters: true, hidden: true });
		this._offset = 0;
	}

	/**
	 * @method append
	 * @description Appends a message to the output file and triggers a model update.
	 * @param {string} message - The message to append.
	 */
	override append(message: string): void {
		this.write(message);
		this.update(OutputChannelUpdateMode.Append, undefined, this.isVisible());
	}

	/**
	 * @method replace
	 * @description Replaces the content of the output file with a new message and triggers a model update.
	 * @param {string} message - The new content.
	 */
	override replace(message: string): void {
		const till = this._offset;
		this.write(message);
		this.update(OutputChannelUpdateMode.Replace, till, true);
	}

	/**
	 * @method write
	 * @description Writes content to the log file and updates the internal offset.
	 * @param {string} content - The content to write.
	 */
	private write(content: string): void {
		this._offset += VSBuffer.fromString(content).byteLength;
		this.logger.info(content);
		if (this.isVisible()) {
			this.logger.flush();
		}
	}

}

/**
 * @class DelegatedOutputChannelModel
 * @description A wrapper model that delegates to an underlying `IOutputChannelModel` which is created
 * asynchronously. This is useful when the creation of the output channel depends on an asynchronous
 * operation, such as resolving an output directory.
 */
export class DelegatedOutputChannelModel extends Disposable implements IOutputChannelModel {

	private readonly _onDispose: Emitter<void> = this._register(new Emitter<void>());
	readonly onDispose: Event<void> = this._onDispose.event;

	private readonly outputChannelModel: Promise<IOutputChannelModel>;

	constructor(
		id: string,
		modelUri: URI,
		language: ILanguageSelection,
		outputDir: Promise<URI>,
		@IInstantiationService private readonly instantiationService: IInstantiationService,
		@IFileService private readonly fileService: IFileService,
	) {
		super();
		this.outputChannelModel = this.createOutputChannelModel(id, modelUri, language, outputDir);
	}

	/**
	 * @method createOutputChannelModel
	 * @description Asynchronously creates the underlying output channel model. This involves resolving the output directory,
	 * creating the log file, and instantiating the channel model.
	 * @returns {Promise<IOutputChannelModel>} A promise that resolves to the created output channel model.
	 */
	private async createOutputChannelModel(id: string, modelUri: URI, language: ILanguageSelection, outputDirPromise: Promise<URI>): Promise<IOutputChannelModel> {
		const outputDir = await outputDirPromise;
		const file = resources.joinPath(outputDir, `${id.replace(/[\\/:\*\?"<>\|]/g, '')}.log`);
		await this.fileService.createFile(file);
		const outputChannelModel = this._register(this.instantiationService.createInstance(OutputChannelBackedByFile, id, modelUri, language, file));
		this._register(outputChannelModel.onDispose(() => this._onDispose.fire()));
		return outputChannelModel;
	}

	/**
	 * @method append
	 * @description Delegates the append operation to the underlying model.
	 */
	append(output: string): void {
		this.outputChannelModel.then(outputChannelModel => outputChannelModel.append(output));
	}

	/**
	 * @method update
	 * @description Delegates the update operation to the underlying model.
	 */
	update(mode: OutputChannelUpdateMode, till: number | undefined, immediate: boolean): void {
		this.outputChannelModel.then(outputChannelModel => outputChannelModel.update(mode, till, immediate));
	}

	/**
	 * @method loadModel
	 * @description Delegates the loadModel operation to the underlying model.
	 */
	loadModel(): Promise<ITextModel> {
		return this.outputChannelModel.then(outputChannelModel => outputChannelModel.loadModel());
	}

	/**
	 * @method clear
	 * @description Delegates the clear operation to the underlying model.
	 */
	clear(): void {
		this.outputChannelModel.then(outputChannelModel => outputChannelModel.clear());
	}

	/**
	 * @method replace
	 * @description Delegates the replace operation to the underlying model.
	 */
	replace(value: string): void {
		this.outputChannelModel.then(outputChannelModel => outputChannelModel.replace(value));
	}
}
