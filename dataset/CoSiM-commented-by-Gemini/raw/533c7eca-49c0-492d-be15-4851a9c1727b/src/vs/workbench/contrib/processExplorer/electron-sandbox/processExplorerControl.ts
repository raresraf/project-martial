/**
 * @file processExplorerControl.ts
 * @brief UI controller for the VS Code Process Explorer view.
 * 
 * Architectural Intent: Implements a tree-based visualization of the VS Code process 
 * hierarchy, including local renderer/extension hosts and remote server processes. 
 * Facilitates diagnostic actions such as process termination and debugger attachment.
 * 
 * Performance Optimization: Uses a Delayer for periodic refreshes to maintain UI 
 * responsiveness without saturating the IPC channel with process table updates.
 */

import './media/processExplorer.css';
import { localize } from '../../../../nls.js';
import { INativeHostService } from '../../../../platform/native/common/native.js';
import { $, append, Dimension, getDocument } from '../../../../base/browser/dom.js';
import { StandardKeyboardEvent } from '../../../../base/browser/keyboardEvent.js';
import { IIdentityProvider, IListVirtualDelegate } from '../../../../base/browser/ui/list/list.js';
import { IDataSource, ITreeRenderer, ITreeNode, ITreeContextMenuEvent } from '../../../../base/browser/ui/tree/tree.js';
import { ProcessItem } from '../../../../base/common/processes.js';
import { IRemoteDiagnosticError, isRemoteDiagnosticError } from '../../../../platform/diagnostics/common/diagnostics.js';
import { ByteSize } from '../../../../platform/files/common/files.js';
import { KeyCode } from '../../../../base/common/keyCodes.js';
import { Disposable } from '../../../../base/common/lifecycle.js';
import { WorkbenchDataTree } from '../../../../platform/list/browser/listService.js';
import { IInstantiationService } from '../../../../platform/instantiation/common/instantiation.js';
import { IListAccessibilityProvider } from '../../../../base/browser/ui/list/listWidget.js';
import { IProductService } from '../../../../platform/product/common/productService.js';
import { IAction, Separator, toAction } from '../../../../base/common/actions.js';
import { IContextMenuService } from '../../../../platform/contextview/browser/contextView.js';
import { coalesce } from '../../../../base/common/arrays.js';
import { ICommandService } from '../../../../platform/commands/common/commands.js';
import { RenderIndentGuides } from '../../../../base/browser/ui/tree/abstractTree.js';
import { isWindows } from '../../../../base/common/platform.js';
import { IProcessService } from '../../../../platform/process/common/process.js';
import { Delayer } from '../../../../base/common/async.js';
import { IHoverService } from '../../../../platform/hover/browser/hover.js';
import { IManagedHover } from '../../../../base/browser/ui/hover/hover.js';
import { getDefaultHoverDelegate } from '../../../../base/browser/ui/hover/hoverDelegateFactory.js';

const DEBUG_FLAGS_PATTERN = /\s--inspect(?:-brk|port)?=(?<port>\d+)?/;
const DEBUG_PORT_PATTERN = /\s--inspect-port=(?<port>\d+)/;

//#region --- process explorer tree

interface IProcessTree {
	readonly processes: IProcessInformation;
}

interface IProcessInformation {
	readonly processRoots: IMachineProcessInformation[];
}

interface IMachineProcessInformation {
	readonly name: string;
	readonly rootProcess: ProcessItem | IRemoteDiagnosticError;
}

/**
 * @class ProcessListDelegate
 * @brief Manages tree item height and template mapping for different diagnostic node types.
 */
class ProcessListDelegate implements IListVirtualDelegate<IMachineProcessInformation | ProcessItem | IRemoteDiagnosticError> {

	getHeight() {
		return 22;
	}

	getTemplateId(element: IProcessInformation | IMachineProcessInformation | ProcessItem | IRemoteDiagnosticError) {
		if (isProcessItem(element)) {
			return 'process';
		}

		if (isMachineProcessInformation(element)) {
			return 'machine';
		}

		if (isRemoteDiagnosticError(element)) {
			return 'error';
		}

		if (isProcessInformation(element)) {
			return 'header';
		}

		return '';
	}
}

/**
 * @class ProcessTreeDataSource
 * @brief Provider for the recursive process tree structure.
 */
class ProcessTreeDataSource implements IDataSource<IProcessTree, IProcessInformation | IMachineProcessInformation | ProcessItem | IRemoteDiagnosticError> {

	/**
	 * Block Logic: Branch node detection.
	 * Invariant: Errors are terminal leaves; processes are branches if they have child subprocesses.
	 */
	hasChildren(element: IProcessTree | IProcessInformation | IMachineProcessInformation | ProcessItem | IRemoteDiagnosticError): boolean {
		if (isRemoteDiagnosticError(element)) {
			return false;
		}

		if (isProcessItem(element)) {
			return !!element.children?.length;
		}

		return true;
	}

	/**
	 * Block Logic: Child resolution.
	 * Invariant: Handles machine boundaries by resolving to the root process of each detected host.
	 */
	getChildren(element: IProcessTree | IProcessInformation | IMachineProcessInformation | ProcessItem | IRemoteDiagnosticError) {
		if (isProcessItem(element)) {
			return element.children ?? [];
		}

		if (isRemoteDiagnosticError(element)) {
			return [];
		}

		if (isProcessInformation(element)) {
			if (element.processRoots.length > 1) {
				return element.processRoots; 
			}

			if (element.processRoots.length > 0) {
				return [element.processRoots[0].rootProcess];
			}

			return [];
		}

		if (isMachineProcessInformation(element)) {
			return [element.rootProcess];
		}

		return element.processes ? [element.processes] : [];
	}
}

/**
 * @class ProcessRenderer
 * @brief Maps ProcessItem data to DOM elements with live telemetry (CPU/Memory).
 */
class ProcessRenderer implements ITreeRenderer<ProcessItem, void, IProcessItemTemplateData> {

	readonly templateId: string = 'process';

	constructor(
		private totalMem: number,
		private model: ProcessExplorerModel,
		@IHoverService private readonly hoverService: IHoverService
	) { }

	renderTemplate(container: HTMLElement): IProcessItemTemplateData {
		const row = createRow(container);

		return {
			name: row.name,
			cpu: row.cpu,
			memory: row.memory,
			pid: row.pid,
			hover: new ProcessItemHover(row.name, this.hoverService)
		};
	}

	/**
	 * Block Logic: Telemetry rendering.
	 * Logic: Scales relative memory percentage to absolute megabytes based on detected system statistics.
	 */
	renderElement(node: ITreeNode<ProcessItem, void>, index: number, templateData: IProcessItemTemplateData, height: number | undefined): void {
		const { element } = node;

		const pid = element.pid.toFixed(0);

		templateData.name.textContent = this.model.getName(element.pid, element.name);
		templateData.cpu.textContent = element.load.toFixed(0);
		templateData.pid.textContent = pid;
		templateData.pid.parentElement!.id = `pid-${pid}`;

		templateData.hover?.update(element.cmd);

		const memory = isWindows ? element.mem : (this.totalMem * (element.mem / 100));
		templateData.memory.textContent = (memory / ByteSize.MB).toFixed(0);
	}

	disposeTemplate(templateData: IProcessItemTemplateData): void {
		templateData.hover?.dispose();
	}
}

//#endregion

/**
 * @class ProcessExplorerControl
 * @brief Orchestrates the lifecycle and interactions of the process explorer UI.
 */
export class ProcessExplorerControl extends Disposable {

	private dimensions: Dimension | undefined = undefined;

	private readonly model: ProcessExplorerModel;
	private tree: WorkbenchDataTree<IProcessTree, IProcessTree | IMachineProcessInformation | ProcessItem | IProcessInformation | IRemoteDiagnosticError> | undefined;

	private readonly delayer = this._register(new Delayer(1000));

	constructor(
		container: HTMLElement,
		@INativeHostService private readonly nativeHostService: INativeHostService,
		@IInstantiationService private readonly instantiationService: IInstantiationService,
		@IProductService private readonly productService: IProductService,
		@IContextMenuService private readonly contextMenuService: IContextMenuService,
		@ICommandService private readonly commandService: ICommandService,
		@IProcessService private readonly processService: IProcessService
	) {
		super();

		this.model = new ProcessExplorerModel(this.productService);
		this.create(container);
	}

	private async create(container: HTMLElement): Promise<void> {
		const { totalmem } = await this.nativeHostService.getOSStatistics();
		this.createProcessTree(container, totalmem);

		this.update();
	}

	private createProcessTree(container: HTMLElement, totalmem: number): void {
		container.classList.add('process-explorer');
		container.id = 'process-explorer';

		const renderers = [
			this.instantiationService.createInstance(ProcessRenderer, totalmem, this.model),
			new ProcessHeaderTreeRenderer(),
			new MachineRenderer(),
			new ErrorRenderer()
		];

		this.tree = this._register(this.instantiationService.createInstance(
			WorkbenchDataTree<IProcessTree, IProcessTree | IMachineProcessInformation | ProcessItem | IProcessInformation | IRemoteDiagnosticError>,
			'processExplorer',
			container,
			new ProcessListDelegate(),
			renderers,
			new ProcessTreeDataSource(),
			{
				accessibilityProvider: new ProcessAccessibilityProvider(),
				identityProvider: new ProcessIdentityProvider(),
				expandOnlyOnTwistieClick: true,
				renderIndentGuides: RenderIndentGuides.OnHover
			}));

		this._register(this.tree.onKeyDown(e => this.onTreeKeyDown(e)));
		this._register(this.tree.onContextMenu(e => this.onTreeContextMenu(container, e)));

		this.tree.setInput(this.model);
		this.layoutTree();
	}

	/**
	 * Block Logic: Keyboard shortcuts.
	 * Invariant: Alt+E triggers SIGTERM for the selected process set.
	 */
	private async onTreeKeyDown(e: KeyboardEvent): Promise<void> {
		const event = new StandardKeyboardEvent(e);
		if (event.keyCode === KeyCode.KeyE && event.altKey) {
			const selectionPids = this.getSelectedPids();
			await Promise.all(selectionPids.map(pid => this.nativeHostService.killProcess(pid, 'SIGTERM')));
		}
	}

	/**
	 * Block Logic: Context menu resolution.
	 * Logic: Dynamically generates actions based on the selected process (Kill, Force Kill, Copy, Debug).
	 */
	private onTreeContextMenu(container: HTMLElement, e: ITreeContextMenuEvent<IProcessTree | IMachineProcessInformation | ProcessItem | IProcessInformation | IRemoteDiagnosticError | null>): void {
		if (!isProcessItem(e.element)) {
			return;
		}

		const item = e.element;
		const pid = Number(item.pid);

		const actions: IAction[] = [];

		actions.push(toAction({ id: 'killProcess', label: localize('killProcess', "Kill Process"), run: () => this.nativeHostService.killProcess(pid, 'SIGTERM') }));
		actions.push(toAction({ id: 'forceKillProcess', label: localize('forceKillProcess', "Force Kill Process"), run: () => this.nativeHostService.killProcess(pid, 'SIGKILL') }));

		actions.push(new Separator());

		actions.push(toAction({
			id: 'copy',
			label: localize('copy', "Copy"),
			run: () => {
				const selectionPids = this.getSelectedPids();

				if (!selectionPids?.includes(pid)) {
					selectionPids.length = 0; 
					selectionPids.push(pid);
				}

				const rows = selectionPids?.map(e => getDocument(container).getElementById(`pid-${e}`)).filter(e => !!e);
				if (rows) {
					const text = rows.map(e => e.innerText).filter(e => !!e);
					this.nativeHostService.writeClipboardText(text.join('\n'));
				}
			}
		}));

		actions.push(toAction({
			id: 'copyAll',
			label: localize('copyAll', "Copy All"),
			run: () => {
				const processList = getDocument(container).getElementById('process-explorer');
				if (processList) {
					this.nativeHostService.writeClipboardText(processList.innerText);
				}
			}
		}));

		// Functional Utility: Enables debugger attachment if the process command line indicates a Node.js target.
		if (this.isDebuggable(item.cmd)) {
			actions.push(new Separator());
			actions.push(toAction({ id: 'debug', label: localize('debug', "Debug"), run: () => this.attachTo(item) }));
		}

		this.contextMenuService.showContextMenu({
			getAnchor: () => e.anchor,
			getActions: () => actions
		});
	}

	private isDebuggable(cmd: string): boolean {
		const matches = DEBUG_FLAGS_PATTERN.exec(cmd);

		return (matches && matches.groups!.port !== '0') || cmd.indexOf('node ') >= 0 || cmd.indexOf('node.exe') >= 0;
	}

	/**
	 * Block Logic: Debugger attachment.
	 * Logic: Extracts the inspect port from flags or falls back to PID-based signal attachment.
	 */
	private attachTo(item: ProcessItem): void {
		const config: { type: string; request: string; name: string; port?: number; processId?: string } = {
			type: 'node',
			request: 'attach',
			name: `process ${item.pid}`
		};

		let matches = DEBUG_FLAGS_PATTERN.exec(item.cmd);
		if (matches) {
			config.port = Number(matches.groups!.port);
		} else {
			config.processId = String(item.pid); 
		}

		matches = DEBUG_PORT_PATTERN.exec(item.cmd);
		if (matches) {
			config.port = Number(matches.groups!.port); 
		}

		this.commandService.executeCommand('debug.startFromConfig', config);
	}

	private getSelectedPids(): number[] {
		return coalesce(this.tree?.getSelection()?.map(e => {
			if (!isProcessItem(e)) {
				return undefined;
			}

			return e.pid;
		}) ?? []);
	}

	/**
	 * Block Logic: Polling loop.
	 * Invariant: Re-triggers the diagnostic resolution every 1 second via the Delayer.
	 */
	private async update(): Promise<void> {
		const { processes, pidToNames } = await this.processService.resolveProcesses();

		this.model.update(processes, pidToNames);

		this.tree?.updateChildren();
		this.layoutTree();

		this.delayer.trigger(() => this.update());
	}

	focus(): void {
		this.tree?.domFocus();
	}

	layout(dimension: Dimension): void {
		this.dimensions = dimension;

		this.layoutTree();
	}

	private layoutTree(): void {
		if (this.dimensions && this.tree) {
			this.tree.layout(this.dimensions.height, this.dimensions.width);
		}
	}
}

/**
 * @class ProcessExplorerModel
 * @brief Encapsulates the state and data mapping for the process explorer.
 */
class ProcessExplorerModel implements IProcessTree {

	processes: IProcessInformation = { processRoots: [] };

	private readonly mapPidToName = new Map<number, string>();

	constructor(@IProductService private productService: IProductService) { }

	/**
	 * @brief Updates the model with fresh diagnostic data.
	 * Invariant: Normalizes root process names based on host machine context.
	 */
	update(processRoots: IMachineProcessInformation[], pidToNames: [number, string][]): void {

		this.mapPidToName.clear();

		for (const [pid, name] of pidToNames) {
			this.mapPidToName.set(pid, name);
		}

		processRoots.forEach((info, index) => {
			if (isProcessItem(info.rootProcess)) {
				info.rootProcess.name = index === 0 ? this.productService.applicationName : 'remote-server';
			}
		});

		this.processes = { processRoots };
	}

	getName(pid: number, fallback: string): string {
		return this.mapPidToName.get(pid) ?? fallback;
	}
}
