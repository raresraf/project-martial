import { common } from '@kit.AbilityKit';
import display from '@ohos.display';
import deviceInfo from '@ohos.deviceInfo';
import promptAction from '@ohos.promptAction';

/**
 * @647a6674-3097-4b33-99e4-2e7d29920a02/support/openharmony/entry/src/main/ets/pages/Index.ets
 * @brief Main entry page for the Servo browser shell on OpenHarmony.
 * 
 * Functional Intent: Provides the top-level UI and native bridge orchestration for 
 * the Servo engine. It manages the browser navigation bar, handles back/forward 
 * requests, and initializes the native XComponent where the Servo engine 
 * renders web content.
 * 
 * Domain: Browser Shell, OpenHarmony ArkUI, Native Interop.
 */

/**
 * @brief Bridge interface between ArkTS and the native Servo implementation (C++).
 * 
 * Logic: Defines the set of operations exposed by the 'servoshell' native library 
 * via the XComponent context.
 */
interface ServoXComponentInterface {
  loadURL(url: string): void;
  goBack(): void;
  goForward(): void;
  registerURLcallback(callback: (url: string) => void): void;
  registerTerminateCallback(callback: () => void): void;
  registerPromptToastCallback(callback: (msg: string) => void): void
  initServo(options: InitOpts): void;
}

/**
 * @brief Configuration parameters for initializing the Servo engine.
 */
interface InitOpts {
  url: string;
  deviceType: string,
  osFullName: string,
  resourceDir: string,
  cacheDir: string,
  displayDensity: number,
  commandlineArgs: string,
}

/**
 * @brief Calculates the screen density scale factor.
 * Logic: Converts DPI to a relative multiplier based on the standard 160 DPI. 
 * Falls back to 3.0 (high-density) on hardware retrieval failure.
 */
function get_density(): number {
    try {
      let displayClass = display.getDefaultDisplaySync();
      console.info('Test densityDPI:' + JSON.stringify(displayClass.densityDPI));
      return displayClass.densityDPI / 160;
  } catch (exception) {
      console.error('Failed to obtain the default display object. Code: ' + JSON.stringify(exception));
      return 3;
  }
}

/**
 * @brief Retrieves the hardware device classification string.
 */
function get_device_type(): string {
  let device_type: string = deviceInfo.deviceType;
  if (device_type == "") {
      console.error("deviceInfo.deviceType is empty string!")
  } else {
    console.info("Device type is " + device_type)
  }
  return device_type;
}

/**
 * @brief Displays a transient toast notification to the user.
 */
function prompt_toast(msg: string) {
    promptAction.showToast({
      message: msg,
      duration: 2000
    });
}

// Global Storage: Obtains the shared LocalStorage instance for the Ability stage.
let storage = LocalStorage.getShared()

/**
 * @brief Primary UI component for the browser shell.
 * 
 * Functional Intent: Orchestrates the visual layout and lifecycle events 
 * (loading, hardware key presses) for the Servo-based web view.
 */
@Entry(storage)
@Component
struct Index {
  xComponentContext: ServoXComponentInterface | undefined = undefined;
  xComponentAttrs: XComponentAttrs = {
    id: 'ServoDemo',
    type: XComponentType.SURFACE,
    libraryname: 'servoshell',
  }

  private context = getContext(this) as common.UIAbilityContext;
  @LocalStorageProp('InitialURI') InitialURI: string = "unused"
  @LocalStorageProp('CommandlineArgs') CommandlineArgs: string = ""
  @State urlToLoad: string = this.InitialURI

  /**
   * Block Logic: Intercepts system back-navigation events.
   * Logic: Overrides default app-backgrounding behavior by forwarding the 
   * request to the Servo engine's history manager.
   */
  onBackPress(): boolean | void {
    this.xComponentContext?.goBack()
    return true;
  }

  /**
   * build - Reactive layout definition.
   * Logic: Uses a Flex-column layout to stack the navigation row and the 
   * primary XComponent view. Configures bidirectional data binding for 
   * the URL input field.
   */
  build() {
    Flex({ direction: FlexDirection.Column}) {
      Row() {
        // Block Logic: Navigation Control Bar.
        Button('⇦').backgroundColor(Color.White)
          .fontColor(Color.Black)
          .fontWeight(FontWeight.Bolder)
          .width('12%')
          .fontSize(12)
          .onClick(()=>{
            this.onBackPress()
          })
        Button('⇨').backgroundColor(Color.White)
          .fontColor(Color.Black)
          .fontWeight(FontWeight.Bolder)
          .fontSize(12)
          .width('12%')
          .onClick(()=> {
            this.xComponentContext?.goForward()
          })
        TextInput({placeholder:'URL',text: $$this.urlToLoad})
          .type(InputType.Normal)
          .width('76%')
          .onChange((value) => {
            this.urlToLoad = value
          })
          .onSubmit((EnterKeyType)=>{
            this.xComponentContext?.loadURL(this.urlToLoad)
            console.info('Load URL: ', this.urlToLoad)
          })
      }

      /**
       * Block Logic: Servo Engine Viewport (Native Surface).
       * Logic: 
       * 1. Loads the 'servoshell' native library into a SURFACE component.
       * 2. On Load: Captures the native context and triggers Servo initialization 
       *    with hardware metadata (DPI, directories, CLI args).
       * 3. Registers callbacks for URL updates and UI events originating from native code.
       */
      XComponent(this.xComponentAttrs)
        .focusable(true)
        .onLoad((xComponentContext) => {
          this.xComponentContext = xComponentContext as ServoXComponentInterface;
          let resource_dir: string = this.context.resourceDir;
          let cache_dir: string = this.context.cacheDir;
          
          let init_options: InitOpts = {
            url: this.urlToLoad,
            deviceType: get_device_type(),
            osFullName: deviceInfo.osFullName,
            displayDensity: get_density(),
            resourceDir: resource_dir,
            cacheDir: cache_dir,
            commandlineArgs: this.CommandlineArgs
          }
          
          // Functional Utility: Finalizes engine setup and synchronizes state.
          this.xComponentContext.initServo(init_options)
          this.xComponentContext.registerURLcallback((new_url) => {
            console.info('New URL from native: ', new_url)
            this.urlToLoad = new_url
          })
          this.xComponentContext.registerTerminateCallback(() => { this.context?.terminateSelf(); })
          this.xComponentContext.registerPromptToastCallback(prompt_toast)
        })
    }
    .width('100%')
  }
}

/**
 * @brief Structural metadata for the native component.
 */
interface XComponentAttrs {
  id: string;
  type: number;
  libraryname: string;
}
