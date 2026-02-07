/*
 * @file swapchain.rs
 * @brief Manages WebGPU swapchain presentation buffers and their integration with the WebRender external image API.
 *        This module handles the lifecycle of presentation buffers, facilitating efficient data transfer
 *        from WebGPU rendered frames to the compositor for display. It ensures proper synchronization
 *        and resource management for graphics presentation.
 */

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::collections::HashMap;
use std::ptr::NonNull;
use std::slice;
use std::sync::{Arc, Mutex};

use arrayvec::ArrayVec;
use compositing_traits::{
    CrossProcessCompositorApi, SerializableImageData, WebrenderExternalImageApi,
    WebrenderImageSource,
};
use euclid::default::Size2D;
use ipc_channel::ipc::IpcSender;
use log::{error, warn};
use pixels::{IpcSnapshot, Snapshot, SnapshotAlphaMode, SnapshotPixelFormat};
use serde::{Deserialize, Serialize};
use webgpu_traits::{
    ContextConfiguration, Error,
    /// @brief Defines the number of presentation buffers used in the swap chain for a WebGPU context.
    /// This constant determines the depth of the buffer queue, influencing rendering latency and smoothness.
    PRESENTATION_BUFFER_COUNT, WebGPUContextId, WebGPUMsg,
};
use webrender_api::units::DeviceIntSize;
use webrender_api::{
    ExternalImageData, ExternalImageId, ExternalImageType, ImageDescriptor, ImageDescriptorFlags,
    ImageFormat, ImageKey,
};
use wgpu_core::device::HostMap;
use wgpu_core::global::Global;
use wgpu_core::id;
use wgpu_core::resource::{BufferAccessError, BufferMapOperation};

use crate::wgt;

/// @brief The default image format used for WebGPU contexts when no specific format is provided.
/// This typically corresponds to an 8-bit RGBA unorm format.
const DEFAULT_IMAGE_FORMAT: ImageFormat = ImageFormat::RGBA8;

pub type WGPUImageMap = Arc<Mutex<HashMap<WebGPUContextId, ContextData>>>;

/// Presentation id encodes current configuration and current image
/// so that async presentation does not update context with older data
/// @brief A unique identifier for a specific presentation configuration and rendered image.
/// This ID is crucial for ensuring that asynchronous presentation updates do not
/// inadvertently apply older data to a newer context, maintaining frame integrity.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
struct PresentationId(u64);

/// @brief Represents a GPU-allocated buffer used for presentation, holding a mapped range
/// of data from the GPU. It provides safe access to the buffer's contents and
/// ensures proper unmapping when dropped.
struct GPUPresentationBuffer {
    global: Arc<Global>,
    buffer_id: id::BufferId,
    data: NonNull<u8>,
    size: usize,
}

// This is safe because `GPUPresentationBuffer` holds exclusive access to ptr
unsafe impl Send for GPUPresentationBuffer {}
unsafe impl Sync for GPUPresentationBuffer {}

impl GPUPresentationBuffer {
    /// @brief Creates a new `GPUPresentationBuffer` by mapping a specified range of a WGPU buffer.
    /// @param global A shared reference to the WGPU `Global` object, providing access to WGPU's global state.
    /// @param buffer_id The identifier of the WGPU buffer to be mapped.
    /// @param buffer_size The size of the buffer in bytes.
    /// @return A new `GPUPresentationBuffer` instance with the mapped data.
    fn new(global: Arc<Global>, buffer_id: id::BufferId, buffer_size: u64) -> Self {
        let (data, size) = global
            .buffer_get_mapped_range(buffer_id, 0, Some(buffer_size))
            .unwrap();
        GPUPresentationBuffer {
            global,
            buffer_id,
            data,
            size: size as usize,
        }
    }

    /// @brief Provides a read-only slice of the mapped buffer's content.
    /// @return A slice (`&[u8]`) representing the data within the mapped buffer.
    fn slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.data.as_ptr(), self.size) }
    }
}

impl Drop for GPUPresentationBuffer {
    /// @brief Unmaps the WGPU buffer when `GPUPresentationBuffer` is dropped, releasing GPU resources.
    fn drop(&mut self) {
        let _ = self.global.buffer_unmap(self.buffer_id);
    }
}

/// @brief Manages a collection of WebGPU external images, providing an interface
/// for WebRender to lock and unlock these images for compositing.
/// It tracks both the WebGPU contexts and the currently locked image data.
#[derive(Default)]
pub struct WGPUExternalImages {
    pub images: WGPUImageMap,
    pub locked_ids: HashMap<WebGPUContextId, Vec<u8>>,
}

impl WebrenderExternalImageApi for WGPUExternalImages {
    /// @brief Locks an external image identified by `id`, providing its raw data and size.
    /// This method retrieves the image data associated with the given context ID,
    /// converts it to a raw byte slice, and stores a copy to prevent premature deallocation.
    /// @param id The `u64` identifier of the WebGPU context whose image is to be locked.
    /// @return A tuple containing the `WebrenderImageSource` (raw data slice) and the `Size2D` of the image.
    fn lock(&mut self, id: u64) -> (WebrenderImageSource, Size2D<i32>) {
        let id = WebGPUContextId(id);
        let webgpu_contexts = self.images.lock().unwrap();
        let context_data = webgpu_contexts.get(&id).unwrap();
        let size = context_data.image_desc.size().cast_unit();
        let data = if let Some(present_buffer) = context_data
            .swap_chain
            .as_ref()
            .and_then(|swap_chain| swap_chain.data.as_ref())
        {
            present_buffer.slice().to_vec()
        } else {
            context_data.dummy_data()
        };
        // Invariant: The `locked_ids` map temporarily stores a copy of the image data
        // to ensure it remains valid for WebRender's consumption while locked.
        self.locked_ids.insert(id, data);
        (
            WebrenderImageSource::Raw(self.locked_ids.get(&id).unwrap().as_slice()),
            size,
        )
    }

    /// @brief Unlocks an external image, releasing the temporary storage of its data.
    /// @param id The `u64` identifier of the WebGPU context whose image is to be unlocked.
    fn unlock(&mut self, id: u64) {
        let id = WebGPUContextId(id);
        // Pre-condition: The image identified by `id` must have been previously locked
        // and its data stored in `locked_ids`.
        self.locked_ids.remove(&id);
    }
}

/// States of presentation buffer
/// @brief Enumerates the possible states of a presentation buffer throughout its lifecycle.
/// These states help manage the creation, mapping, and availability of buffers
/// used for rendering and display.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
enum PresentationBufferState {
    /// @brief Initial state: the buffer's ID is reserved, but the buffer resource itself
    /// has not yet been created on the GPU.
    #[default]
    Unassigned,
    /// @brief The buffer resource has been created on the GPU and is ready for use
    /// (e.g., for WGPU operations or WebRender compositing).
    Available,
    /// @brief The buffer is currently undergoing an asynchronous mapping operation (`mapAsync`).
    Mapping,
    /// @brief The buffer is actively mapped and its contents are accessible,
    /// typically by WebRender for compositing.
    Mapped,
}

/// @brief Represents an active WebGPU swap chain, linking a device and a queue
/// with an optional presentation buffer. This structure is central to managing
/// the output target for WebGPU rendering operations.
struct SwapChain {
    device_id: id::DeviceId,
    queue_id: id::QueueId,
    data: Option<GPUPresentationBuffer>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// @brief A wrapper around `webrender_api::ImageDescriptor` specifically tailored for WebGPU.
/// It encapsulates the format, size, stride, and flags necessary to describe
/// a WebGPU-backed image for use within the WebRender compositor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WebGPUImageDescriptor(pub ImageDescriptor);

impl WebGPUImageDescriptor {
    /// @brief Creates a new `WebGPUImageDescriptor` with the specified format, size, and opacity.
    /// It calculates the appropriate stride based on the image format and ensures
    /// correct flags for opaque images.
    /// @param format The `ImageFormat` of the image (e.g., RGBA8).
    /// @param size The `DeviceIntSize` (width and height) of the image in device pixels.
    /// @param is_opaque A boolean indicating whether the image is opaque.
    /// @return A new `WebGPUImageDescriptor` instance.
    fn new(format: ImageFormat, size: DeviceIntSize, is_opaque: bool) -> Self {
        let stride = ((size.width * format.bytes_per_pixel()) |
            (wgt::COPY_BYTES_PER_ROW_ALIGNMENT as i32 - 1)) +
            1;
        Self(ImageDescriptor {
            format,
            size,
            stride: Some(stride),
            offset: 0,
            flags: if is_opaque {
                ImageDescriptorFlags::IS_OPAQUE
            } else {
                ImageDescriptorFlags::empty()
            },
        })
    }

    /// @brief Creates a default `WebGPUImageDescriptor` using `DEFAULT_IMAGE_FORMAT` and non-opaque settings.
    /// @param size The `DeviceIntSize` (width and height) of the image.
    /// @return A new `WebGPUImageDescriptor` instance with default settings.
    fn default(size: DeviceIntSize) -> Self {
        Self::new(DEFAULT_IMAGE_FORMAT, size, false)
    }

    /// @brief Updates the current image descriptor with a new one if there are changes.
    /// @param new The new `WebGPUImageDescriptor` to compare against and potentially apply.
    /// @return `true` if the descriptor was updated (i.e., `self.0` was different from `new.0`), `false` otherwise.
    fn update(&mut self, new: Self) -> bool {
        if self.0 != new.0 {
            self.0 = new.0;
            true
        } else {
            false
        }
    }

    /// @brief Retrieves the buffer stride in bytes for the image.
    /// @return The calculated buffer stride.
    /// Pre-condition: `stride` must be set in the internal `ImageDescriptor`.
    fn buffer_stride(&self) -> i32 {
        self.0
            .stride
            .expect("Stride should be set by WebGPUImageDescriptor")
    }

    /// @brief Calculates the total buffer size required for the image data.
    /// This is computed as `buffer_stride * image_height`.
    /// @return The total buffer size as `wgt::BufferAddress`.
    fn buffer_size(&self) -> wgt::BufferAddress {
        (self.buffer_stride() * self.0.size.height) as wgt::BufferAddress
    }

    /// @brief Returns the `DeviceIntSize` (width and height) of the image.
    /// @return The size of the image.
    fn size(&self) -> DeviceIntSize {
        self.0.size
    }
}

pub struct ContextData {
    image_key: ImageKey,
    image_desc: WebGPUImageDescriptor,
    image_data: ExternalImageData,
    buffer_ids: ArrayVec<(id::BufferId, PresentationBufferState), PRESENTATION_BUFFER_COUNT>,
    /// @brief Optional `SwapChain` instance. If `None`, the context is considered dummy,
    /// typically rendering transparent black.
    swap_chain: Option<SwapChain>,
    /// @brief The next `PresentationId` to be issued for a new frame.
    next_presentation_id: PresentationId,
    /// @brief The `PresentationId` of the currently presented or configured frame.
    /// This value is monotonically increasing.
    current_presentation_id: PresentationId,
}

impl ContextData {
    /// @brief Initializes a new `ContextData` instance as a dummy (transparent black) context.
    /// This is used when a WebGPU context is created but not yet fully configured with a swapchain.
    /// @param context_id The `WebGPUContextId` for this context.
    /// @param image_key The `ImageKey` associated with this context for WebRender.
    /// @param size The `DeviceIntSize` (width and height) of the context's target surface.
    /// @param buffer_ids An `ArrayVec` of `id::BufferId`s representing pre-reserved buffer identifiers.
    /// @return A new `ContextData` instance.
    fn new(
        context_id: WebGPUContextId,
        image_key: ImageKey,
        size: DeviceIntSize,
        buffer_ids: ArrayVec<id::BufferId, PRESENTATION_BUFFER_COUNT>,
    ) -> Self {
        let image_data = ExternalImageData {
            id: ExternalImageId(context_id.0),
            channel_index: 0,
            image_type: ExternalImageType::Buffer,
            normalized_uvs: false,
        };

        Self {
            image_key,
            image_desc: WebGPUImageDescriptor::default(size),
            image_data,
            swap_chain: None,
            buffer_ids: buffer_ids
                .iter()
                .map(|&buffer_id| (buffer_id, PresentationBufferState::Unassigned))
                .collect(),
            current_presentation_id: PresentationId(0),
            next_presentation_id: PresentationId(1),
        }
    }

    /// @brief Generates a vector of bytes representing a transparent black image.
    /// This is used for dummy contexts or when no actual image data is available.
    /// @return A `Vec<u8>` filled with zeros, with a size corresponding to the `image_desc.buffer_size()`.
    fn dummy_data(&self) -> Vec<u8> {
        vec![0; self.image_desc.buffer_size() as usize]
    }

    /// @brief Retrieves an available presentation buffer ID and updates its state to `Mapping`.
    /// This method first checks for `Available` buffers, then `Unassigned` ones,
    /// creating a new WGPU buffer if necessary for `Unassigned` IDs.
    /// @param global A reference to the WGPU `Global` object.
    /// @return An `Option<id::BufferId>` containing the ID of an available buffer, or `None` if none are found.
    /// Pre-condition: A `swap_chain` must be present in `ContextData`.
    fn get_available_buffer(&'_ mut self, global: &Arc<Global>) -> Option<id::BufferId> {
        assert!(self.swap_chain.is_some());
        if let Some((buffer_id, buffer_state)) = self
            .buffer_ids
            .iter_mut()
            .find(|(_, state)| *state == PresentationBufferState::Available)
        {
            *buffer_state = PresentationBufferState::Mapping;
            Some(*buffer_id)
        } else if let Some((buffer_id, buffer_state)) = self
            .buffer_ids
            .iter_mut()
            .find(|(_, state)| *state == PresentationBufferState::Unassigned)
        {
            *buffer_state = PresentationBufferState::Mapping;
            let buffer_id = *buffer_id;
            // Invariant: If a buffer is `Unassigned`, it means its ID was reserved but the WGPU buffer
            // resource itself needs to be created before use.
            let buffer_desc = wgt::BufferDescriptor {
                label: None,
                size: self.image_desc.buffer_size(),
                usage: wgt::BufferUsages::MAP_READ | wgt::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            };
            let _ = global.device_create_buffer(
                self.swap_chain.as_ref().unwrap().device_id,
                &buffer_desc,
                Some(buffer_id),
            );
            Some(buffer_id)
        } else {
            // Invariant: This branch should ideally not be reached if PRESENTATION_BUFFER_COUNT
            // is sufficient for the number of in-flight frames. If reached, it indicates a resource
            // exhaustion or logic error.
            error!("No available presentation buffer: {:?}", self.buffer_ids);
            None
        }
    }

    /// @brief Retrieves a mutable reference to the `PresentationBufferState` for a given buffer ID.
    /// @param buffer_id The `id::BufferId` of the buffer whose state is to be retrieved.
    /// @return A mutable reference to the `PresentationBufferState`.
    /// Pre-condition: The provided `buffer_id` must have an associated state within `self.buffer_ids`.
    fn get_buffer_state(&mut self, buffer_id: id::BufferId) -> &mut PresentationBufferState {
        &mut self
            .buffer_ids
            .iter_mut()
            .find(|(id, _)| *id == buffer_id)
            .expect("Presentation buffer should have associated state")
            .1
    }

    /// @brief Unmaps an old `GPUPresentationBuffer` and updates its state to `Available`.
    /// This marks the buffer as ready for reuse in future presentation operations.
    /// @param presentation_buffer The `GPUPresentationBuffer` instance to be unmapped.
    /// Pre-condition: A `swap_chain` must be present. The buffer's state must be `Mapped`.
    fn unmap_old_buffer(&mut self, presentation_buffer: GPUPresentationBuffer) {
        assert!(self.swap_chain.is_some());
        let buffer_state = self.get_buffer_state(presentation_buffer.buffer_id);
        assert_eq!(*buffer_state, PresentationBufferState::Mapped);
        *buffer_state = PresentationBufferState::Available;
        // Invariant: Dropping `presentation_buffer` implicitly calls its `drop` implementation,
        // which handles the WGPU buffer unmapping.
        drop(presentation_buffer);
    }

    /// @brief Destroys the current swapchain and frees all associated WGPU buffers.
    /// This method is called when the swapchain configuration changes or the context is destroyed.
    /// @param global A reference to the WGPU `Global` object.
    fn destroy_swapchain(&mut self, global: &Arc<Global>) {
        // Invariant: Taking the swap_chain out of the Option drops it, implicitly releasing its resources.
        drop(self.swap_chain.take());
        // free all buffers
        // Invariant: Iterates through all buffer IDs and frees the underlying WGPU buffer
        // resources if they were created.
        for (buffer_id, buffer_state) in &mut self.buffer_ids {
            match buffer_state {
                PresentationBufferState::Unassigned => {
                    /* These buffer were not yet created in wgpu */
                },
                _ => {
                    global.buffer_drop(*buffer_id);
                },
            }
            *buffer_state = PresentationBufferState::Unassigned;
        }
    }

    /// @brief Fully destroys the `ContextData` instance, including its swapchain,
    /// associated buffers, and removes its image from the compositor.
    /// @param global A reference to the WGPU `Global` object.
    /// @param script_sender An `IpcSender` to send `WebGPUMsg`s, used here to free script-side buffer resources.
    /// @param compositor_api The `CrossProcessCompositorApi` to remove the image from the compositor.
    fn destroy(
        mut self,
        global: &Arc<Global>,
        script_sender: &IpcSender<WebGPUMsg>,
        compositor_api: &CrossProcessCompositorApi,
    ) {
        self.destroy_swapchain(global);
        // Invariant: Ensures that any buffer IDs managed by this context are explicitly
        // freed on the script side to prevent leaks.
        for (buffer_id, _) in self.buffer_ids {
            if let Err(e) = script_sender.send(WebGPUMsg::FreeBuffer(buffer_id)) {
                warn!("Unable to send FreeBuffer({:?}) ({:?})", buffer_id, e);
            };
        }
        compositor_api.delete_image(self.image_key);
    }

    /// @brief Checks if a given `presentation_id` is newer than the current one and updates it if so.
    /// This is crucial for managing asynchronous updates and ensuring that older frames
    /// do not overwrite newer ones.
    /// @param presentation_id The `PresentationId` to check against the current ID.
    /// @return `true` if the `presentation_id` was newer and `current_presentation_id` was updated, `false` otherwise.
    fn check_and_update_presentation_id(&mut self, presentation_id: PresentationId) -> bool {
        if presentation_id > self.current_presentation_id {
            self.current_presentation_id = presentation_id;
            true
        } else {
            false
        }
    }

    /// @brief Generates and returns the next `PresentationId` in sequence, incrementing the internal counter.
    /// @return The newly generated `PresentationId`.
    /// Invariant: The returned ID is unique and monotonically increasing within this context.
    fn next_presentation_id(&mut self) -> PresentationId {
        let res = PresentationId(self.next_presentation_id.0);
        self.next_presentation_id.0 += 1;
        res
    }
}

impl crate::WGPU {
    /// @brief Creates and initializes a new WebGPU context, associating it with a unique ID and image key.
    /// This involves setting up the initial `ContextData` and registering the image with the compositor.
    /// @param context_id The unique identifier for the new WebGPU context.
    /// @param image_key The `ImageKey` to be used by WebRender for this context's image.
    /// @param size The initial `DeviceIntSize` (width and height) of the context's drawing surface.
    /// @param buffer_ids An `ArrayVec` of `id::BufferId`s pre-allocated for presentation buffers.
    /// Pre-condition: `context_id` must not already exist in `self.wgpu_image_map`.
    pub(crate) fn create_context(
        &self,
        context_id: WebGPUContextId,
        image_key: ImageKey,
        size: DeviceIntSize,
        buffer_ids: ArrayVec<id::BufferId, PRESENTATION_BUFFER_COUNT>,
    ) {
        let context_data = ContextData::new(context_id, image_key, size, buffer_ids);
        self.compositor_api.add_image(
            image_key,
            context_data.image_desc.0,
            SerializableImageData::External(context_data.image_data),
        );
        assert!(
            self.wgpu_image_map
                .lock()
                .unwrap()
                .insert(context_id, context_data)
                .is_none(),
            "Context should be created only once!"
        );
    }

    /// @brief Retrieves the current image data from a specified WebGPU context as an `IpcSnapshot`.
    /// This involves locking the image map, accessing the context's current presentation buffer,
    /// and converting the raw buffer data into a snapshot format with appropriate pixel and alpha modes.
    /// @param context_id The `WebGPUContextId` of the context from which to retrieve the image.
    /// @return An `IpcSnapshot` containing the image data and its metadata.
    pub(crate) fn get_image(&self, context_id: WebGPUContextId) -> IpcSnapshot {
        let webgpu_contexts = self.wgpu_image_map.lock().unwrap();
        let context_data = webgpu_contexts.get(&context_id).unwrap();
        let size = context_data.image_desc.size().cast().cast_unit();
        let data = if let Some(present_buffer) = context_data
            .swap_chain
            .as_ref()
            .and_then(|swap_chain| swap_chain.data.as_ref())
        {
            let format = match context_data.image_desc.0.format {
                ImageFormat::RGBA8 => SnapshotPixelFormat::RGBA,
                ImageFormat::BGRA8 => SnapshotPixelFormat::BGRA,
                _ => unimplemented!(),
            };
            let alpha_mode = if context_data.image_desc.0.is_opaque() {
                SnapshotAlphaMode::AsOpaque {
                    premultiplied: false,
                }
            } else {
                SnapshotAlphaMode::Transparent {
                    premultiplied: true,
                }
            };
            Snapshot::from_vec(size, format, alpha_mode, present_buffer.slice().to_vec())
        } else {
            // Invariant: If no swapchain data is available, a cleared (transparent black) snapshot is returned,
            // aligning with the dummy context behavior.
            Snapshot::cleared(size)
        };
        data.as_ipc()
    }

    /// @brief Updates the configuration of an existing WebGPU context, including its size and optional `ContextConfiguration`.
    /// This method handles potential swapchain re-creation if the image descriptor changes
    /// or if a new configuration is provided.
    /// @param context_id The `WebGPUContextId` of the context to update.
    /// @param size The new `DeviceIntSize` for the context's drawing surface.
    /// @param config An `Option<ContextConfiguration>` specifying the new WebGPU configuration.
    /// If `None`, the context becomes a dummy (transparent black).
    pub(crate) fn update_context(
        &self,
        context_id: WebGPUContextId,
        size: DeviceIntSize,
        config: Option<ContextConfiguration>,
    ) {
        let mut webgpu_contexts = self.wgpu_image_map.lock().unwrap();
        let context_data = webgpu_contexts.get_mut(&context_id).unwrap();

        let presentation_id = context_data.next_presentation_id();
        // Invariant: Always check and update the presentation ID to ensure that later operations
        // only process the most recent context configurations.
        context_data.check_and_update_presentation_id(presentation_id);

        // If configuration is not provided
        // the context will be dummy/empty until recreation
        let needs_image_update = if let Some(config) = config {
            let new_image_desc =
                WebGPUImageDescriptor::new(config.format(), size, config.is_opaque);
            let needs_swapchain_rebuild = context_data.swap_chain.is_none() ||
                new_image_desc.buffer_size() != context_data.image_desc.buffer_size();
            // Block Logic: If the swapchain is missing or the buffer size changes, destroy and re-create it.
            if needs_swapchain_rebuild {
                context_data.destroy_swapchain(&self.global);
                context_data.swap_chain = Some(SwapChain {
                    device_id: config.device_id,
                    queue_id: config.queue_id,
                    data: None,
                });
            }
            context_data.image_desc.update(new_image_desc)
        } else {
            // Block Logic: If no configuration is provided, destroy the existing swapchain and revert to a default image descriptor,
            // effectively making the context a dummy.
            context_data.destroy_swapchain(&self.global);
            context_data
                .image_desc
                .update(WebGPUImageDescriptor::default(size))
        };

        // Invariant: If any image properties (format, size, opacity) have changed,
        // the compositor needs to be informed to update its image representation.
        if needs_image_update {
            self.compositor_api.update_image(
                context_data.image_key,
                context_data.image_desc.0,
                SerializableImageData::External(context_data.image_data),
            );
        }
    }

    /// @brief Copies rendered data from a WebGPU texture to an available staging presentation buffer,
    /// then asynchronously maps the buffer and updates the WebRender image.
    /// This is the core mechanism for presenting a WebGPU frame.
    /// @param context_id The `WebGPUContextId` of the context performing the presentation.
    /// @param encoder_id The `id::Id<id::markers::CommandEncoder>` to use for the copy operation.
    /// @param texture_id The `id::Id<id::markers::Texture>` containing the rendered frame data.
    /// @return A `Result<(), Box<dyn std::error::Error>>` indicating success or failure of the operation.
    pub(crate) fn swapchain_present(
        &mut self,
        context_id: WebGPUContextId,
        encoder_id: id::Id<id::markers::CommandEncoder>,
        texture_id: id::Id<id::markers::Texture>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        fn err<T: std::error::Error + 'static>(e: Option<T>) -> Result<(), T> {
            if let Some(error) = e {
                Err(error)
            } else {
                Ok(())
            }
        }

        let global = &self.global;
        let device_id;
        let queue_id;
        let buffer_id;
        let image_desc;
        let presentation_id;
        {
            // Block Logic: Extracts necessary information (device, queue, buffer, image descriptor, presentation ID)
            // from the context data, ensuring exclusive access via mutex lock.
            if let Some(context_data) = self.wgpu_image_map.lock().unwrap().get_mut(&context_id) {
                let Some(swap_chain) = context_data.swap_chain.as_ref() else {
                    return Ok(());
                };
                device_id = swap_chain.device_id;
                queue_id = swap_chain.queue_id;
                // Pre-condition: An available presentation buffer must be successfully acquired.
                buffer_id = context_data.get_available_buffer(global).unwrap();
                image_desc = context_data.image_desc;
                // Invariant: A new presentation ID is generated to uniquely identify this frame.
                presentation_id = context_data.next_presentation_id();
            } else {
                // Invariant: If the context is not found, it implies it has been destroyed,
                // and no further action is needed for this presentation request.
                return Ok(());
            }
        }
        let comm_desc = wgt::CommandEncoderDescriptor { label: None };
        let (encoder_id, error) =
            global.device_create_command_encoder(device_id, &comm_desc, Some(encoder_id));
        err(error)?;
        let buffer_cv = wgt::TexelCopyBufferInfo {
            buffer: buffer_id,
            layout: wgt::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(image_desc.buffer_stride() as u32),
                rows_per_image: None,
            },
        };
        let texture_cv = wgt::TexelCopyTextureInfo {
            texture: texture_id,
            mip_level: 0,
            origin: wgt::Origin3d::ZERO,
            aspect: wgt::TextureAspect::All,
        };
        let copy_size = wgt::Extent3d {
            width: image_desc.size().width as u32,
            height: image_desc.size().height as u32,
            depth_or_array_layers: 1,
        };
        // Block Logic: Submits a command to the GPU queue to copy the texture data into the presentation buffer.
        global.command_encoder_copy_texture_to_buffer(
            encoder_id,
            &texture_cv,
            &buffer_cv,
            &copy_size,
        )?;
        let (command_buffer_id, error) =
            global.command_encoder_finish(encoder_id, &wgt::CommandBufferDescriptor::default());
        err(error)?;
        {
            let _guard = self.poller.lock();
            // Block Logic: Submits the command buffer containing the copy command to the GPU queue.
            global
                .queue_submit(queue_id, &[command_buffer_id])
                .map_err(|(_, error)| Error::from_error(error))?;
        }
        let callback = {
            let global = Arc::clone(&self.global);
            let wgpu_image_map = Arc::clone(&self.wgpu_image_map);
            let compositor_api = self.compositor_api.clone();
            let token = self.poller.token();
            // Invariant: The callback closure captures necessary data to update the WebRender image
            // once the buffer mapping operation completes asynchronously.
            Box::new(move |result| {
                drop(token);
                update_wr_image(
                    result,
                    global,
                    buffer_id,
                    wgpu_image_map,
                    context_id,
                    compositor_api,
                    image_desc,
                    presentation_id,
                );
            })
        };
        let map_op = BufferMapOperation {
            host: HostMap::Read,
            callback: Some(callback),
        };
        // Block Logic: Initiates an asynchronous read-mapping operation on the presentation buffer.
        // Once the buffer is mapped by the GPU, the `callback` will be executed.
        global.buffer_map_async(buffer_id, 0, Some(image_desc.buffer_size()), map_op)?;
        self.poller.wake();
        Ok(())
    }

    /// @brief Destroys a specified WebGPU context and releases all its associated resources.
    /// This includes dropping the `ContextData`, which in turn destroys the swapchain,
    /// frees buffers, and removes the image from the compositor.
    /// @param context_id The `WebGPUContextId` of the context to destroy.
    pub(crate) fn destroy_context(&mut self, context_id: WebGPUContextId) {
        self.wgpu_image_map
            .lock()
            .unwrap()
            .remove(&context_id)
            .unwrap()
            .destroy(&self.global, &self.script_sender, &self.compositor_api);
    }
}

/// @brief Callback function executed after a WebGPU buffer mapping operation completes.
/// It updates the WebRender external image if the associated `presentation_id` is still current,
/// ensuring that only the most recent frame data is presented.
/// @param result The `Result` of the buffer mapping operation, indicating success or failure.
/// @param global A shared reference to the WGPU `Global` object.
/// @param buffer_id The `id::BufferId` of the buffer that was mapped.
/// @param wgpu_image_map A shared map of `WebGPUContextId`s to their `ContextData`.
/// @param context_id The `WebGPUContextId` associated with this image update.
/// @param compositor_api The `CrossProcessCompositorApi` to update the image in WebRender.
/// @param image_desc The `WebGPUImageDescriptor` describing the image data.
/// @param presentation_id The `PresentationId` of the frame that was rendered into the buffer.
#[allow(clippy::too_many_arguments)]
fn update_wr_image(
    result: Result<(), BufferAccessError>,
    global: Arc<Global>,
    buffer_id: id::BufferId,
    wgpu_image_map: WGPUImageMap,
    context_id: WebGPUContextId,
    compositor_api: CrossProcessCompositorApi,
    image_desc: WebGPUImageDescriptor,
    presentation_id: PresentationId,
) {
    match result {
        Ok(()) => {
            if let Some(context_data) = wgpu_image_map.lock().unwrap().get_mut(&context_id) {
                // Pre-condition: Checks if the current `presentation_id` is still relevant.
                // If it's older than the context's `current_presentation_id`, the work is discarded.
                if !context_data.check_and_update_presentation_id(presentation_id) {
                    let buffer_state = context_data.get_buffer_state(buffer_id);
                    // Block Logic: If the presentation ID is outdated, unmap the buffer and make it available for reuse.
                    if *buffer_state == PresentationBufferState::Mapping {
                        let _ = global.buffer_unmap(buffer_id);
                        *buffer_state = PresentationBufferState::Available;
                    }
                    // throw away all work, because we are too old
                    return;
                }
                // Invariant: The image descriptor must match the context's current image descriptor.
                assert_eq!(image_desc, context_data.image_desc);
                let buffer_state = context_data.get_buffer_state(buffer_id);
                // Pre-condition: The buffer must be in the `Mapping` state to transition to `Mapped`.
                assert_eq!(*buffer_state, PresentationBufferState::Mapping);
                *buffer_state = PresentationBufferState::Mapped;
                let presentation_buffer =
                    GPUPresentationBuffer::new(global, buffer_id, image_desc.buffer_size());
                let Some(swap_chain) = context_data.swap_chain.as_mut() else {
                    return;
                };
                // Invariant: The new `presentation_buffer` replaces any existing one in the swapchain,
                // and the old buffer (if present) is unmapped for reuse.
                let old_presentation_buffer = swap_chain.data.replace(presentation_buffer);
                compositor_api.update_image(
                    context_data.image_key,
                    context_data.image_desc.0,
                    SerializableImageData::External(context_data.image_data),
                );
                if let Some(old_presentation_buffer) = old_presentation_buffer {
                    context_data.unmap_old_buffer(old_presentation_buffer)
                }
            } else {
                error!("WebGPU Context {:?} is destroyed", context_id);
            }
        },
        _ => error!("Could not map buffer({:?})", buffer_id),
    }
}
