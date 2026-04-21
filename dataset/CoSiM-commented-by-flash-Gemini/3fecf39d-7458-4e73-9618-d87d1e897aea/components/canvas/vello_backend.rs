/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * @file vello_backend.rs
 * @brief GPU-accelerated 2D canvas implementation utilizing the Vello rendering engine.
 * 
 * Functional Intent: Provides a complete implementation of the `GenericDrawTarget` 
 * trait using Vello. It orchestrates the translation of high-level 2D drawing 
 * commands (paths, text, images) into efficient GPU-bound command streams. 
 * The backend manages complex lifecycle states, including asynchronous rendering 
 * to textures and the subsequent read-back (download) to CPU memory for 
 * compositing or serialization.
 * 
 * Domain: Production Systems, Graphics Engines, Web Browsers (Servo), GPU Programming.
 */

use std::cell::RefCell;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::rc::Rc;

use canvas_traits::canvas::{
    CompositionOptions, FillOrStrokeStyle, FillRule, LineOptions, Path, ShadowOptions,
};
use compositing_traits::SerializableImageData;
use euclid::default::{Point2D, Rect, Size2D, Transform2D};
use fonts::{ByteIndex, FontIdentifier, FontTemplateRefMethods as _};
use ipc_channel::ipc::IpcSharedMemory;
use pixels::{Snapshot, SnapshotAlphaMode, SnapshotPixelFormat};
use range::Range;
use vello::wgpu::{
    BackendOptions, Backends, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Device,
    Extent3d, Instance, InstanceDescriptor, InstanceFlags, MapMode, Queue, TexelCopyBufferInfo,
    TexelCopyBufferLayout, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureViewDescriptor,
};
use vello::{kurbo, peniko};
use webrender_api::{ImageDescriptor, ImageDescriptorFlags};

use crate::backend::{Convert as _, GenericDrawTarget};
use crate::canvas_data::{Filter, TextRun};

thread_local! {
    /**
     * @thread_local SHARED_FONT_CACHE
     * @brief Context-aware cache for Peniko-compatible font handles.
     * 
     * Logic: Offsets the overhead of repeated FreeType initialization and font 
     * data transfer by maintaining thread-local persistence of loaded fonts.
     */
    static SHARED_FONT_CACHE: RefCell<HashMap<FontIdentifier, peniko::Font>> = RefCell::default();
}

/**
 * @struct VelloDrawTarget
 * @brief Encapsulates the state required for a GPU-backed rendering session.
 * 
 * Logic: Combines a WGPU device/queue context with Vello's Scene (command accumulator) 
 * and Renderer (execution engine).
 */
pub(crate) struct VelloDrawTarget {
    device: Device,
    queue: Queue,
    renderer: Rc<RefCell<vello::Renderer>>,
    scene: vello::Scene,
    size: Size2D<u32>,
}

fn options() -> vello::RendererOptions {
    vello::RendererOptions {
        use_cpu: false,
        num_init_threads: NonZeroUsize::new(1),
        antialiasing_support: vello::AaSupport::area_only(),
        pipeline_cache: None,
    }
}

impl VelloDrawTarget {
    /**
     * with_draw_options - Scopes a sequence of drawing commands within a specific layer.
     * 
     * Functional Utility: Handles alpha-blending and composition operations 
     * by pushing a layer onto the Vello scene stack.
     */
    fn with_draw_options<F: FnOnce(&mut Self)>(&mut self, draw_options: &CompositionOptions, f: F) {
        self.scene.push_layer(
            draw_options.composition_operation.convert(),
            1.0,
            kurbo::Affine::IDENTITY,
            &kurbo::Rect::ZERO.with_size(self.size.cast()),
        );
        f(self);
        self.scene.pop_layer();
    }

    /**
     * render_and_download - Triggers GPU execution and retrieves pixel data.
     * 
     * Algorithm: Async Read-back pipeline.
     * 1. Creates a transient destination texture and view.
     * 2. Renders the accumulated Vello Scene to the texture.
     * 3. Creates a staging buffer with hardware-specific alignment (256 bytes).
     * 4. Encodes and submits a copy command from texture to buffer.
     * 5. Asynchronously maps the buffer and invokes the provided closure with raw data.
     * 
     * Invariant: Properly handles WGPU alignment requirements and asynchronous mapping 
     * synchronization via oneshot channels.
     */
    fn render_and_download<F, R>(&self, f: F) -> R
    where
        F: FnOnce(u32, Option<&[u8]>) -> R,
    {
        let size = Extent3d {
            width: self.size.width,
            height: self.size.height,
            depth_or_array_layers: 1,
        };
        let target = self.device.create_texture(&TextureDescriptor {
            label: Some("Target texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&TextureViewDescriptor::default());
        
        // Synchronization: Execute the rendering workload.
        self.renderer
            .borrow_mut()
            .render_to_texture(
                &self.device,
                &self.queue,
                &self.scene,
                &view,
                &vello::RenderParams {
                    base_color: peniko::color::AlphaColor::TRANSPARENT,
                    width: self.size.width,
                    height: self.size.height,
                    antialiasing_method: vello::AaConfig::Area,
                },
            )
            .unwrap();

        // Block Logic: Data Read-back orchestration.
        // Optimization: Aligns rows to 256 bytes for WGPU compatibility.
        let padded_byte_width = (self.size.width * 4).next_multiple_of(256);
        let buffer_size = padded_byte_width as u64 * self.size.height as u64;
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("val"),
            size: buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Copy out buffer"),
            });
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            TexelCopyBufferInfo {
                buffer: &buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_byte_width),
                    rows_per_image: None,
                },
            },
            size,
        );
        self.queue.submit([encoder.finish()]);

        // Logic: Blocks on the GPU to ensure data is available before returning to host code.
        let result = {
            let buf_slice = buffer.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buf_slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());
            if let Err(error) =
                vello::util::block_on_wgpu(&self.device, receiver.receive()).unwrap()
            {
                log::warn!("VELLO WGPU MAP ASYNC ERROR {error}");
                return f(padded_byte_width, None);
            }
            let data = buf_slice.get_mapped_range();
            f(padded_byte_width, Some(&data))
        };
        buffer.unmap();
        result
    }
}

impl GenericDrawTarget for VelloDrawTarget {
    type SourceSurface = Vec<u8>; 

    /**
     * new - Bootstraps the WGPU instance and Vello renderer.
     * 
     * Logic: Discovers available GPU backends (excluding OpenGL) and initializes 
     * a shared RenderContext and Renderer.
     */
    fn new(size: Size2D<u32>) -> Self {
        let backends = Backends::from_env().unwrap_or_default() - Backends::GL;
        let flags = InstanceFlags::from_build_config().with_env();
        let backend_options = BackendOptions::from_env_or_default();
        let instance = Instance::new(&InstanceDescriptor {
            backends,
            flags,
            backend_options,
        });
        let mut context = vello::util::RenderContext {
            instance,
            devices: Vec::new(),
        };
        let device_id = pollster::block_on(context.device(None)).unwrap();
        let device_handle = &mut context.devices[device_id];
        let device = device_handle.device.clone();
        let queue = device_handle.queue.clone();
        let renderer = vello::Renderer::new(&device, options()).unwrap();
        let scene = vello::Scene::new();
        device.on_uncaptured_error(Box::new(|error| {
            log::error!("VELLO WGPU ERROR: {error}");
        }));
        Self {
            device,
            queue,
            renderer: Rc::new(RefCell::new(renderer)),
            scene,
            size,
        }
    }

    /**
     * clear_rect - Fills a region with transparency.
     */
    fn clear_rect(&mut self, rect: &Rect<f32>, transform: Transform2D<f32>) {
        let rect: kurbo::Rect = rect.cast().into();
        let transform = transform.cast().into();
        self.scene
            .push_layer(peniko::Compose::Clear, 0.0, transform, &rect);
        self.scene.fill(
            peniko::Fill::NonZero,
            transform,
            peniko::BrushRef::Solid(peniko::color::AlphaColor::TRANSPARENT),
            None,
            &rect,
        );
        self.scene.pop_layer();
    }

    /**
     * copy_surface - High-performance pixel blit from a source buffer.
     */
    fn copy_surface(&mut self, surface: Vec<u8>, source: Rect<i32>, destination: Point2D<i32>) {
        let destination: kurbo::Point = destination.cast::<f64>().into();
        let rect = kurbo::Rect::from_origin_size(destination, source.size.cast());

        self.scene
            .push_layer(peniko::Compose::Copy, 1.0, kurbo::Affine::IDENTITY, &rect);

        self.scene.fill(
            peniko::Fill::NonZero,
            kurbo::Affine::IDENTITY,
            &peniko::Image {
                data: peniko::Blob::from(surface),
                format: peniko::ImageFormat::Rgba8,
                width: source.size.width as u32,
                height: source.size.height as u32,
                x_extend: peniko::Extend::Pad,
                y_extend: peniko::Extend::Pad,
                quality: peniko::ImageQuality::Low,
                alpha: 1.0,
            },
            Some(kurbo::Affine::translate(destination.to_vec2())),
            &rect,
        );

        self.scene.pop_layer();
    }

    fn create_similar_draw_target(&self, size: &Size2D<i32>) -> Self {
        Self {
            device: self.device.clone(),
            queue: self.queue.clone(),
            renderer: self.renderer.clone(),
            scene: vello::Scene::new(),
            size: size.cast(),
        }
    }

    /**
     * draw_surface - Encodes a texture-drawing command with scaling and filtering.
     * 
     * Logic: Adjusts sampling quality (Bicubic vs Low) based on whether the 
     * image is being scaled up, to optimize the power/quality trade-off.
     */
    fn draw_surface(
        &mut self,
        surface: Vec<u8>,
        dest: Rect<f64>,
        source: Rect<f64>,
        filter: Filter,
        composition_options: CompositionOptions,
        transform: Transform2D<f32>,
    ) {
        let scale_up = dest.size.width > source.size.width || dest.size.height > source.size.height;
        let shape: kurbo::Rect = dest.into();
        self.with_draw_options(&composition_options, move |self_| {
            self_.scene.fill(
                peniko::Fill::NonZero,
                transform.cast().into(),
                &peniko::Image {
                    data: peniko::Blob::from(surface),
                    format: peniko::ImageFormat::Rgba8,
                    width: source.size.width as u32,
                    height: source.size.height as u32,
                    x_extend: peniko::Extend::Pad,
                    y_extend: peniko::Extend::Pad,
                    quality: if scale_up {
                        filter.convert()
                    } else {
                        peniko::ImageQuality::Low
                    },
                    alpha: composition_options.alpha as f32,
                },
                Some(
                    kurbo::Affine::translate((dest.origin.x, dest.origin.y)).pre_scale_non_uniform(
                        dest.size.width / source.size.width,
                        dest.size.height / source.size.height,
                    ),
                ),
                &shape,
            )
        })
    }

    fn draw_surface_with_shadow(
        &self,
        _surface: Vec<u8>,
        _dest: &Point2D<f32>,
        _shadow_options: ShadowOptions,
        _composition_options: CompositionOptions,
    ) {
        log::warn!("no support for drawing shadows");
    }

    fn fill(
        &mut self,
        path: &Path,
        fill_rule: FillRule,
        style: FillOrStrokeStyle,
        composition_options: CompositionOptions,
        transform: Transform2D<f32>,
    ) {
        self.with_draw_options(&composition_options, |self_| {
            self_.scene.fill(
                fill_rule.convert(),
                transform.cast().into(),
                &convert_to_brush(style, composition_options),
                None,
                &path.0,
            );
        })
    }

    /**
     * fill_text - Orchestrates glyph rendering using thread-local font caching.
     * 
     * Algorithm: Multi-run glyph layout.
     * 1. Resolves and caches peniko::Font handles in the thread-local store.
     * 2. Iterates through text runs and encodes glyph indices and positions 
     *    into the Vello Scene.
     */
    fn fill_text(
        &mut self,
        text_runs: Vec<TextRun>,
        start: Point2D<f32>,
        style: FillOrStrokeStyle,
        composition_options: CompositionOptions,
        transform: Transform2D<f32>,
    ) {
        let pattern = convert_to_brush(style, composition_options);
        let transform = transform.cast().into();
        self.with_draw_options(&composition_options, |self_| {
            let mut advance = 0.;
            for run in text_runs.iter() {
                let glyphs = &run.glyphs;
                let template = &run.font.template;

                SHARED_FONT_CACHE.with(|font_cache| {
                    let identifier = template.identifier();
                    // Synchronization: Lazy font loading into thread-local registry.
                    if !font_cache.borrow().contains_key(&identifier) {
                        font_cache.borrow_mut().insert(
                            identifier.clone(),
                            peniko::Font::new(
                                peniko::Blob::from(run.font.data().as_ref().to_vec()),
                                identifier.index(),
                            ),
                        );
                    }

                    let font_cache = font_cache.borrow();
                    let Some(font) = font_cache.get(&identifier) else {
                        return;
                    };

                    self_
                        .scene
                        .draw_glyphs(font)
                        .transform(transform)
                        .brush(&pattern)
                        .font_size(run.font.descriptor.pt_size.to_f32_px())
                        .draw(
                            peniko::Fill::NonZero,
                            glyphs
                                .iter_glyphs_for_byte_range(&Range::new(ByteIndex(0), glyphs.len()))
                                .map(|glyph| {
                                    let glyph_offset = glyph.offset().unwrap_or(Point2D::zero());
                                    let x = advance + start.x + glyph_offset.x.to_f32_px();
                                    let y = start.y + glyph_offset.y.to_f32_px();
                                    advance += glyph.advance().to_f32_px();
                                    vello::Glyph {
                                        id: glyph.id(),
                                        x,
                                        y,
                                    }
                                }),
                        );
                });
            }
        })
    }

    fn fill_rect(
        &mut self,
        rect: &Rect<f32>,
        style: FillOrStrokeStyle,
        composition_options: CompositionOptions,
        transform: Transform2D<f32>,
    ) {
        let pattern = convert_to_brush(style, composition_options);
        let transform = transform.cast().into();
        let rect: kurbo::Rect = rect.cast().into();
        self.with_draw_options(&composition_options, |self_| {
            self_
                .scene
                .fill(peniko::Fill::NonZero, transform, &pattern, None, &rect);
        })
    }

    fn get_size(&self) -> Size2D<i32> {
        self.size.cast()
    }

    fn pop_clip(&mut self) {
        self.scene.pop_layer();
    }

    fn push_clip(&mut self, path: &Path, _fill_rule: FillRule, transform: Transform2D<f32>) {
        self.scene
            .push_layer(peniko::Mix::Clip, 1.0, transform.cast().into(), &path.0);
    }

    fn push_clip_rect(&mut self, rect: &Rect<i32>) {
        let mut path = Path::new();
        let rect = rect.cast();
        path.rect(
            rect.origin.x,
            rect.origin.y,
            rect.size.width,
            rect.size.height,
        );
        self.push_clip(&path, FillRule::Nonzero, Transform2D::identity());
    }

    fn stroke(
        &mut self,
        path: &Path,
        style: FillOrStrokeStyle,
        line_options: LineOptions,
        composition_options: CompositionOptions,
        transform: Transform2D<f32>,
    ) {
        self.with_draw_options(&composition_options, |self_| {
            self_.scene.stroke(
                &line_options.convert(),
                transform.cast().into(),
                &convert_to_brush(style, composition_options),
                None,
                &path.0,
            );
        })
    }

    fn stroke_rect(
        &mut self,
        rect: &Rect<f32>,
        style: FillOrStrokeStyle,
        line_options: LineOptions,
        composition_options: CompositionOptions,
        transform: Transform2D<f32>,
    ) {
        let rect: kurbo::Rect = rect.cast().into();
        self.with_draw_options(&composition_options, |self_| {
            self_.scene.stroke(
                &line_options.convert(),
                transform.cast().into(),
                &convert_to_brush(style, composition_options),
                None,
                &rect,
            );
        })
    }

    /**
     * image_descriptor_and_serializable_data - Prepares high-level surface data for IPC transfer.
     * 
     * Logic: Triggers a render/download cycle and wraps the resulting buffer 
     * in IPC shared memory. Performs an in-place transformation if necessary 
     * to match the required byte layout.
     */
    fn image_descriptor_and_serializable_data(
        &mut self,
    ) -> (ImageDescriptor, SerializableImageData) {
        let size = self.size;
        self.render_and_download(|stride, data| {
            let image_desc = ImageDescriptor {
                format: webrender_api::ImageFormat::RGBA8,
                size: size.cast().cast_unit(),
                stride: data.map(|_| stride as i32),
                offset: 0,
                flags: ImageDescriptorFlags::empty(),
            };
            let data = SerializableImageData::Raw(if let Some(data) = data {
                let mut data = IpcSharedMemory::from_bytes(data);
                #[allow(unsafe_code)]
                unsafe {
                    pixels::generic_transform_inplace::<1, false, false>(data.deref_mut());
                };
                data
            } else {
                IpcSharedMemory::from_byte(0, size.area() as usize * 4)
            });
            (image_desc, data)
        })
    }

    /**
     * snapshot - Captures the current canvas state into an unpadded pixel buffer.
     * 
     * Logic: Removes the 256-byte row padding required by the GPU staging buffer 
     * to produce a tight linear array of RGBA bytes.
     */
    fn snapshot(&mut self) -> pixels::Snapshot {
        let size = self.size;
        self.render_and_download(|padded_byte_width, data| {
            let data = data
                .map(|data| {
                    let mut result_unpadded = Vec::<u8>::with_capacity(size.area() as usize * 4);
                    // Block Logic: Row-wise de-padding.
                    for row in 0..self.size.height {
                        let start = (row * padded_byte_width).try_into().unwrap();
                        result_unpadded
                            .extend(&data[start..start + (self.size.width * 4) as usize]);
                    }
                    result_unpadded
                })
                .unwrap_or_else(|| vec![0; size.area() as usize * 4]);
            Snapshot::from_vec(
                size,
                SnapshotPixelFormat::RGBA,
                SnapshotAlphaMode::Transparent {
                    premultiplied: false,
                },
                data,
            )
        })
    }

    fn surface(&mut self) -> Vec<u8> {
        self.snapshot().to_vec(None, None).0
    }

    fn create_source_surface_from_data(&self, data: Snapshot) -> Option<Vec<u8>> {
        let (data, _, _) = data.to_vec(
            Some(SnapshotAlphaMode::Transparent {
                premultiplied: false,
            }),
            Some(SnapshotPixelFormat::RGBA),
        );
        Some(data)
    }
}

/**
 * convert_to_brush - Helper to create a Peniko Brush with applied global alpha.
 */
fn convert_to_brush(
    style: FillOrStrokeStyle,
    composition_options: CompositionOptions,
) -> peniko::Brush {
    let brush: peniko::Brush = style.convert();
    brush.multiply_alpha(composition_options.alpha as f32)
}
