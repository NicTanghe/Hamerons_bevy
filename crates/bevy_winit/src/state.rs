use alloc::sync::Arc;
use approx::relative_eq;
use bevy_app::{App, AppExit, PluginsState};
use bevy_ecs::{
    change_detection::{DetectChanges, Res},
    entity::Entity,
    message::MessageCursor,
    prelude::*,
    system::SystemState,
    world::FromWorld,
};
use bevy_input::{
    gestures::*,
    mouse::{MouseButtonInput, MouseMotion, MouseScrollUnit, MouseWheel},
    pen::{PenAction, PenInfo, PenInput},
    touch::TouchPhase,
};
use bevy_log::{trace, warn};
use bevy_math::{ivec2, DVec2, Vec2};
use bevy_platform::collections::HashMap;
use bevy_platform::time::Instant;
#[cfg(not(target_arch = "wasm32"))]
use bevy_tasks::tick_global_task_pools_on_main_thread;
use core::sync::atomic::{AtomicBool, Ordering};
use std::{path::PathBuf, sync::Mutex};
#[cfg(target_arch = "wasm32")]
use winit::platform::web::EventLoopExtWebSys;
use winit::{
    application::ApplicationHandler,
    data_transfer::TypeHint,
    dpi::{PhysicalPosition, PhysicalSize},
    event,
    event::{ButtonSource, DeviceEvent, PointerKind, PointerSource, StartCause, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, DndAction, EventLoop},
    window::WindowId,
};

use bevy_window::{
    AppLifecycle, CursorEntered, CursorLeft, CursorMoved, FileDragAndDrop, Ime, RequestRedraw,
    Window, WindowBackendScaleFactorChanged, WindowCloseRequested, WindowDestroyed,
    WindowEvent as BevyWindowEvent, WindowFocused, WindowMoved, WindowOccluded, WindowResized,
    WindowScaleFactorChanged, WindowThemeChanged,
};
#[cfg(target_os = "android")]
use bevy_window::{CursorOptions, PrimaryWindow, RawHandleWrapper};

use crate::{
    accessibility::ACCESS_KIT_ADAPTERS,
    converters::{self, convert_touch_phase},
    create_windows,
    system::{create_monitors, CachedWindow, WinitWindowPressedKeys},
    AppSendEvent, CreateMonitorParams, CreateWindowParams, RawWinitWindowEvent, UpdateMode,
    WinitSettings, WINIT_WINDOWS,
};

struct PendingFileDrag {
    window: Entity,
    paths: Option<Vec<PathBuf>>,
    dropped: bool,
}

fn send_file_drag_events(
    events: &mut Vec<BevyWindowEvent>,
    window: Entity,
    paths: Vec<PathBuf>,
    dropped: bool,
) {
    for path_buf in paths {
        if dropped {
            events.send(FileDragAndDrop::DroppedFile { window, path_buf });
        } else {
            events.send(FileDragAndDrop::HoveredFile { window, path_buf });
        }
    }
}

/// Persistent state that is used to run the [`App`] according to the current
/// [`UpdateMode`].
pub(crate) struct WinitAppRunnerState {
    /// The running app.
    app: App,
    /// Exit value once the loop is finished.
    app_exit: Arc<Mutex<Option<AppExit>>>,
    /// Coalesced notification that an ECS window component was added.
    window_added: Arc<AtomicBool>,
    /// Current update mode of the app.
    update_mode: UpdateMode,
    /// Is `true` if a new [`WindowEvent`] event has been received since the last update.
    window_event_received: bool,
    /// Is `true` if a new [`DeviceEvent`] event has been received since the last update.
    device_event_received: bool,
    /// Is `true` if a new `T` event has been received since the last update.
    user_event_received: bool,
    /// Is `true` if the app has requested a redraw since the last update.
    redraw_requested: bool,
    /// Is `true` if the app has already updated since the last redraw.
    ran_update_since_last_redraw: bool,
    /// Is `true` if enough time has elapsed since `last_update` to run another update.
    wait_elapsed: bool,
    /// Number of "forced" updates to trigger on application start
    startup_forced_updates: u32,

    /// Current app lifecycle state.
    lifecycle: AppLifecycle,
    /// The previous app lifecycle state.
    previous_lifecycle: AppLifecycle,
    /// Bevy window events to send
    bevy_window_events: Vec<bevy_window::WindowEvent>,
    /// Raw Winit window events to send
    raw_winit_events: Vec<RawWinitWindowEvent>,
    /// Active touch contacts used to distinguish a normal release-followed-by-leave
    /// sequence from a canceled touch.
    active_touches: HashMap<usize, (PhysicalPosition<f64>, Option<event::Force>)>,
    /// File transfers requested from winit's asynchronous drag-and-drop API.
    pending_file_drags: HashMap<i64, PendingFileDrag>,

    windows_system_state: SystemState<
        Query<
            'static,
            'static,
            (
                &'static mut Window,
                &'static mut CachedWindow,
                &'static mut WinitWindowPressedKeys,
            ),
        >,
    >,
    /// time at which next tick is scheduled to run when `update_mode` is [`UpdateMode::Reactive`]
    scheduled_tick_start: Option<Instant>,
}

impl WinitAppRunnerState {
    fn new(
        mut app: App,
        app_exit: Arc<Mutex<Option<AppExit>>>,
        window_added: Arc<AtomicBool>,
    ) -> Self {
        let windows_system_state: SystemState<
            Query<(&mut Window, &mut CachedWindow, &mut WinitWindowPressedKeys)>,
        > = SystemState::new(app.world_mut());

        Self {
            app,
            lifecycle: AppLifecycle::Idle,
            previous_lifecycle: AppLifecycle::Idle,
            app_exit,
            window_added,
            update_mode: UpdateMode::Continuous,
            window_event_received: false,
            device_event_received: false,
            user_event_received: false,
            redraw_requested: false,
            ran_update_since_last_redraw: false,
            wait_elapsed: false,
            // 3 seems to be enough, 5 is a safe margin
            startup_forced_updates: 5,
            bevy_window_events: Vec::new(),
            raw_winit_events: Vec::new(),
            active_touches: HashMap::new(),
            pending_file_drags: HashMap::new(),
            windows_system_state,
            scheduled_tick_start: None,
        }
    }

    fn reset_on_update(&mut self) {
        self.window_event_received = false;
        self.device_event_received = false;
        self.user_event_received = false;
    }

    fn world(&self) -> &World {
        self.app.world()
    }

    pub(crate) fn world_mut(&mut self) -> &mut World {
        self.app.world_mut()
    }
}

impl ApplicationHandler for WinitAppRunnerState {
    fn new_events(&mut self, event_loop: &dyn ActiveEventLoop, cause: StartCause) {
        if event_loop.exiting() {
            return;
        }

        #[cfg(feature = "trace")]
        let _span = tracing::info_span!("winit event_handler").entered();

        if self.app.plugins_state() != PluginsState::Cleaned {
            if self.app.plugins_state() != PluginsState::Ready {
                #[cfg(not(target_arch = "wasm32"))]
                tick_global_task_pools_on_main_thread();
            } else {
                self.app.finish();
                self.app.cleanup();
            }
            self.redraw_requested = true;
        }

        self.wait_elapsed = match cause {
            StartCause::WaitCancelled {
                requested_resume, ..
            } => {
                // If the resume time is not after now, it means that at least the wait timeout
                // has elapsed. Alternatively, if the resume time is unset, the wait never elapses.
                requested_resume
                    .map(|resume| resume <= Instant::now())
                    .unwrap_or_default()
            }
            _ => true,
        };
    }

    fn resumed(&mut self, _event_loop: &dyn ActiveEventLoop) {
        // Mark the state as `WillResume`. This will let the schedule run one extra time
        // when actually resuming the app
        self.lifecycle = AppLifecycle::WillResume;
    }

    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        self.lifecycle = AppLifecycle::WillResume;

        // Create the initial window if needed.
        let mut create_window = SystemState::<CreateWindowParams>::from_world(self.world_mut());
        create_windows(event_loop, create_window.get_mut(self.world_mut()).unwrap());
        create_window.apply(self.world_mut());
    }

    fn proxy_wake_up(&mut self, event_loop: &dyn ActiveEventLoop) {
        self.user_event_received = true;
        self.redraw_requested = true;

        if self.window_added.swap(false, Ordering::AcqRel) {
            let mut create_window = SystemState::<CreateWindowParams>::from_world(self.world_mut());
            create_windows(event_loop, create_window.get_mut(self.world_mut()).unwrap());
            create_window.apply(self.world_mut());
        }
    }

    fn window_event(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        self.window_event_received = true;

        #[cfg_attr(
            not(target_os = "windows"),
            expect(unused_mut, reason = "only needs to be mut on windows for now")
        )]
        let mut manual_run_redraw_requested = false;

        WINIT_WINDOWS.with_borrow(|winit_windows| {
            ACCESS_KIT_ADAPTERS.with_borrow_mut(|access_kit_adapters| {
                let mut windows = self
                    .windows_system_state
                    .get_mut(self.app.world_mut())
                    .unwrap();

                let Some(window) = winit_windows.get_window_entity(window_id) else {
                    warn!("Skipped event {event:?} for unknown winit Window Id {window_id:?}");
                    return;
                };

                let Ok((mut win, _, mut pressed_keys)) = windows.get_mut(window) else {
                    warn!(
                        "Window {window:?} is missing `Window` component, skipping event {event:?}"
                    );
                    return;
                };

                // Store a copy of the event to send to a MessageWriter later.
                self.raw_winit_events.push(RawWinitWindowEvent {
                    window_id,
                    event: event.clone(),
                });

                // Allow AccessKit to respond to `WindowEvent`s before they reach
                // the engine.
                if let Some(adapter) = access_kit_adapters.get_mut(&window)
                    && let Some(winit_window) = winit_windows.get_window(window)
                {
                    adapter.process_event(winit_window.as_ref(), &event);
                }

                match event {
                    WindowEvent::SurfaceResized(size) => self
                        .bevy_window_events
                        .send(react_to_resize(window, &mut win, size)),
                    WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                        let (window_backend_scale_factor_changed, window_scale_factor_changed) =
                            react_to_scale_factor_change(window, &mut win, scale_factor);

                        self.bevy_window_events
                            .send(window_backend_scale_factor_changed);
                        if let Some(window_scale_factor_changed) = window_scale_factor_changed {
                            self.bevy_window_events.send(window_scale_factor_changed);
                        }
                    }
                    WindowEvent::CloseRequested => self
                        .bevy_window_events
                        .send(WindowCloseRequested { window }),
                    WindowEvent::KeyboardInput {
                        ref event,
                        // On some platforms, winit sends "synthetic" key press events when the window
                        // gains or loses focus. These are not implemented on every platform, so we ignore
                        // winit's synthetic key pressed and implement the same mechanism ourselves.
                        // (See the `WinitWindowPressedKeys` component)
                        is_synthetic: false,
                        ..
                    } => {
                        let keyboard_input = converters::convert_keyboard_input(event, window);
                        if event.state.is_pressed() {
                            pressed_keys.0.insert(
                                keyboard_input.key_code,
                                keyboard_input.logical_key.clone(),
                            );
                        } else {
                            pressed_keys.0.remove(&keyboard_input.key_code);
                        }
                        self.bevy_window_events.send(keyboard_input);
                    }
                    WindowEvent::PointerMoved {
                        device_id,
                        position,
                        primary,
                        source,
                    } => match source {
                        PointerSource::Mouse | PointerSource::Unknown => {
                            let physical_position = DVec2::new(position.x, position.y);
                            let last_position = win.physical_cursor_position();
                            let delta = last_position.map(|last_pos| {
                                (physical_position.as_vec2() - last_pos)
                                    / win.resolution.scale_factor()
                            });

                            win.set_physical_cursor_position(Some(physical_position));
                            let position = (physical_position
                                / win.resolution.scale_factor() as f64)
                                .as_vec2();
                            self.bevy_window_events.send(CursorMoved {
                                window,
                                position,
                                delta,
                            });
                        }
                        PointerSource::Touch { finger_id, force } => {
                            if let Some(contact) =
                                self.active_touches.get_mut(&finger_id.into_raw())
                            {
                                *contact = (position, force);
                                let location = position
                                    .to_logical::<f64>(win.resolution.scale_factor() as f64);
                                self.bevy_window_events
                                    .send(converters::convert_touch_input(
                                        TouchPhase::Moved,
                                        location,
                                        force,
                                        finger_id,
                                        window,
                                    ));
                            }
                        }
                        PointerSource::TabletTool { kind, data } => {
                            let location =
                                position.to_logical::<f64>(win.resolution.scale_factor() as f64);
                            self.bevy_window_events.send(PenInput {
                                pen: PenInfo {
                                    window,
                                    device: converters::convert_pen_id(device_id),
                                    primary,
                                    position: Some(Vec2::new(location.x as f32, location.y as f32)),
                                    tool: converters::convert_pen_tool_kind(kind),
                                },
                                action: PenAction::Moved(converters::convert_pen_data(data)),
                            });
                        }
                    },
                    WindowEvent::PointerEntered {
                        device_id,
                        position,
                        primary,
                        kind,
                    } => match kind {
                        PointerKind::Mouse | PointerKind::Unknown => {
                            self.bevy_window_events.send(CursorEntered { window });
                        }
                        PointerKind::Touch(_) => {}
                        PointerKind::TabletTool(kind) => {
                            let location =
                                position.to_logical::<f64>(win.resolution.scale_factor() as f64);
                            self.bevy_window_events.send(PenInput {
                                pen: PenInfo {
                                    window,
                                    device: converters::convert_pen_id(device_id),
                                    primary,
                                    position: Some(Vec2::new(location.x as f32, location.y as f32)),
                                    tool: converters::convert_pen_tool_kind(kind),
                                },
                                action: PenAction::Entered,
                            });
                        }
                    },
                    WindowEvent::PointerLeft {
                        device_id,
                        position,
                        primary,
                        kind,
                    } => match kind {
                        PointerKind::Mouse | PointerKind::Unknown => {
                            win.set_physical_cursor_position(None);
                            self.bevy_window_events.send(CursorLeft { window });
                        }
                        PointerKind::Touch(finger_id) => {
                            if let Some((last_position, force)) =
                                self.active_touches.remove(&finger_id.into_raw())
                            {
                                let position = position.unwrap_or(last_position);
                                let location = position
                                    .to_logical::<f64>(win.resolution.scale_factor() as f64);
                                self.bevy_window_events
                                    .send(converters::convert_touch_input(
                                        TouchPhase::Canceled,
                                        location,
                                        force,
                                        finger_id,
                                        window,
                                    ));
                            }
                        }
                        PointerKind::TabletTool(kind) => {
                            let position = position.map(|position| {
                                let location = position
                                    .to_logical::<f64>(win.resolution.scale_factor() as f64);
                                Vec2::new(location.x as f32, location.y as f32)
                            });
                            self.bevy_window_events.send(PenInput {
                                pen: PenInfo {
                                    window,
                                    device: converters::convert_pen_id(device_id),
                                    primary,
                                    position,
                                    tool: converters::convert_pen_tool_kind(kind),
                                },
                                action: PenAction::Left,
                            });
                        }
                    },
                    WindowEvent::PointerButton {
                        device_id,
                        state,
                        position,
                        primary,
                        button,
                    } => match button {
                        ButtonSource::Mouse(button) => {
                            self.bevy_window_events.send(MouseButtonInput {
                                button: converters::convert_mouse_button(button),
                                state: converters::convert_element_state(state),
                                window,
                            });
                        }
                        ButtonSource::Touch { finger_id, force } => {
                            let phase = if state.is_pressed() {
                                self.active_touches
                                    .insert(finger_id.into_raw(), (position, force));
                                TouchPhase::Started
                            } else {
                                TouchPhase::Ended
                            };
                            let location =
                                position.to_logical::<f64>(win.resolution.scale_factor() as f64);
                            self.bevy_window_events
                                .send(converters::convert_touch_input(
                                    phase, location, force, finger_id, window,
                                ));
                            if !state.is_pressed() {
                                self.active_touches.remove(&finger_id.into_raw());
                            }
                        }
                        ButtonSource::TabletTool { kind, button, data } => {
                            let location =
                                position.to_logical::<f64>(win.resolution.scale_factor() as f64);
                            self.bevy_window_events.send(PenInput {
                                pen: PenInfo {
                                    window,
                                    device: converters::convert_pen_id(device_id),
                                    primary,
                                    position: Some(Vec2::new(location.x as f32, location.y as f32)),
                                    tool: converters::convert_pen_tool_kind(kind),
                                },
                                action: PenAction::Button {
                                    button: converters::convert_pen_button(button),
                                    state: converters::convert_element_state(state),
                                    data: converters::convert_pen_data(data),
                                },
                            });
                        }
                        ButtonSource::Unknown(_) => {}
                    },
                    WindowEvent::PinchGesture { delta, .. } => {
                        self.bevy_window_events.send(PinchGesture(delta as f32));
                    }
                    WindowEvent::RotationGesture { delta, .. } => {
                        self.bevy_window_events.send(RotationGesture(delta));
                    }
                    WindowEvent::DoubleTapGesture { .. } => {
                        self.bevy_window_events.send(DoubleTapGesture);
                    }
                    WindowEvent::PanGesture { delta, .. } => {
                        self.bevy_window_events.send(PanGesture(Vec2 {
                            x: delta.x,
                            y: delta.y,
                        }));
                    }
                    WindowEvent::MouseWheel { delta, phase, .. } => {
                        let phase = convert_touch_phase(phase);
                        match delta {
                            event::MouseScrollDelta::LineDelta(x, y) => {
                                self.bevy_window_events.send(MouseWheel {
                                    unit: MouseScrollUnit::Line,
                                    x,
                                    y,
                                    window,
                                    phase,
                                });
                            }
                            event::MouseScrollDelta::PixelDelta(p) => {
                                self.bevy_window_events.send(MouseWheel {
                                    unit: MouseScrollUnit::Pixel,
                                    x: p.x as f32,
                                    y: p.y as f32,
                                    window,
                                    phase,
                                });
                            }
                        }
                    }
                    WindowEvent::Focused(focused) => {
                        win.focused = focused;
                        self.bevy_window_events
                            .send(WindowFocused { window, focused });
                    }
                    WindowEvent::Occluded(occluded) => {
                        self.bevy_window_events
                            .send(WindowOccluded { window, occluded });
                    }
                    WindowEvent::DragEntered { id, .. } => {
                        if event_loop
                            .data_transfer(id)
                            .is_ok_and(|transfer| transfer.has_type(&TypeHint::UriList))
                        {
                            self.pending_file_drags.insert(
                                id.into_raw(),
                                PendingFileDrag {
                                    window,
                                    paths: None,
                                    dropped: false,
                                },
                            );

                            if let Err(error) =
                                event_loop.set_valid_dnd_actions(id, &[DndAction::Copy])
                            {
                                warn!("failed to accept file drag {id:?}: {error}");
                            }
                            if let Err(error) =
                                event_loop.fetch_data_transfer(id, &TypeHint::UriList)
                            {
                                warn!("failed to request paths for file drag {id:?}: {error}");
                            }
                        }
                    }
                    WindowEvent::DragDropped { id, .. } => {
                        let paths =
                            self.pending_file_drags
                                .get_mut(&id.into_raw())
                                .and_then(|pending| {
                                    pending.dropped = true;
                                    pending.paths.clone()
                                });

                        if let Some(paths) = paths {
                            send_file_drag_events(
                                &mut self.bevy_window_events,
                                window,
                                paths,
                                true,
                            );
                            self.pending_file_drags.remove(&id.into_raw());
                        } else if self.pending_file_drags.contains_key(&id.into_raw())
                            && let Err(error) =
                                event_loop.fetch_data_transfer(id, &TypeHint::UriList)
                        {
                            warn!("failed to request dropped file paths for {id:?}: {error}");
                        }
                    }
                    WindowEvent::DragLeft { id } => {
                        if let Some(pending) = self.pending_file_drags.remove(&id.into_raw()) {
                            self.bevy_window_events
                                .send(FileDragAndDrop::HoveredFileCanceled {
                                    window: pending.window,
                                });
                        }
                    }
                    WindowEvent::DataTransferReceived { id, value, .. } => {
                        if value.type_().hint() == Some(TypeHint::UriList)
                            && let Some(pending) = self.pending_file_drags.get_mut(&id.into_raw())
                        {
                            match value.try_as_file_paths() {
                                Ok(paths) => {
                                    let dropped = pending.dropped;
                                    let window = pending.window;
                                    pending.paths = Some(paths.clone());
                                    send_file_drag_events(
                                        &mut self.bevy_window_events,
                                        window,
                                        paths,
                                        dropped,
                                    );
                                    if dropped {
                                        self.pending_file_drags.remove(&id.into_raw());
                                    }
                                }
                                Err(error) => {
                                    warn!("failed to read paths for file drag {id:?}: {error}");
                                }
                            }
                        }
                    }
                    WindowEvent::Moved(position) => {
                        let position = ivec2(position.x, position.y);
                        win.position.set(position);
                        self.bevy_window_events
                            .send(WindowMoved { window, position });
                    }
                    WindowEvent::Ime(event) => match event {
                        event::Ime::Preedit(value, cursor) => {
                            self.bevy_window_events.send(Ime::Preedit {
                                window,
                                value,
                                cursor,
                            });
                        }
                        event::Ime::Commit(value) => {
                            self.bevy_window_events.send(Ime::Commit { window, value });
                        }
                        event::Ime::Enabled => {
                            self.bevy_window_events.send(Ime::Enabled { window });
                        }
                        event::Ime::Disabled => {
                            self.bevy_window_events.send(Ime::Disabled { window });
                        }
                        event::Ime::DeleteSurrounding { .. } => {}
                    },
                    WindowEvent::ThemeChanged(theme) => {
                        self.bevy_window_events.send(WindowThemeChanged {
                            window,
                            theme: converters::convert_winit_theme(theme),
                        });
                    }
                    WindowEvent::Destroyed => {
                        self.bevy_window_events.send(WindowDestroyed { window });
                    }
                    WindowEvent::RedrawRequested => {
                        self.ran_update_since_last_redraw = false;

                        // https://github.com/bevyengine/bevy/issues/17488
                        #[cfg(target_os = "windows")]
                        {
                            // Have the startup behavior run in about_to_wait, which prevents issues with
                            // invisible window creation. https://github.com/bevyengine/bevy/issues/18027
                            if self.startup_forced_updates == 0 {
                                manual_run_redraw_requested = true;
                            }
                        }
                    }
                    _ => {}
                }

                let mut windows = self.world_mut().query::<(&mut Window, &mut CachedWindow)>();
                if let Ok((window_component, mut cache)) = windows.get_mut(self.world_mut(), window)
                    && window_component.is_changed()
                {
                    **cache = window_component.clone();
                }
            });
        });

        if manual_run_redraw_requested {
            self.redraw_requested(event_loop);
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &dyn ActiveEventLoop,
        _device_id: Option<event::DeviceId>,
        event: DeviceEvent,
    ) {
        self.device_event_received = true;

        if let DeviceEvent::PointerMotion { delta: (x, y) } = event {
            let delta = Vec2::new(x as f32, y as f32);
            self.bevy_window_events.send(MouseMotion { delta });
        }
    }

    fn about_to_wait(&mut self, event_loop: &dyn ActiveEventLoop) {
        let mut create_monitor = SystemState::<CreateMonitorParams>::from_world(self.world_mut());
        create_monitors(
            event_loop,
            create_monitor.get_mut(self.world_mut()).unwrap(),
        );
        create_monitor.apply(self.world_mut());

        // TODO: This is a workaround for https://github.com/bevyengine/bevy/issues/17488
        //       while preserving the iOS fix in https://github.com/bevyengine/bevy/pull/11245
        //       The monitor sync logic likely belongs in monitor event handlers and not here.
        #[cfg(not(target_os = "windows"))]
        self.redraw_requested(event_loop);

        // Have the startup behavior run in about_to_wait, which prevents issues with
        // invisible window creation. https://github.com/bevyengine/bevy/issues/18027
        #[cfg(target_os = "windows")]
        {
            fn headless_or_all_invisible() -> bool {
                WINIT_WINDOWS.with_borrow(|winit_windows| {
                    winit_windows
                        .windows
                        .iter()
                        .all(|(_, w)| !w.is_visible().unwrap_or(false))
                })
            }

            if self.app_exit.lock().unwrap().is_none()
                && (self.startup_forced_updates > 0
                    || matches!(self.update_mode, UpdateMode::Reactive { .. })
                    || self.window_event_received
                    || headless_or_all_invisible())
            {
                self.redraw_requested(event_loop);
            }
        }
    }

    fn suspended(&mut self, _event_loop: &dyn ActiveEventLoop) {
        // Mark the state as `WillSuspend`. This will let the schedule run one last time
        // before actually suspending to let the application react
        self.lifecycle = AppLifecycle::WillSuspend;
    }

    fn destroy_surfaces(&mut self, _event_loop: &dyn ActiveEventLoop) {
        self.lifecycle = AppLifecycle::WillSuspend;
    }
}

impl Drop for WinitAppRunnerState {
    fn drop(&mut self) {
        // Drop windows while the event-loop-owned application state is still
        // being torn down on the event loop thread.
        WINIT_WINDOWS.with(|ww| ww.borrow_mut().windows.clear());
        self.app.world_mut().clear_all();
    }
}

impl WinitAppRunnerState {
    fn redraw_requested(&mut self, event_loop: &dyn ActiveEventLoop) {
        let mut redraw_message_cursor = MessageCursor::<RequestRedraw>::default();
        let mut close_message_cursor = MessageCursor::<WindowCloseRequested>::default();

        let mut focused_windows_state: SystemState<(Res<WinitSettings>, Query<(Entity, &Window)>)> =
            SystemState::new(self.world_mut());

        let (config, windows) = focused_windows_state.get(self.world()).unwrap();
        let focused = windows.iter().any(|(_, window)| window.focused);

        let mut update_mode = config.update_mode(focused);
        let mut should_update = self.should_update(update_mode);

        if self.startup_forced_updates > 0 {
            self.startup_forced_updates -= 1;
            // Ensure that an update is triggered on the first iterations for app initialization
            should_update = true;
        }

        if self.lifecycle == AppLifecycle::WillSuspend {
            self.lifecycle = AppLifecycle::Suspended;
            // Trigger one last update to enter the suspended state
            should_update = true;
            self.ran_update_since_last_redraw = false;

            #[cfg(target_os = "android")]
            {
                // Remove the `RawHandleWrapper` from the primary window.
                // This will trigger the surface destruction.
                let mut query = self
                    .world_mut()
                    .query_filtered::<Entity, With<PrimaryWindow>>();
                if let Ok(entity) = query.single(&self.world()) {
                    self.world_mut()
                        .entity_mut(entity)
                        .remove::<RawHandleWrapper>();
                }
            }
        }

        if self.lifecycle == AppLifecycle::WillResume {
            self.lifecycle = AppLifecycle::Running;
            // Trigger the update to enter the running state
            should_update = true;
            // Trigger the next redraw to refresh the screen immediately
            self.redraw_requested = true;

            #[cfg(target_os = "android")]
            {
                // Get windows that are cached but without raw handles. Those window were already created, but got their
                // handle wrapper removed when the app was suspended.

                let mut query = self.world_mut()
                    .query_filtered::<(Entity, &Window, &CursorOptions), (With<CachedWindow>, Without<RawHandleWrapper>)>();
                if let Ok((entity, window, cursor_options)) = query.single(&self.world()) {
                    let window = window.clone();
                    let cursor_options = cursor_options.clone();

                    WINIT_WINDOWS.with_borrow_mut(|winit_windows| {
                        ACCESS_KIT_ADAPTERS.with_borrow_mut(|adapters| {
                            let mut create_window =
                                SystemState::<CreateWindowParams>::from_world(self.world_mut());

                            let (.., mut handlers, accessibility_requested, monitors) =
                                create_window.get_mut(self.world_mut()).unwrap();

                            let winit_window = winit_windows.create_window(
                                event_loop,
                                entity,
                                &window,
                                &cursor_options,
                                adapters,
                                &mut handlers,
                                &accessibility_requested,
                                &monitors,
                            );

                            let wrapper = RawHandleWrapper::new(winit_window).unwrap();

                            self.world_mut().entity_mut(entity).insert(wrapper);
                        });
                    });
                }
            }
        }

        // Notifies a lifecycle change
        if self.lifecycle != self.previous_lifecycle {
            self.previous_lifecycle = self.lifecycle;
            self.bevy_window_events.send(self.lifecycle);
        }

        // This is recorded before running app.update(), to run the next cycle after a correct timeout.
        // If the cycle takes more than the wait timeout, it will be re-executed immediately.
        let begin_frame_time = Instant::now();

        if should_update {
            let (_, windows) = focused_windows_state.get(self.world()).unwrap();
            // If no windows exist, this will evaluate to `true`.
            let all_invisible = windows.iter().all(|w| !w.1.visible);

            // Not redrawing, but the timeout elapsed.
            //
            // Additional condition for Windows OS.
            // If no windows are visible, redraw calls will never succeed, which results in no app update calls being performed.
            // This is a temporary solution, full solution is mentioned here: https://github.com/bevyengine/bevy/issues/1343#issuecomment-770091684
            if !self.ran_update_since_last_redraw || all_invisible {
                self.run_app_update();
                #[cfg(feature = "custom_cursor")]
                self.update_cursors(event_loop);
                #[cfg(not(feature = "custom_cursor"))]
                self.update_cursors();
                self.ran_update_since_last_redraw = true;
            } else {
                self.redraw_requested = true;
            }

            // Read RequestRedraw events that may have been sent during the update
            if let Some(app_redraw_events) = self.world().get_resource::<Messages<RequestRedraw>>()
                && redraw_message_cursor
                    .read(app_redraw_events)
                    .last()
                    .is_some()
            {
                self.redraw_requested = true;
            }

            // Running the app may have produced WindowCloseRequested events that should be processed
            if let Some(close_request_messages) = self
                .world()
                .get_resource::<Messages<WindowCloseRequested>>()
                && close_message_cursor
                    .read(close_request_messages)
                    .last()
                    .is_some()
            {
                self.redraw_requested = true;
            }

            // Running the app may have changed the WinitSettings resource, so we have to re-extract it.
            let (config, windows) = focused_windows_state.get(self.world()).unwrap();
            let focused = windows.iter().any(|(_, window)| window.focused);
            update_mode = config.update_mode(focused);
        }

        // The update mode could have been changed, so we need to redraw and force an update
        if update_mode != self.update_mode {
            // Trigger the next redraw since we're changing the update mode
            self.redraw_requested = true;
            // Consider the wait as elapsed since it could have been cancelled by a user event
            self.wait_elapsed = true;
            // reset the scheduled start time
            self.scheduled_tick_start = None;

            self.update_mode = update_mode;
        }

        match update_mode {
            UpdateMode::Continuous => {
                // per winit's docs on [Window::is_visible](https://docs.rs/winit/latest/winit/window/struct.Window.html#method.is_visible),
                // we cannot use the visibility to drive rendering on these platforms
                // so we cannot discern whether to beneficially use `Poll` or not?
                cfg_select! {
                    not(any(
                        target_arch = "wasm32",
                        target_os = "android",
                        target_os = "ios",
                        all(target_os = "linux", any(feature = "x11", feature = "wayland"))
                    )) =>
                    {
                        let visible = WINIT_WINDOWS.with_borrow(|winit_windows| {
                            winit_windows.windows.iter().any(|(_, w)| {
                                w.is_visible().unwrap_or(false)
                            })
                        });

                        event_loop.set_control_flow(if visible {
                            ControlFlow::Wait
                        } else {
                            ControlFlow::Poll
                        });
                    }
                    _ => {
                        event_loop.set_control_flow(ControlFlow::Wait);
                    }
                }

                // Trigger the next redraw to refresh the screen immediately if waiting
                if let ControlFlow::Wait = event_loop.control_flow() {
                    self.redraw_requested = true;
                }
            }
            UpdateMode::Reactive { wait, .. } => {
                // Set the next timeout, starting from the instant we were scheduled to begin
                if self.wait_elapsed {
                    self.redraw_requested = true;

                    let begin_instant = self.scheduled_tick_start.unwrap_or(begin_frame_time);
                    if let Some(next) = begin_instant.checked_add(wait) {
                        let now = Instant::now();
                        if next < now {
                            // request next redraw as soon as possible if we are already past the next scheduled frame start time
                            event_loop.set_control_flow(ControlFlow::Poll);
                            self.scheduled_tick_start = Some(now);
                        } else {
                            event_loop.set_control_flow(ControlFlow::WaitUntil(next));
                            self.scheduled_tick_start = Some(next);
                        }
                    }
                }
            }
        }

        if self.redraw_requested && self.lifecycle != AppLifecycle::Suspended {
            WINIT_WINDOWS.with_borrow(|winit_windows| {
                for window in winit_windows.windows.values() {
                    window.request_redraw();
                }
            });
            self.redraw_requested = false;
        }

        if let Some(app_exit) = self.app.should_exit() {
            *self.app_exit.lock().unwrap() = Some(app_exit);

            event_loop.exit();
        }
    }

    fn should_update(&self, update_mode: UpdateMode) -> bool {
        let handle_event = match update_mode {
            UpdateMode::Continuous => {
                self.wait_elapsed
                    || self.user_event_received
                    || self.window_event_received
                    || self.device_event_received
            }
            UpdateMode::Reactive {
                react_to_device_events,
                react_to_user_events,
                react_to_window_events,
                ..
            } => {
                self.wait_elapsed
                    || (react_to_device_events && self.device_event_received)
                    || (react_to_user_events && self.user_event_received)
                    || (react_to_window_events && self.window_event_received)
            }
        };

        handle_event && self.lifecycle.is_active()
    }

    fn run_app_update(&mut self) {
        self.reset_on_update();

        self.forward_bevy_events();

        if self.app.plugins_state() == PluginsState::Cleaned {
            self.app.update();
        }
    }

    fn forward_bevy_events(&mut self) {
        let raw_winit_events = core::mem::take(&mut self.raw_winit_events);
        let window_events = core::mem::take(&mut self.bevy_window_events);
        let world = self.world_mut();

        if !raw_winit_events.is_empty() {
            world
                .resource_mut::<Messages<RawWinitWindowEvent>>()
                .write_batch(raw_winit_events);
        }

        for window_event in window_events.iter() {
            match window_event.clone() {
                BevyWindowEvent::AppLifecycle(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::CursorEntered(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::CursorLeft(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::CursorMoved(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::FileDragAndDrop(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::Ime(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::RequestRedraw(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::WindowBackendScaleFactorChanged(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::WindowCloseRequested(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::WindowCreated(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::WindowDestroyed(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::WindowFocused(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::WindowMoved(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::WindowOccluded(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::WindowResized(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::WindowScaleFactorChanged(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::WindowThemeChanged(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::MouseButtonInput(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::MouseMotion(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::MouseWheel(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::PenInput(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::PinchGesture(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::RotationGesture(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::DoubleTapGesture(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::PanGesture(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::TouchInput(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::KeyboardInput(e) => {
                    world.write_message(e);
                }
                BevyWindowEvent::KeyboardFocusLost(e) => {
                    world.write_message(e);
                }
            }
        }

        if !window_events.is_empty() {
            world
                .resource_mut::<Messages<BevyWindowEvent>>()
                .write_batch(window_events);
        }
    }
}

/// The default [`App::runner`] for the [`WinitPlugin`](crate::WinitPlugin) plugin.
///
/// Overriding the app's [runner](bevy_app::App::runner) while using `WinitPlugin` will bypass the
/// `EventLoop`.
pub fn winit_runner(mut app: App, event_loop: EventLoop, window_added: Arc<AtomicBool>) -> AppExit {
    if app.plugins_state() == PluginsState::Ready {
        app.finish();
        app.cleanup();
    }

    let app_exit = Arc::new(Mutex::new(None));
    let runner_state = WinitAppRunnerState::new(app, Arc::clone(&app_exit), window_added);

    trace!("starting winit event loop");
    // The winit docs mention using `spawn` instead of `run` on Wasm.
    // https://docs.rs/winit/latest/winit/platform/web/trait.EventLoopExtWebSys.html#tymethod.spawn_app
    cfg_select! {
        target_arch = "wasm32" => {
            event_loop.spawn_app(runner_state);
            AppExit::Success
        }
        _ => {
            if let Err(err) = event_loop.run_app(runner_state) {
                bevy_log::error!("winit event loop returned an error: {err}");
            }
            // If everything is working correctly then the event loop only exits after it's sent an exit code.
            app_exit.lock().unwrap().take().unwrap_or_else(|| {
                bevy_log::error!("Failed to receive an app exit code! This is a bug");
                AppExit::error()
            })
        }
    }
}

pub(crate) fn react_to_resize(
    window_entity: Entity,
    window: &mut Window,
    size: PhysicalSize<u32>,
) -> WindowResized {
    window
        .resolution
        .set_physical_resolution(size.width, size.height);

    WindowResized {
        window: window_entity,
        width: window.width(),
        height: window.height(),
    }
}

pub(crate) fn react_to_scale_factor_change(
    window_entity: Entity,
    window: &mut Window,
    scale_factor: f64,
) -> (
    WindowBackendScaleFactorChanged,
    Option<WindowScaleFactorChanged>,
) {
    let prior_factor = window.resolution.scale_factor();
    window.resolution.set_scale_factor(scale_factor as f32);

    let window_backend_scale_factor_changed = WindowBackendScaleFactorChanged {
        window: window_entity,
        scale_factor,
    };

    let scale_factor_override = window.resolution.scale_factor_override();

    let window_scale_factor_changed =
        if scale_factor_override.is_none() && !relative_eq!(scale_factor as f32, prior_factor) {
            let window_scale_factor_changed = WindowScaleFactorChanged {
                window: window_entity,
                scale_factor,
            };
            Some(window_scale_factor_changed)
        } else {
            None
        };

    (
        window_backend_scale_factor_changed,
        window_scale_factor_changed,
    )
}

#[cfg(test)]
mod tests {
    use bevy_app::Update;

    use super::*;

    #[test]
    fn test_react_to_scale_factor_change_with_changed_scale_factor() {
        let (mut app, window_entity) = setup_react_to_scale_factor_change_test_app(1.0, 2.0);
        app.update();

        let window = app.world().get::<Window>(window_entity);
        assert_eq!(window.unwrap().resolution.scale_factor(), 2.0);

        let window_backend_scale_factor_changed_messages = app
            .world()
            .resource::<Messages<WindowBackendScaleFactorChanged>>();
        assert_eq!(window_backend_scale_factor_changed_messages.len(), 1);

        let mut window_backend_scale_factor_changed_messages_iter =
            window_backend_scale_factor_changed_messages.iter_current_update_messages();
        assert_eq!(
            window_backend_scale_factor_changed_messages_iter.next(),
            Some(&WindowBackendScaleFactorChanged {
                window: window_entity,
                scale_factor: 2.0,
            })
        );
        assert_eq!(
            window_backend_scale_factor_changed_messages_iter.next(),
            None
        );

        let window_scale_factor_changed_messages =
            app.world().resource::<Messages<WindowScaleFactorChanged>>();
        assert_eq!(window_scale_factor_changed_messages.len(), 1);

        let mut window_scale_factor_changed_messages_iter =
            window_scale_factor_changed_messages.iter_current_update_messages();
        assert_eq!(
            window_scale_factor_changed_messages_iter.next(),
            Some(&WindowScaleFactorChanged {
                window: window_entity,
                scale_factor: 2.0,
            })
        );
        assert_eq!(window_scale_factor_changed_messages_iter.next(), None);

        let window_event_messages = app.world().resource::<Messages<BevyWindowEvent>>();
        assert_eq!(window_event_messages.len(), 2);

        let mut window_event_messages_iter = window_event_messages.iter_current_update_messages();
        assert_eq!(
            window_event_messages_iter.next(),
            Some(&BevyWindowEvent::WindowBackendScaleFactorChanged(
                WindowBackendScaleFactorChanged {
                    window: window_entity,
                    scale_factor: 2.0
                }
            ))
        );
        assert_eq!(
            window_event_messages_iter.next(),
            Some(&BevyWindowEvent::WindowScaleFactorChanged(
                WindowScaleFactorChanged {
                    window: window_entity,
                    scale_factor: 2.0
                }
            ))
        );
        assert_eq!(window_event_messages_iter.next(), None);
    }

    #[test]
    fn test_react_to_scale_factor_change_with_same_scale_factor() {
        let (mut app, window_entity) = setup_react_to_scale_factor_change_test_app(1.0, 1.0);
        app.update();

        let window = app.world().get::<Window>(window_entity);
        assert_eq!(window.unwrap().resolution.scale_factor(), 1.0);

        let window_backend_scale_factor_changed_messages = app
            .world()
            .resource::<Messages<WindowBackendScaleFactorChanged>>();
        assert_eq!(window_backend_scale_factor_changed_messages.len(), 1);

        let mut window_backend_scale_factor_changed_messages_iter =
            window_backend_scale_factor_changed_messages.iter_current_update_messages();
        assert_eq!(
            window_backend_scale_factor_changed_messages_iter.next(),
            Some(&WindowBackendScaleFactorChanged {
                window: window_entity,
                scale_factor: 1.0,
            })
        );
        assert_eq!(
            window_backend_scale_factor_changed_messages_iter.next(),
            None
        );

        let window_scale_factor_changed_messages =
            app.world().resource::<Messages<WindowScaleFactorChanged>>();
        assert!(window_scale_factor_changed_messages.is_empty());

        let window_event_messages = app.world().resource::<Messages<BevyWindowEvent>>();
        assert_eq!(window_event_messages.len(), 1);

        let mut window_event_messages_iter = window_event_messages.iter_current_update_messages();
        assert_eq!(
            window_event_messages_iter.next(),
            Some(&BevyWindowEvent::WindowBackendScaleFactorChanged(
                WindowBackendScaleFactorChanged {
                    window: window_entity,
                    scale_factor: 1.0
                }
            ))
        );
        assert_eq!(window_event_messages_iter.next(), None);
    }

    fn setup_react_to_scale_factor_change_test_app(
        initial_scale_factor: f32,
        changed_scale_factor: f64,
    ) -> (App, Entity) {
        let mut app = App::new();
        app.add_message::<WindowBackendScaleFactorChanged>();
        app.add_message::<WindowScaleFactorChanged>();
        app.add_message::<BevyWindowEvent>();
        app.add_systems(
            Update,
            move |mut window: Single<(Entity, &mut Window)>,
                  mut window_backend_scale_factor_changed_writer: MessageWriter<
                WindowBackendScaleFactorChanged,
            >,
                  mut window_scale_factor_changed_writer: MessageWriter<
                WindowScaleFactorChanged,
            >,
                  mut window_event: MessageWriter<BevyWindowEvent>| {
                let (window_backend_scale_factor_changed, window_scale_factor_changed) =
                    react_to_scale_factor_change(window.0, &mut window.1, changed_scale_factor);
                window_backend_scale_factor_changed_writer
                    .write(window_backend_scale_factor_changed.clone());
                window_event.write(BevyWindowEvent::WindowBackendScaleFactorChanged(
                    window_backend_scale_factor_changed,
                ));
                if let Some(window_scale_factor_changed) = window_scale_factor_changed {
                    window_scale_factor_changed_writer.write(window_scale_factor_changed.clone());
                    window_event.write(BevyWindowEvent::WindowScaleFactorChanged(
                        window_scale_factor_changed,
                    ));
                }
            },
        );

        let mut window = Window::default();
        window.resolution.set_scale_factor(initial_scale_factor);
        let window_entity = app.world_mut().spawn(window).id();

        (app, window_entity)
    }

    #[test]
    fn test_react_to_resize_with_changed_size() {
        let (mut app, window_entity) =
            setup_react_to_resize(PhysicalSize::new(1280, 720), PhysicalSize::new(1920, 1080));
        app.update();

        let window = app.world().get::<Window>(window_entity).unwrap();
        assert_eq!(window.resolution.physical_width(), 1920);
        assert_eq!(window.resolution.physical_height(), 1080);

        let window_resized_messages = app.world().resource::<Messages<WindowResized>>();
        assert_eq!(window_resized_messages.len(), 1);

        let mut window_resized_messages_iter =
            window_resized_messages.iter_current_update_messages();
        assert_eq!(
            window_resized_messages_iter.next(),
            Some(&WindowResized {
                window: window_entity,
                width: 1920.0,
                height: 1080.0
            })
        );
        assert_eq!(window_resized_messages_iter.next(), None);

        let window_event_messages = app.world().resource::<Messages<BevyWindowEvent>>();
        assert_eq!(window_event_messages.len(), 1);

        let mut window_event_messages_iter = window_event_messages.iter_current_update_messages();
        assert_eq!(
            window_event_messages_iter.next(),
            Some(&BevyWindowEvent::WindowResized(WindowResized {
                window: window_entity,
                width: 1920.0,
                height: 1080.0
            }))
        );
        assert_eq!(window_event_messages_iter.next(), None);
    }

    #[test]
    fn test_react_to_resize_with_same_size() {
        let (mut app, window_entity) =
            setup_react_to_resize(PhysicalSize::new(1280, 720), PhysicalSize::new(1280, 720));
        app.update();

        let window = app.world().get::<Window>(window_entity).unwrap();
        assert_eq!(window.resolution.physical_width(), 1280);
        assert_eq!(window.resolution.physical_height(), 720);

        let window_resized_messages = app.world().resource::<Messages<WindowResized>>();
        assert_eq!(window_resized_messages.len(), 1);

        let mut window_resized_messages_iter =
            window_resized_messages.iter_current_update_messages();
        assert_eq!(
            window_resized_messages_iter.next(),
            Some(&WindowResized {
                window: window_entity,
                width: 1280.0,
                height: 720.0
            })
        );
        assert_eq!(window_resized_messages_iter.next(), None);

        let window_event_messages = app.world().resource::<Messages<BevyWindowEvent>>();
        assert_eq!(window_event_messages.len(), 1);

        let mut window_event_messages_iter = window_event_messages.iter_current_update_messages();
        assert_eq!(
            window_event_messages_iter.next(),
            Some(&BevyWindowEvent::WindowResized(WindowResized {
                window: window_entity,
                width: 1280.0,
                height: 720.0
            }))
        );
        assert_eq!(window_event_messages_iter.next(), None);
    }

    fn setup_react_to_resize(
        initial_size: PhysicalSize<u32>,
        changed_size: PhysicalSize<u32>,
    ) -> (App, Entity) {
        let mut app = App::new();
        app.add_message::<WindowResized>();
        app.add_message::<BevyWindowEvent>();
        app.add_systems(
            Update,
            move |mut window: Single<(Entity, &mut Window)>,
                  mut window_resized_writer: MessageWriter<WindowResized>,
                  mut window_event: MessageWriter<BevyWindowEvent>| {
                let window_resized = react_to_resize(window.0, &mut window.1, changed_size);
                window_resized_writer.write(window_resized.clone());
                window_event.write(BevyWindowEvent::WindowResized(window_resized));
            },
        );

        let mut window = Window::default();
        window
            .resolution
            .set_physical_resolution(initial_size.width, initial_size.height);
        let window_entity = app.world_mut().spawn(window).id();

        (app, window_entity)
    }
}
