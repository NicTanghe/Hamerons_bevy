//! This module provides unsurprising default inputs to `bevy_picking` through [`PointerInput`].
//! The included systems are responsible for sending mouse, touch, and pen inputs to their
//! respective `Pointer`s.
//!
//! Because this has it's own plugin, it's easy to omit it, and provide your own inputs as
//! needed. Because `Pointer`s aren't coupled to the underlying input hardware, you can easily mock
//! inputs, and allow users full accessibility to map whatever inputs they need to pointer input.
//!
//! If, for example, you wanted to add support for VR input, all you need to do is spawn a pointer
//! entity with a custom [`PointerId`], and write a system
//! that updates its position. If you want this to work properly with the existing interaction events,
//! you need to be sure that you also write a [`PointerInput`] event stream.

use bevy_app::prelude::*;
use bevy_camera::RenderTarget;
use bevy_ecs::prelude::*;
use bevy_input::{
    mouse::MouseWheel,
    pen::{PenAction, PenButton, PenId, PenInfo, PenInput},
    prelude::*,
    touch::{TouchInput, TouchPhase},
    ButtonState,
};
use bevy_math::Vec2;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_reflect::prelude::*;
use bevy_window::{PrimaryWindow, WindowEvent, WindowRef};
use tracing::debug;

use crate::pointer::{
    Location, PointerAction, PointerButton, PointerId, PointerInput, PointerLocation,
};

use crate::PickingSystems;

/// The picking input prelude.
///
/// This includes the most common types in this module, re-exported for your convenience.
pub mod prelude {
    pub use crate::input::PointerInputPlugin;
}

#[derive(Copy, Clone, Resource, Debug, Reflect)]
#[reflect(Resource, Default, Clone)]
/// Settings for enabling and disabling mouse, touch, and pen inputs for picking.
///
/// ## Custom initialization
/// ```
/// # use bevy_app::App;
/// # use bevy_picking::input::{PointerInputSettings,PointerInputPlugin};
/// App::new()
///     .insert_resource(PointerInputSettings {
///         is_touch_enabled: false,
///         is_mouse_enabled: true,
///         is_pen_enabled: true,
///     })
///     // or DefaultPlugins
///     .add_plugins(PointerInputPlugin);
/// ```
pub struct PointerInputSettings {
    /// Should touch inputs be updated?
    pub is_touch_enabled: bool,
    /// Should mouse inputs be updated?
    pub is_mouse_enabled: bool,
    /// Should pen and tablet-tool inputs be updated?
    pub is_pen_enabled: bool,
}

impl PointerInputSettings {
    fn is_mouse_enabled(state: Res<Self>) -> bool {
        state.is_mouse_enabled
    }

    fn is_touch_enabled(state: Res<Self>) -> bool {
        state.is_touch_enabled
    }

    fn is_pen_enabled(state: Res<Self>) -> bool {
        state.is_pen_enabled
    }
}

impl Default for PointerInputSettings {
    fn default() -> Self {
        Self {
            is_touch_enabled: true,
            is_mouse_enabled: true,
            is_pen_enabled: true,
        }
    }
}

/// Adds mouse, touch, and pen inputs for picking pointers to your app. This is a default input plugin,
/// that you can replace with your own plugin as needed.
///
/// Toggling mouse, touch, or pen input can be done at runtime by modifying the
/// [`PointerInputSettings`] resource.
///
/// [`PointerInputSettings`] can be initialized with custom values, but will be
/// initialized with default values if it is not present at the moment this is
/// added to the app.
pub struct PointerInputPlugin;

impl Plugin for PointerInputPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PointerInputSettings>()
            .add_systems(Startup, spawn_mouse_pointer)
            .add_systems(
                First,
                (
                    mouse_pick_events.run_if(PointerInputSettings::is_mouse_enabled),
                    touch_pick_events.run_if(PointerInputSettings::is_touch_enabled),
                    pen_pick_events.run_if(PointerInputSettings::is_pen_enabled),
                )
                    .chain()
                    .in_set(PickingSystems::Input),
            )
            .add_systems(
                Last,
                (
                    deactivate_touch_pointers.run_if(PointerInputSettings::is_touch_enabled),
                    deactivate_pen_pointers.run_if(PointerInputSettings::is_pen_enabled),
                ),
            );
    }
}

/// Sends pen pointer events to be consumed by the core picking plugin.
pub fn pen_pick_events(
    mut window_events: MessageReader<WindowEvent>,
    primary_window: Query<Entity, With<PrimaryWindow>>,
    mut pen_cache: Local<HashMap<PenId, PenInfo>>,
    mut commands: Commands,
    mut pointer_inputs: MessageWriter<PointerInput>,
) {
    for window_event in window_events.read() {
        let WindowEvent::PenInput(input) = window_event else {
            continue;
        };

        let pointer = PointerId::Pen(input.pen.device);
        let position = input.pen.position.or_else(|| {
            pen_cache
                .get(&input.pen.device)
                .and_then(|pen| pen.position)
        });
        let Some(position) = position else {
            if matches!(input.action, PenAction::Left) {
                pen_cache.remove(&input.pen.device);
            }
            continue;
        };
        let location = Location {
            target: match RenderTarget::Window(WindowRef::Entity(input.pen.window))
                .normalize(primary_window.single().ok())
            {
                Some(target) => target,
                None => continue,
            },
            position,
        };

        match &input.action {
            PenAction::Entered => {
                debug!("Spawning pen pointer {:?}", pointer);
                if pen_cache.insert(input.pen.device, input.pen).is_none() {
                    commands.spawn((pointer, PointerLocation::new(location)));
                }
            }
            PenAction::Moved(_) => {
                let delta = pen_cache
                    .get(&input.pen.device)
                    .and_then(|pen| pen.position)
                    .map_or(Vec2::ZERO, |last| position - last);
                pointer_inputs.write(PointerInput::new(
                    pointer,
                    location,
                    PointerAction::Move { delta },
                ));
                pen_cache.insert(input.pen.device, input.pen);
            }
            PenAction::Button { button, state, .. } => {
                let button = match button {
                    PenButton::Contact => PointerButton::Primary,
                    PenButton::Barrel => PointerButton::Secondary,
                    PenButton::Other(1) => PointerButton::Middle,
                    PenButton::Other(_) => continue,
                };
                let action = match state {
                    ButtonState::Pressed => PointerAction::Press(button),
                    ButtonState::Released => PointerAction::Release(button),
                };
                pointer_inputs.write(PointerInput::new(pointer, location, action));
                pen_cache.insert(input.pen.device, input.pen);
            }
            PenAction::Left => {
                pointer_inputs.write(PointerInput::new(pointer, location, PointerAction::Cancel));
                pen_cache.remove(&input.pen.device);
            }
        }
    }
}

/// Spawns the default mouse pointer.
pub fn spawn_mouse_pointer(mut commands: Commands) {
    commands.spawn(PointerId::Mouse);
}

/// Sends mouse pointer events to be processed by the core plugin
pub fn mouse_pick_events(
    // Input
    mut window_events: MessageReader<WindowEvent>,
    primary_window: Query<Entity, With<PrimaryWindow>>,
    // Locals
    mut cursor_last: Local<Vec2>,
    // Output
    mut pointer_inputs: MessageWriter<PointerInput>,
) {
    for window_event in window_events.read() {
        match window_event {
            // Handle cursor movement events
            WindowEvent::CursorMoved(event) => {
                let location = Location {
                    target: match RenderTarget::Window(WindowRef::Entity(event.window))
                        .normalize(primary_window.single().ok())
                    {
                        Some(target) => target,
                        None => continue,
                    },
                    position: event.position,
                };
                pointer_inputs.write(PointerInput::new(
                    PointerId::Mouse,
                    location,
                    PointerAction::Move {
                        delta: event.position - *cursor_last,
                    },
                ));
                *cursor_last = event.position;
            }
            // Handle mouse button press events
            WindowEvent::MouseButtonInput(input) => {
                let location = Location {
                    target: match RenderTarget::Window(WindowRef::Entity(input.window))
                        .normalize(primary_window.single().ok())
                    {
                        Some(target) => target,
                        None => continue,
                    },
                    position: *cursor_last,
                };
                let button = match input.button {
                    MouseButton::Left => PointerButton::Primary,
                    MouseButton::Right => PointerButton::Secondary,
                    MouseButton::Middle => PointerButton::Middle,
                    MouseButton::Other(_) | MouseButton::Back | MouseButton::Forward => continue,
                };
                let action = match input.state {
                    ButtonState::Pressed => PointerAction::Press(button),
                    ButtonState::Released => PointerAction::Release(button),
                };
                pointer_inputs.write(PointerInput::new(PointerId::Mouse, location, action));
            }
            WindowEvent::MouseWheel(event) => {
                let MouseWheel {
                    unit,
                    x,
                    y,
                    window,
                    phase,
                } = *event;

                let location = Location {
                    target: match RenderTarget::Window(WindowRef::Entity(window))
                        .normalize(primary_window.single().ok())
                    {
                        Some(target) => target,
                        None => continue,
                    },
                    position: *cursor_last,
                };

                let action = PointerAction::Scroll { x, y, unit, phase };

                pointer_inputs.write(PointerInput::new(PointerId::Mouse, location, action));
            }
            _ => {}
        }
    }
}

/// Sends touch pointer events to be consumed by the core plugin
pub fn touch_pick_events(
    // Input
    mut window_events: MessageReader<WindowEvent>,
    primary_window: Query<Entity, With<PrimaryWindow>>,
    // Locals
    mut touch_cache: Local<HashMap<u64, TouchInput>>,
    // Output
    mut commands: Commands,
    mut pointer_inputs: MessageWriter<PointerInput>,
) {
    for window_event in window_events.read() {
        if let WindowEvent::TouchInput(touch) = window_event {
            let pointer = PointerId::Touch(touch.id);
            let location = Location {
                target: match RenderTarget::Window(WindowRef::Entity(touch.window))
                    .normalize(primary_window.single().ok())
                {
                    Some(target) => target,
                    None => continue,
                },
                position: touch.position,
            };
            match touch.phase {
                TouchPhase::Started => {
                    debug!("Spawning pointer {:?}", pointer);
                    commands.spawn((pointer, PointerLocation::new(location.clone())));

                    pointer_inputs.write(PointerInput::new(
                        pointer,
                        location,
                        PointerAction::Press(PointerButton::Primary),
                    ));

                    touch_cache.insert(touch.id, *touch);
                }
                TouchPhase::Moved => {
                    // Send a move event only if it isn't the same as the last one
                    if let Some(last_touch) = touch_cache.get(&touch.id) {
                        if last_touch == touch {
                            continue;
                        }
                        pointer_inputs.write(PointerInput::new(
                            pointer,
                            location,
                            PointerAction::Move {
                                delta: touch.position - last_touch.position,
                            },
                        ));
                    }
                    touch_cache.insert(touch.id, *touch);
                }
                TouchPhase::Ended => {
                    pointer_inputs.write(PointerInput::new(
                        pointer,
                        location,
                        PointerAction::Release(PointerButton::Primary),
                    ));
                    touch_cache.remove(&touch.id);
                }
                TouchPhase::Canceled => {
                    pointer_inputs.write(PointerInput::new(
                        pointer,
                        location,
                        PointerAction::Cancel,
                    ));
                    touch_cache.remove(&touch.id);
                }
            }
        }
    }
}

/// Deactivates unused touch pointers.
///
/// Because each new touch gets assigned a new ID, we need to remove the pointers associated with
/// touches that are no longer active.
pub fn deactivate_touch_pointers(
    mut commands: Commands,
    mut despawn_list: Local<HashSet<(Entity, PointerId)>>,
    pointers: Query<(Entity, &PointerId)>,
    mut touches: MessageReader<TouchInput>,
) {
    for touch in touches.read() {
        if let TouchPhase::Ended | TouchPhase::Canceled = touch.phase {
            for (entity, pointer) in &pointers {
                if pointer.get_touch_id() == Some(touch.id) {
                    despawn_list.insert((entity, *pointer));
                }
            }
        }
    }
    // A hash set is used to prevent despawning the same entity twice.
    for (entity, pointer) in despawn_list.drain() {
        debug!("Despawning pointer {:?}", pointer);
        commands.entity(entity).despawn();
    }
}

/// Despawns pen pointers after their tool leaves sensing range.
pub fn deactivate_pen_pointers(
    mut commands: Commands,
    pointers: Query<(Entity, &PointerId)>,
    mut pens: MessageReader<PenInput>,
) {
    for pen in pens.read() {
        if !matches!(pen.action, PenAction::Left) {
            continue;
        }
        for (entity, pointer) in &pointers {
            if pointer.get_pen_id() == Some(pen.pen.device) {
                debug!("Despawning pen pointer {:?}", pointer);
                commands.entity(entity).despawn();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::message::Messages;
    use bevy_input::pen::{PenData, PenToolKind};
    use bevy_window::Window;

    #[test]
    fn pen_window_events_only_create_pen_pointer_input() {
        let mut app = App::new();
        app.add_message::<WindowEvent>()
            .add_message::<PointerInput>()
            .add_systems(
                First,
                (mouse_pick_events, touch_pick_events, pen_pick_events).chain(),
            );

        let window = app
            .world_mut()
            .spawn((Window::default(), PrimaryWindow))
            .id();
        let device = PenId::Device(42);
        let pen = PenInfo {
            window,
            device,
            primary: true,
            position: Some(Vec2::new(10.0, 20.0)),
            tool: PenToolKind::Pen,
        };
        app.world_mut()
            .write_message(WindowEvent::PenInput(PenInput {
                pen,
                action: PenAction::Entered,
            }));
        app.world_mut()
            .write_message(WindowEvent::PenInput(PenInput {
                pen: PenInfo {
                    position: Some(Vec2::new(15.0, 27.0)),
                    ..pen
                },
                action: PenAction::Moved(PenData::default()),
            }));
        app.world_mut()
            .write_message(WindowEvent::PenInput(PenInput {
                pen,
                action: PenAction::Button {
                    button: PenButton::Contact,
                    state: ButtonState::Pressed,
                    data: PenData::default(),
                },
            }));
        app.world_mut()
            .write_message(WindowEvent::PenInput(PenInput {
                pen,
                action: PenAction::Button {
                    button: PenButton::Contact,
                    state: ButtonState::Released,
                    data: PenData::default(),
                },
            }));
        app.world_mut()
            .write_message(WindowEvent::PenInput(PenInput {
                pen: PenInfo {
                    position: None,
                    ..pen
                },
                action: PenAction::Left,
            }));

        app.update();

        let mut pointer_query = app.world_mut().query::<&PointerId>();
        let pointers: Vec<_> = pointer_query.iter(app.world()).copied().collect();
        assert_eq!(pointers, vec![PointerId::Pen(device)]);

        let events = app.world().resource::<Messages<PointerInput>>();
        let mut cursor = events.get_cursor();
        let inputs: Vec<_> = cursor.read(events).collect();
        assert_eq!(inputs.len(), 4);
        assert!(inputs
            .iter()
            .all(|input| input.pointer_id == PointerId::Pen(device)));
        assert!(inputs
            .iter()
            .all(|input| !input.pointer_id.is_mouse() && !input.pointer_id.is_touch()));
        assert!(matches!(
            inputs[1].action,
            PointerAction::Press(PointerButton::Primary)
        ));
        assert!(matches!(
            inputs[2].action,
            PointerAction::Release(PointerButton::Primary)
        ));
        assert!(matches!(inputs[3].action, PointerAction::Cancel));
    }
}
