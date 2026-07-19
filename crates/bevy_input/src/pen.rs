//! Pen and tablet-tool input.
//!
//! Platform backends are responsible for detecting tablet tools and interpreting
//! their native data. The types in this module preserve the completed values
//! supplied by the backend; they do not normalize pressure, derive angles, or
//! synthesize mouse or touch input.

#[cfg(feature = "bevy_reflect")]
use bevy_ecs::prelude::ReflectMessage;
use bevy_ecs::{
    entity::Entity,
    message::{Message, MessageReader},
    resource::Resource,
    system::ResMut,
};
use bevy_math::Vec2;
use bevy_platform::collections::{HashMap, HashSet};
#[cfg(feature = "bevy_reflect")]
use bevy_reflect::Reflect;

use crate::ButtonState;

#[cfg(all(feature = "serialize", feature = "bevy_reflect"))]
use bevy_reflect::{ReflectDeserialize, ReflectSerialize};

/// Stable identity of the tablet device that produced a pen event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Debug, Hash, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub enum PenId {
    /// A backend-provided device identifier.
    Device(i64),
    /// The backend could not identify the device.
    Unidentified,
}

/// The kind of physical tablet tool being used.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Debug, Hash, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub enum PenToolKind {
    /// A general-purpose pen or stylus.
    #[default]
    Pen,
    /// The eraser end of a stylus.
    Eraser,
    /// A brush tool.
    Brush,
    /// A pencil tool.
    Pencil,
    /// An airbrush tool.
    Airbrush,
    /// A finger reported through a tablet-tool interface.
    Finger,
    /// A mouse-shaped tablet tool.
    Mouse,
    /// A lens cursor.
    Lens,
    /// A tool kind not known by this version of Bevy.
    Unknown,
}

/// A button on a pen or tablet tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Debug, Hash, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub enum PenButton {
    /// Contact between the tool tip and tablet surface.
    Contact,
    /// The primary barrel or side button.
    Barrel,
    /// A backend-defined additional button.
    Other(u16),
}

/// Pressure reported by the tablet backend.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Debug, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub enum PenPressure {
    /// Calibrated force and the device's maximum possible force.
    Calibrated {
        /// Force applied along the tool's axis.
        force: f64,
        /// Maximum force the device can report.
        max_possible_force: f64,
    },
    /// Backend-normalized force, conventionally in the inclusive range `0.0..=1.0`.
    Normalized(f64),
}

/// Plane tilt in degrees, as reported by the tablet backend.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Debug, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct PenTilt {
    /// Tilt on the x axis in degrees.
    pub x: i8,
    /// Tilt on the y axis in degrees.
    pub y: i8,
}

/// Angular position of a pen, in radians.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Debug, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct PenAngle {
    /// Angle above the tablet surface.
    pub altitude: f64,
    /// Clockwise angle around the surface's normal axis.
    pub azimuth: f64,
}

/// Analog data supplied for a tablet-tool event.
#[derive(Debug, Default, Clone, PartialEq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Debug, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct PenData {
    /// Force applied against the tablet surface.
    pub pressure: Option<PenPressure>,
    /// Tangential (barrel) pressure in the range `-1.0..=1.0`.
    pub tangential_pressure: Option<f32>,
    /// Clockwise rotation around the tool's major axis, in degrees `0..=359`.
    pub twist: Option<u16>,
    /// Plane tilt in degrees.
    pub tilt: Option<PenTilt>,
    /// Angular position in radians.
    pub angle: Option<PenAngle>,
}

/// Information common to every pen event.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Debug, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct PenInfo {
    /// Window that received the event.
    pub window: Entity,
    /// Tablet device that produced the event.
    pub device: PenId,
    /// Whether this is the primary pointer.
    pub primary: bool,
    /// Position in logical window pixels, if supplied by the backend.
    pub position: Option<Vec2>,
    /// Kind of tool being used.
    pub tool: PenToolKind,
}

/// Change represented by a [`PenInput`] event.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Debug, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub enum PenAction {
    /// The tool entered sensing range over a window.
    Entered,
    /// The tool moved or its analog data changed.
    Moved(PenData),
    /// A tool button changed state.
    Button {
        /// Button that changed.
        button: PenButton,
        /// New button state.
        state: ButtonState,
        /// Analog data sampled with the button event.
        data: PenData,
    },
    /// The tool left sensing range over a window.
    Left,
}

/// A first-class pen or tablet-tool input event.
#[derive(Message, Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Debug, PartialEq, Clone, Message)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct PenInput {
    /// Device, window, tool, and position information.
    pub pen: PenInfo,
    /// Change represented by this event.
    pub action: PenAction,
}

/// Latest derived state for a pen device.
#[derive(Debug, Clone)]
pub struct Pen {
    id: PenId,
    window: Entity,
    primary: bool,
    position: Option<Vec2>,
    tool: PenToolKind,
    data: Option<PenData>,
    pressed_buttons: HashSet<PenButton>,
    in_range: bool,
}

impl Pen {
    /// Returns the tablet device identity.
    pub fn id(&self) -> PenId {
        self.id
    }

    /// Returns the window currently associated with this pen.
    pub fn window(&self) -> Entity {
        self.window
    }

    /// Returns whether this pen is the primary pointer.
    pub fn is_primary(&self) -> bool {
        self.primary
    }

    /// Returns its latest logical position, if known.
    pub fn position(&self) -> Option<Vec2> {
        self.position
    }

    /// Returns the physical tool kind.
    pub fn tool(&self) -> PenToolKind {
        self.tool
    }

    /// Returns the latest analog sample, if one has been received.
    pub fn data(&self) -> Option<&PenData> {
        self.data.as_ref()
    }

    /// Returns whether the tool is in sensing range over its window.
    pub fn is_in_range(&self) -> bool {
        self.in_range
    }

    /// Returns whether a tool button is currently pressed.
    pub fn pressed(&self, button: PenButton) -> bool {
        self.pressed_buttons.contains(&button)
    }

    /// Iterates over the currently pressed tool buttons.
    pub fn pressed_buttons(&self) -> impl Iterator<Item = PenButton> + '_ {
        self.pressed_buttons.iter().copied()
    }
}

/// Derived pen state, keyed by tablet device identity.
#[derive(Debug, Default, Resource)]
pub struct Pens {
    pens: HashMap<PenId, Pen>,
}

impl Pens {
    /// Returns state for a tablet device.
    pub fn get(&self, id: PenId) -> Option<&Pen> {
        self.pens.get(&id)
    }

    /// Iterates over all tablet devices seen by the application.
    pub fn iter(&self) -> impl Iterator<Item = &Pen> {
        self.pens.values()
    }
}

/// Updates [`Pens`] from the ordered [`PenInput`] stream.
pub(crate) fn pen_input_system(mut events: MessageReader<PenInput>, mut pens: ResMut<Pens>) {
    for event in events.read() {
        let info = event.pen;
        let pen = pens.pens.entry(info.device).or_insert_with(|| Pen {
            id: info.device,
            window: info.window,
            primary: info.primary,
            position: info.position,
            tool: info.tool,
            data: None,
            pressed_buttons: HashSet::new(),
            in_range: false,
        });

        pen.window = info.window;
        pen.primary = info.primary;
        pen.position = info.position;
        pen.tool = info.tool;

        match &event.action {
            PenAction::Entered => pen.in_range = true,
            PenAction::Moved(data) => {
                pen.in_range = true;
                pen.data = Some(data.clone());
            }
            PenAction::Button {
                button,
                state,
                data,
            } => {
                pen.in_range = true;
                pen.data = Some(data.clone());
                if state.is_pressed() {
                    pen.pressed_buttons.insert(*button);
                } else {
                    pen.pressed_buttons.remove(button);
                }
            }
            PenAction::Left => {
                pen.in_range = false;
                pen.pressed_buttons.clear();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_app::{App, PreUpdate};

    #[test]
    fn state_preserves_completed_pen_values() {
        let mut app = App::new();
        app.add_message::<PenInput>()
            .init_resource::<Pens>()
            .add_systems(PreUpdate, pen_input_system);

        let id = PenId::Device(7);
        let data = PenData {
            pressure: Some(PenPressure::Normalized(0.625)),
            tangential_pressure: Some(-0.25),
            twist: Some(271),
            tilt: Some(PenTilt { x: -17, y: 42 }),
            angle: Some(PenAngle {
                altitude: 0.75,
                azimuth: 1.25,
            }),
        };
        app.world_mut().write_message(PenInput {
            pen: PenInfo {
                window: Entity::PLACEHOLDER,
                device: id,
                primary: true,
                position: Some(Vec2::new(12.5, 24.0)),
                tool: PenToolKind::Pen,
            },
            action: PenAction::Moved(data.clone()),
        });
        app.world_mut().write_message(PenInput {
            pen: PenInfo {
                window: Entity::PLACEHOLDER,
                device: id,
                primary: true,
                position: Some(Vec2::new(12.5, 24.0)),
                tool: PenToolKind::Pen,
            },
            action: PenAction::Button {
                button: PenButton::Barrel,
                state: ButtonState::Pressed,
                data: data.clone(),
            },
        });

        app.update();

        {
            let pens = app.world().resource::<Pens>();
            let pen = pens.get(id).unwrap();
            assert_eq!(pen.data(), Some(&data));
            assert_eq!(pen.position(), Some(Vec2::new(12.5, 24.0)));
            assert!(pen.is_in_range());
            assert!(pen.pressed(PenButton::Barrel));
        }

        app.world_mut().write_message(PenInput {
            pen: PenInfo {
                window: Entity::PLACEHOLDER,
                device: id,
                primary: true,
                position: None,
                tool: PenToolKind::Pen,
            },
            action: PenAction::Left,
        });
        app.update();

        let pens = app.world().resource::<Pens>();
        let pen = pens.get(id).unwrap();
        assert!(!pen.is_in_range());
        assert!(!pen.pressed(PenButton::Barrel));
        assert_eq!(pen.data(), Some(&data));
    }
}
