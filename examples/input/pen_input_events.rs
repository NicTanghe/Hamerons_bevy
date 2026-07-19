//! Prints first-class pen events and the raw winit events they came from.
//!
//! This example is deliberately only a diagnostic adapter: winit performs tablet
//! detection and produces pressure, tilt, angle, twist, and button data. Bevy
//! prints those completed values without recalculating them or turning them into
//! mouse or touch input.

use std::collections::VecDeque;

use bevy::{
    input::pen::{PenAction, PenData, PenInput},
    prelude::*,
    window::{WindowEvent, WindowPlugin},
    winit::{
        RawWinitWindowEvent, WinitButtonSource, WinitPointerKind, WinitPointerSource,
        WinitSettings, WinitWindowEvent,
    },
};

const MAX_SCREEN_LOG_LINES: usize = 14;

fn main() {
    App::new()
        // A visible X11 window without a renderer contains undefined pixels. A
        // minimal clear pass makes it obvious that this diagnostic is alive.
        .insert_resource(ClearColor(Color::srgb(0.025, 0.035, 0.055)))
        // Pen input is window-event driven, so there is no need to redraw as
        // fast as possible while waiting for the next tablet sample.
        .insert_resource(WinitSettings::desktop_app())
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bevy Pen Input Diagnostic".into(),
                ..default()
            }),
            ..default()
        }))
        .init_resource::<DiagnosticCounts>()
        .init_resource::<DiagnosticLog>()
        .add_systems(Startup, setup_diagnostic_window)
        .add_systems(
            Update,
            (pen_diagnostic_system, update_diagnostic_text).chain(),
        )
        .run();
}

fn setup_diagnostic_window(mut commands: Commands) {
    commands.spawn(Camera2d);
    commands.spawn((
        DiagnosticText,
        Text::new("Starting Pen diagnostic…"),
        TextFont::from_font_size(18.0),
        TextColor(Color::srgb(0.88, 0.95, 1.0)),
        TextShadow::default(),
        Node {
            position_type: PositionType::Absolute,
            top: px(18),
            left: px(22),
            right: px(22),
            ..default()
        },
    ));
    info!(
        "pen diagnostic ready: move the tablet tool over the blue window, then test pressure, tilt, buttons, and the eraser"
    );
}

#[derive(Component)]
struct DiagnosticText;

#[derive(Default, Resource)]
struct DiagnosticCounts {
    raw_tablet: u64,
    pen: u64,
    cursor: u64,
    mouse: u64,
    touch: u64,
}

#[derive(Resource)]
struct DiagnosticLog {
    latest_sample: String,
    lines: VecDeque<String>,
    isolation_warning: bool,
}

impl Default for DiagnosticLog {
    fn default() -> Self {
        Self {
            latest_sample: "Waiting for a tablet tool…".into(),
            lines: VecDeque::from(["READY  Hover the pen over this window.".into()]),
            isolation_warning: false,
        }
    }
}

impl DiagnosticLog {
    fn push(&mut self, line: String) {
        self.lines.push_back(line);
        if self.lines.len() > MAX_SCREEN_LOG_LINES {
            self.lines.pop_front();
        }
    }

    fn screen_text(&self, counts: &DiagnosticCounts) -> String {
        let isolation = if self.isolation_warning {
            "WARNING: Pen and non-Pen pointer messages occurred in one update."
        } else {
            "OK so far (use a normal mouse separately to verify both still work)."
        };
        let lines = self.lines.iter().cloned().collect::<Vec<_>>().join("\n");

        format!(
            "BEVY X11 PEN DIAGNOSTIC\n\
             Hover, draw with varying pressure/tilt, press barrel buttons, then use the eraser.\n\n\
             COUNTS  raw tablet: {}   Bevy Pen: {}   cursor: {}   mouse: {}   touch: {}\n\
             ISOLATION  {isolation}\n\n\
             LATEST SAMPLE\n{}\n\n\
             EVENT LOG (oldest → newest)\n{}",
            counts.raw_tablet,
            counts.pen,
            counts.cursor,
            counts.mouse,
            counts.touch,
            self.latest_sample,
            lines,
        )
    }
}

fn pen_diagnostic_system(
    mut raw_events: MessageReader<RawWinitWindowEvent>,
    mut pen_events: MessageReader<PenInput>,
    mut window_events: MessageReader<WindowEvent>,
    mut counts: ResMut<DiagnosticCounts>,
    mut screen_log: ResMut<DiagnosticLog>,
) {
    let mut raw_tablet_this_update = 0;
    for event in raw_events.read() {
        if is_raw_tablet_event(&event.event) {
            raw_tablet_this_update += 1;
            debug!(?event, "raw winit tablet event");
        }
    }

    let mut pen_this_update = 0;
    for event in pen_events.read() {
        pen_this_update += 1;
        record_pen_event(event, &mut screen_log);
    }

    let mut cursor_this_update = 0;
    let mut mouse_this_update = 0;
    let mut touch_this_update = 0;
    for event in window_events.read() {
        match event {
            WindowEvent::CursorEntered(_)
            | WindowEvent::CursorLeft(_)
            | WindowEvent::CursorMoved(_) => cursor_this_update += 1,
            WindowEvent::MouseButtonInput(_) | WindowEvent::MouseMotion(_) => {
                mouse_this_update += 1;
            }
            WindowEvent::TouchInput(_) => touch_this_update += 1,
            _ => {}
        }
    }

    counts.raw_tablet += raw_tablet_this_update;
    counts.pen += pen_this_update;
    counts.cursor += cursor_this_update;
    counts.mouse += mouse_this_update;
    counts.touch += touch_this_update;

    if pen_this_update > 0 {
        debug!(
            raw_tablet = counts.raw_tablet,
            pen = counts.pen,
            cursor = counts.cursor,
            mouse = counts.mouse,
            touch = counts.touch,
            "tablet diagnostic totals"
        );

        if cursor_this_update + mouse_this_update + touch_this_update > 0
            && !screen_log.isolation_warning
        {
            screen_log.isolation_warning = true;
            screen_log.push("WARNING  Pen coincided with non-Pen pointer output.".into());
            warn!(
                cursor_this_update,
                mouse_this_update,
                touch_this_update,
                "pen and non-pen pointer messages arrived in the same update; verify that a separate mouse or touch was not used simultaneously"
            );
        }
    }
}

fn is_raw_tablet_event(event: &WinitWindowEvent) -> bool {
    matches!(
        event,
        WinitWindowEvent::PointerEntered {
            kind: WinitPointerKind::TabletTool(_),
            ..
        } | WinitWindowEvent::PointerMoved {
            source: WinitPointerSource::TabletTool { .. },
            ..
        } | WinitWindowEvent::PointerButton {
            button: WinitButtonSource::TabletTool { .. },
            ..
        } | WinitWindowEvent::PointerLeft {
            kind: WinitPointerKind::TabletTool(_),
            ..
        }
    )
}

fn record_pen_event(event: &PenInput, screen_log: &mut DiagnosticLog) {
    let pen = event.pen;
    match &event.action {
        PenAction::Entered => {
            info!(
                device = ?pen.device,
                tool = ?pen.tool,
                primary = pen.primary,
                position = ?pen.position,
                "pen entered"
            );
            screen_log.push(format!(
                "ENTER  device={:?} tool={:?} primary={} position={:?}",
                pen.device, pen.tool, pen.primary, pen.position
            ));
        }
        PenAction::Moved(data) => {
            debug!(
                device = ?pen.device,
                tool = ?pen.tool,
                primary = pen.primary,
                position = ?pen.position,
                "pen moved"
            );
            debug_pen_data(data);
            screen_log.latest_sample = format!(
                "MOVE  device={:?} tool={:?} position={:?}\n{}",
                pen.device,
                pen.tool,
                pen.position,
                pen_data_text(data)
            );
        }
        PenAction::Button {
            button,
            state,
            data,
        } => {
            info!(
                device = ?pen.device,
                tool = ?pen.tool,
                button = ?button,
                state = ?state,
                position = ?pen.position,
                "pen button"
            );
            debug_pen_data(data);
            let line = format!(
                "BUTTON  {:?} {:?} tool={:?} position={:?}",
                button, state, pen.tool, pen.position
            );
            screen_log.push(line);
            screen_log.latest_sample = format!(
                "BUTTON  device={:?} tool={:?} {:?} {:?}\n{}",
                pen.device,
                pen.tool,
                button,
                state,
                pen_data_text(data)
            );
        }
        PenAction::Left => {
            info!(
                device = ?pen.device,
                tool = ?pen.tool,
                primary = pen.primary,
                position = ?pen.position,
                "pen left"
            );
            screen_log.push(format!(
                "LEFT  device={:?} tool={:?} position={:?}",
                pen.device, pen.tool, pen.position
            ));
        }
    }
}

fn debug_pen_data(data: &PenData) {
    debug!(
        pressure = ?data.pressure,
        tangential_pressure = ?data.tangential_pressure,
        twist = ?data.twist,
        tilt = ?data.tilt,
        angle = ?data.angle,
        "winit-provided pen sample"
    );
}

fn pen_data_text(data: &PenData) -> String {
    format!(
        "pressure={:?}  tangential={:?}  twist={:?}\ntilt={:?}  angle={:?}",
        data.pressure, data.tangential_pressure, data.twist, data.tilt, data.angle
    )
}

fn update_diagnostic_text(
    counts: Res<DiagnosticCounts>,
    screen_log: Res<DiagnosticLog>,
    mut text: Single<&mut Text, With<DiagnosticText>>,
) {
    text.0 = screen_log.screen_text(&counts);
}
