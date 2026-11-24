//! Window tweaks the UI toolkit has no command for.

/// Turn the window's drop shadow on or off. The translucent side panel goes
/// without one: macOS draws stale shadows around see-through windows.
#[cfg(target_os = "macos")]
pub fn set_shadow(on: bool) {
    use objc2::runtime::AnyObject;
    use objc2::{class, msg_send};
    // Called from the UI thread, which on macOS is the main thread.
    unsafe {
        let app: *mut AnyObject = msg_send![class!(NSApplication), sharedApplication];
        let windows: *mut AnyObject = msg_send![app, windows];
        let count: usize = msg_send![windows, count];
        for index in 0..count {
            let window: *mut AnyObject = msg_send![windows, objectAtIndex: index];
            let _: () = msg_send![window, setHasShadow: on];
            let _: () = msg_send![window, invalidateShadow];
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub fn set_shadow(_on: bool) {}

/// Show a folder in the platform's file manager.
pub fn reveal(folder: &std::path::Path) {
    let opener = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(target_os = "windows") {
        "explorer"
    } else {
        "xdg-open"
    };
    let _ = std::process::Command::new(opener).arg(folder).spawn();
}

/// Whether [`install_settings_menu`] provides a menu item on this platform
/// (otherwise the window needs its own way to open the settings).
pub const HAS_SETTINGS_MENU: bool = cfg!(target_os = "macos");

/// True once after each time the user chose Settings… in the menu bar.
pub fn settings_requested() -> bool {
    SETTINGS_REQUESTED.swap(false, std::sync::atomic::Ordering::Relaxed)
}

static SETTINGS_REQUESTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(target_os = "macos")]
static WAKER: std::sync::OnceLock<Box<dyn Fn() + Send + Sync>> = std::sync::OnceLock::new();

#[cfg(target_os = "macos")]
objc2::define_class!(
    // Receives the menu item's action.
    #[unsafe(super(objc2_foundation::NSObject))]
    #[name = "TWAMenuTarget"]
    struct MenuTarget;

    impl MenuTarget {
        #[unsafe(method(openSettings:))]
        fn open_settings(&self, _sender: *mut objc2::runtime::AnyObject) {
            SETTINGS_REQUESTED.store(true, std::sync::atomic::Ordering::Relaxed);
            if let Some(waker) = WAKER.get() {
                waker();
            }
        }
    }
);

/// Add "Settings…" (⌘,) to the application menu. `waker` wakes the UI so it
/// notices [`settings_requested`]. Call once, from the UI thread, after the
/// application has started.
#[cfg(target_os = "macos")]
pub fn install_settings_menu(waker: impl Fn() + Send + Sync + 'static) {
    use objc2::runtime::AnyObject;
    use objc2::{ClassType, class, msg_send, sel};
    use objc2_foundation::NSString;
    if WAKER.set(Box::new(waker)).is_err() {
        return; // already installed
    }
    unsafe {
        let app: *mut AnyObject = msg_send![class!(NSApplication), sharedApplication];
        let main_menu: *mut AnyObject = msg_send![app, mainMenu];
        if main_menu.is_null() {
            return;
        }
        let count: isize = msg_send![main_menu, numberOfItems];
        if count == 0 {
            return;
        }
        // The first menu is the application's own ("About …", …, "Quit …").
        let app_item: *mut AnyObject = msg_send![main_menu, itemAtIndex: 0isize];
        let app_menu: *mut AnyObject = msg_send![app_item, submenu];
        if app_menu.is_null() {
            return;
        }
        // The menu item doesn't retain its target; this one lives forever.
        let target: *mut AnyObject = msg_send![MenuTarget::class(), new];
        let item: *mut AnyObject = msg_send![class!(NSMenuItem), alloc];
        let item: *mut AnyObject = msg_send![
            item,
            initWithTitle: &*NSString::from_str("Settings…"),
            action: sel!(openSettings:),
            keyEquivalent: &*NSString::from_str(","),
        ];
        let _: () = msg_send![item, setTarget: target];
        let separator: *mut AnyObject = msg_send![class!(NSMenuItem), separatorItem];
        // After "About …", as its own group.
        let items: isize = msg_send![app_menu, numberOfItems];
        let at = 1.min(items);
        let _: () = msg_send![app_menu, insertItem: separator, atIndex: at];
        let _: () = msg_send![app_menu, insertItem: item, atIndex: at + 1];
    }
}

#[cfg(not(target_os = "macos"))]
pub fn install_settings_menu(_waker: impl Fn() + Send + Sync + 'static) {}
