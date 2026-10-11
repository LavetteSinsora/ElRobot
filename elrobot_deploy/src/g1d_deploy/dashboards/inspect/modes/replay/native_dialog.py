"""The desktop's file dialog, as a script for the SYSTEM Python.

    /usr/bin/python3 native_dialog.py deploy|ckpt START

Run by `browse.native_pick` (the replay panel's Browse buttons). It is a
separate script, not a function, because the GTK bindings (`gi`) are a system
package: g1d_deploy's venv does not have them, the system interpreter does.
It imports nothing from g1d_deploy.

`deploy` picks a file, with the list filtered to `*.toml` (and folders);
`ckpt` picks a folder. `START` is the path currently in the field: the dialog
opens in its folder with it selected. Prints the chosen path and exits 0;
exits 1 when cancelled, 2 when no dialog could be shown.

For a user: `docs/deploying_a_policy/rollout_inspection.md`, "policy
checkpoint (`replay`)".
"""

import os
import sys


def main(argv) -> int:
    try:
        import gi
        gi.require_version("Gtk", "3.0")
        from gi.repository import GLib, Gtk
    except Exception as exc:                      # no bindings, no display
        print(f"no GTK: {exc}", file=sys.stderr)
        return 2
    what = argv[1] if len(argv) > 1 else "ckpt"
    start = os.path.expanduser(argv[2]) if len(argv) > 2 and argv[2] else os.path.expanduser("~")
    folder = what != "deploy"
    ok, _ = Gtk.init_check(None)
    if not ok:
        print("no display", file=sys.stderr)
        return 2

    dlg = Gtk.FileChooserDialog(
        title="Pick a checkpoint folder" if folder else "Pick a deploy.toml",
        action=Gtk.FileChooserAction.SELECT_FOLDER if folder else Gtk.FileChooserAction.OPEN)
    dlg.add_buttons("Cancel", Gtk.ResponseType.CANCEL, "Select", Gtk.ResponseType.OK)
    dlg.set_default_response(Gtk.ResponseType.OK)
    if not folder:
        toml = Gtk.FileFilter()
        toml.set_name("deploy files (*.toml)")
        toml.add_pattern("*.toml")
        dlg.add_filter(toml)
        everything = Gtk.FileFilter()
        everything.set_name("all files")
        everything.add_pattern("*")
        dlg.add_filter(everything)
        dlg.set_filter(toml)                      # the one shown when it opens

    # Open where the field points: inside its folder, with it selected.
    here = start
    while here and not os.path.exists(here) and os.path.dirname(here) != here:
        here = os.path.dirname(here)
    if here == start and (os.path.isfile(here) or folder):
        dlg.set_current_folder(os.path.dirname(here.rstrip("/")) or "/")
        dlg.select_filename(here)
    else:
        dlg.set_current_folder(here if os.path.isdir(here) else os.path.dirname(here))
    dlg.set_keep_above(True)

    # A test hook: close by itself, so the script can be exercised unattended.
    auto = os.environ.get("G1D_DIALOG_AUTOCLOSE_MS")
    if auto:
        GLib.timeout_add(int(auto), lambda: dlg.response(Gtk.ResponseType.CANCEL) or False)

    response = dlg.run()
    path = dlg.get_filename()
    dlg.destroy()
    while Gtk.events_pending():
        Gtk.main_iteration()
    if response != Gtk.ResponseType.OK or not path:
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
