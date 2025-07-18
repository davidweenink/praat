/* GuiDialog.cpp
 *
 * Copyright (C) 1993-2018,2020,2021,2024 Paul Boersma, 2013 Tom Naughton
 *
 * This code is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 * This code is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this work. If not, see <http://www.gnu.org/licenses/>.
 */

#include "GuiP.h"

Thing_implement (GuiDialog, GuiShell, 0);

#if gtk
	static void _GuiGtkDialog_destroyCallback (GuiObject widget, gpointer void_me) {
		(void) widget;
		iam (GuiDialog);
		forget (me);
	}
	static gboolean _GuiGtkDialog_goAwayCallback (GuiObject widget, GdkEvent *event, gpointer void_me) {
		(void) widget;
		(void) event;
		iam (GuiDialog);
		if (my d_goAwayCallback)
			my d_goAwayCallback (my d_goAwayBoss);
		return true;   // signal handled (don't destroy dialog)
	}
#elif motif
	static void _GuiMotifDialog_destroyCallback (GuiObject widget, XtPointer void_me, XtPointer call) {
		(void) widget; (void) call;
		iam (GuiDialog);
		forget (me);
	}
	static void _GuiMotifDialog_goAwayCallback (GuiObject widget, XtPointer void_me, XtPointer call) {
		(void) widget; (void) call;
		iam (GuiDialog);
		if (my d_goAwayCallback)
			my d_goAwayCallback (my d_goAwayBoss);
	}
#endif

static void gui_blocking_dialog_cb_close (GuiDialog me) {
	my clickedButtonId = 0;
	#if cocoa
		[NSApp stopModal];
	#endif
}
GuiDialog GuiDialog_create (GuiWindow parent, int x, int y, int width, int height,
	conststring32 title, GuiShell_GoAwayCallback goAwayCallback, Thing goAwayBoss, GuiDialog_Modality modality)
{
	autoGuiDialog me = Thing_new (GuiDialog);
	my d_parent = parent;
	if (modality == GuiDialog_Modality::BLOCKING) {
		Melder_assert (! goAwayCallback);
		Melder_assert (! goAwayBoss);
		my d_goAwayCallback = gui_blocking_dialog_cb_close;
		my d_goAwayBoss = me.get();
	} else {
		my d_goAwayCallback = goAwayCallback;
		my d_goAwayBoss = goAwayBoss;
	}
	#if gtk
		my d_gtkWindow = (GtkWindow *) gtk_dialog_new ();
		#if 0
			static const GdkRGBA backgroundColour { 0.92, 0.92, 0.92, 1.0 };
			gtk_widget_override_background_color (GTK_WIDGET (my d_gtkWindow), GTK_STATE_FLAG_NORMAL, & backgroundColour);
		#endif
		if (parent) {
			Melder_assert (parent -> d_widget);
			GuiObject toplevel = gtk_widget_get_ancestor (GTK_WIDGET (parent -> d_widget), GTK_TYPE_WINDOW);
			if (toplevel) {
				gtk_window_set_transient_for (GTK_WINDOW (my d_gtkWindow), GTK_WINDOW (toplevel));
				gtk_window_set_destroy_with_parent (GTK_WINDOW (my d_gtkWindow), true);
			}
		}
		g_signal_connect (G_OBJECT (my d_gtkWindow), "delete-event",
				my d_goAwayCallback ? G_CALLBACK (_GuiGtkDialog_goAwayCallback) : G_CALLBACK (gtk_widget_hide_on_delete), me.get());
		gtk_window_set_default_size (GTK_WINDOW (my d_gtkWindow), width, height);
		//gtk_window_set_modal (GTK_WINDOW (my d_gtkWindow), modality >= GuiDialog_Modality::MODAL);
		gtk_window_set_resizable (GTK_WINDOW (my d_gtkWindow), false);
		//GuiObject vbox = GTK_DIALOG (my d_gtkWindow) -> vbox;
		GuiObject vbox = gtk_dialog_get_content_area (GTK_DIALOG (my d_gtkWindow));
		my d_widget = gtk_fixed_new ();
		_GuiObject_setUserData (my d_widget, me.get());
		gtk_widget_set_size_request (GTK_WIDGET (my d_widget), width, height);
		gtk_container_add (GTK_CONTAINER (vbox /*my d_gtkWindow*/), GTK_WIDGET (my d_widget));
		gtk_widget_show (GTK_WIDGET (my d_widget));
		g_signal_connect (G_OBJECT (my d_widget), "destroy", G_CALLBACK (_GuiGtkDialog_destroyCallback), me.get());
		#if defined (chrome)
			my chrome_surrogateShellTitleLabelWidget = gtk_label_new (Melder_peek32to8 (title));
			gtk_label_set_use_markup (GTK_LABEL (my chrome_surrogateShellTitleLabelWidget), true);
			gtk_widget_set_size_request (GTK_WIDGET (my chrome_surrogateShellTitleLabelWidget), width, 31 /*Machine_getTextHeight()*/);
			gtk_misc_set_alignment (GTK_MISC (my chrome_surrogateShellTitleLabelWidget), 0.5, 0.0);
			//gtk_widget_set_xalign (GTK_WIDGET (my chrome_surrogateShellTitleLabelWidget), 0.5);
			gtk_fixed_put (GTK_FIXED (my d_widget), GTK_WIDGET (my chrome_surrogateShellTitleLabelWidget), 0 /*8*/, 0);
			gtk_widget_show (GTK_WIDGET (my chrome_surrogateShellTitleLabelWidget));
		#endif
		GuiShell_setTitle (me.get(), title);
	#elif motif
		my d_xmShell = XmCreateDialogShell (parent ? parent -> d_widget : nullptr, "dialogShell", nullptr, 0);
		XtVaSetValues (my d_xmShell, XmNdeleteResponse, my d_goAwayCallback ? XmDO_NOTHING : XmUNMAP, XmNx, x, XmNy, y, nullptr);
		if (my d_goAwayCallback)
			XmAddWMProtocolCallback (my d_xmShell, 'delw', _GuiMotifDialog_goAwayCallback, (char *) me.get());
		GuiShell_setTitle (me.get(), title);
		my d_widget = XmCreateForm (my d_xmShell, "dialog", nullptr, 0);
		XtVaSetValues (my d_widget, XmNwidth, (Dimension) width, XmNheight, (Dimension) height, nullptr);
		_GuiObject_setUserData (my d_widget, me.get());
		XtAddCallback (my d_widget, XmNdestroyCallback, _GuiMotifDialog_destroyCallback, me.get());
		XtVaSetValues (my d_widget, XmNdialogStyle,
			modality >= GuiDialog_Modality::MODAL ? XmDIALOG_FULL_APPLICATION_MODAL : XmDIALOG_MODELESS,
			XmNautoUnmanage, False, nullptr
		);
	#elif cocoa
		(void) parent;
		NSRect rect = { { (CGFloat) x, (CGFloat) y }, { (CGFloat) width, (CGFloat) height } };
		my d_cocoaShell = [[GuiCocoaShell alloc]
			initWithContentRect: rect
			styleMask: NSTitledWindowMask | NSClosableWindowMask
			backing: NSBackingStoreBuffered
			defer: false
		];
		[my d_cocoaShell   setMinSize: NSMakeSize (500.0, 500.0)];   // BUG: should not be needed
		[my d_cocoaShell   setTitle: (NSString *) Melder_peek32toCfstring (title)];
		//[my d_cocoaShell   makeKeyAndOrderFront: nil];
		my d_widget = (GuiObject) [my d_cocoaShell   contentView];
		[my d_cocoaShell   setUserData: me.get()];
		[my d_cocoaShell   setReleasedWhenClosed: NO];
	#endif
	my d_shell = me.get();
	return me.releaseToAmbiguousOwner();
}

void GuiDialog_setDefaultCallback (GuiDialog me, GuiDialog_DefaultCallback callback, Thing boss) {
	my d_defaultCallback = callback;
	my d_defaultBoss = boss;
}

static void gui_blocking_dialog_cb_default (GuiDialog me) {
	my clickedButtonId = my defaultButtonId;
	#if cocoa
		[NSApp stopModal];
	#endif
}
#if motif
std::vector <HWND> theOtherWindows;
BOOL CALLBACK enumWindowsProc (HWND window, LPARAM lParam) {
	HWND modalWindow = (HWND) lParam;
	if (window != modalWindow)
		theOtherWindows. push_back (window);
	return true;
}
#endif
integer GuiDialog_run (GuiDialog me) {
	//TRACE
	GuiDialog_setDefaultCallback (me, gui_blocking_dialog_cb_default, me);
	my clickedButtonId = -1;
	/*
		`my clickedButtonId` will be modified away from -1 in one of three ways:
		- clicking the dialog's Close button, via `gui_blocking_dialog_cb_close`, which will set it to 0
		- clicking a normal button (with text), via `gui_blocking_dialog_cb_ok`,
		  which will set it to the sequential id of the clicked button (1/2/3...)
		- typing Enter, via `gui_blocking_dialog_cb_default`,
		  which will set it to the sequential id of the default button (1/2/3...), or to 0 if there is no default button
	*/
	#if gtk
		gtk_dialog_run (GTK_DIALOG (my d_gtkWindow));
	#elif motif
		theOtherWindows. clear ();
		EnumThreadWindows (GetCurrentThreadId (), enumWindowsProc, (LPARAM) my d_xmShell -> window);
		for (HWND window: theOtherWindows)
			EnableWindow (window, false);
		UpdateWindow (my d_xmShell -> window);   // the only way to actually show the contents of the dialog (or my d_widget -> window)
		if (my defaultButton)
			SetFocus (my defaultButton -> d_widget -> window);   // otherwise, no key-down messges will be received by this window
		do {
			MSG event;
			GetMessage (& event, nullptr, 0, 0);
			if (event. message == WM_COMMAND)
				trace (event. message);
			if (event. hwnd) {
				GuiObject object = (GuiObject) GetWindowLongPtr (event. hwnd, GWLP_USERDATA);
				/*
					In case the window goes out of and into focus, the default button will no longer be in focus,
					but (fortunately) the window will continue to respond to keys.
					`IsDialogMessage` will no longer handle the Enter key, though, so we should capture the Enter key here.
					(The window will react even after the user clicks in another window, i.e. when we expect to be out of focus!)
					This is a HACK.
				*/
				if (event. message == WM_KEYDOWN && LOWORD (event. wParam) == VK_RETURN && my defaultButton) {
					my clickedButtonId = my defaultButtonId;
					break;
				}

				if (IsDialogMessage (my d_xmShell -> window, & event)) {   // not my d_widget -> window, because that would prevent closing
					trace (U"dialog message ", event. message);
				} else if (event. message == WM_PAINT) {
					trace (U"paint ", event. message);
					//TranslateMessage (& event);
					//DispatchMessage (& event);
				}
			}
		} while (my clickedButtonId == -1);
		for (HWND window: theOtherWindows)
			EnableWindow (window, true);
	#elif cocoa
		[[NSApplication sharedApplication] runModalForWindow: my d_cocoaShell];
	#endif
	return my clickedButtonId;
}

/* End of file GuiDialog.cpp */
