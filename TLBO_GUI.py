"""
TLBO_GUI_auto_dark.py

Dark-mode Tkinter GUI that automatically loads .npy files from the provided BASE_PATH,
runs TLBO in a background thread, captures stdout live, updates a progress bar, shows logs,
and plots iteration history after completion.

Usage:
    python3 TLBO_GUI_auto_dark.py
"""

import os
import sys
import threading
import queue
import re
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# Matplotlib for plotting results after run
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# === EDIT THIS PATH (already set to the path you provided) ===
BASE_PATH = "/Users/vidhirohira/Documents/E DRIVE/VJTI/SEM V/OT LAB/LAB PROJECT/Hostel-Room-Allocation/npy Files"
# =================================================================

# Import your TLBO class (assumes TLBO_Room_Allocation.py is in PYTHONPATH / same folder)
try:
    from TLBO_Room_Allocation import TLBO
except Exception as e:
    # If import fails, keep the exception but allow GUI to start and show the error later
    TLBO = None
    import_error = e
else:
    import_error = None

# ---------- Helper: redirect stdout to a queue ----------
class StdoutRedirector:
    def __init__(self, q, orig_stdout):
        self.queue = q
        self.orig = orig_stdout

    def write(self, msg):
        if msg is None:
            return
        # put into queue for the GUI thread to read
        self.queue.put(msg)
        # also keep original stdout (optional)
        try:
            self.orig.write(msg)
        except Exception:
            pass

    def flush(self):
        try:
            self.orig.flush()
        except Exception:
            pass

# ---------- Main GUI App ----------
class TLBOApp:
    def __init__(self, root):
        self.root = root
        root.title("TLBO Room Allocation — Dark Mode")
        root.geometry("900x640")
        root.minsize(820, 560)

        # Dark theme colors
        self.bg = "#1e1f22"
        self.panel = "#252628"
        self.fg = "#e6e6e6"
        self.accent = "#4fb0c6"
        self.warn = "#f39c12"

        root.configure(bg=self.bg)

        # Top frame (title + controls)
        top = tk.Frame(root, bg=self.panel, bd=0, relief=tk.FLAT)
        top.pack(fill=tk.X, pady=12, padx=12)

        title = tk.Label(top, text="TLBO Room Allocation", font=("Segoe UI", 18, "bold"),
                         bg=self.panel, fg=self.fg)
        title.grid(row=0, column=0, sticky="w", padx=(10, 6))

        # Controls frame
        ctrl = tk.Frame(top, bg=self.panel)
        ctrl.grid(row=0, column=1, sticky="e", padx=8)

        tk.Label(ctrl, text="Population:", bg=self.panel, fg=self.fg).grid(row=0, column=0, padx=6)
        self.pop_entry = tk.Entry(ctrl, width=6, justify="center")
        self.pop_entry.insert(0, "80")
        self.pop_entry.grid(row=0, column=1, padx=(0,10))

        tk.Label(ctrl, text="Max Iter:", bg=self.panel, fg=self.fg).grid(row=0, column=2, padx=6)
        self.iter_entry = tk.Entry(ctrl, width=6, justify="center")
        self.iter_entry.insert(0, "400")
        self.iter_entry.grid(row=0, column=3, padx=(0,10))

        self.run_btn = tk.Button(ctrl, text="Run TLBO (Auto-load .npy)", bg=self.accent, fg="#0b0b0b",
                                 activebackground="#38b7d0", command=self.start_run_thread, padx=8)
        self.run_btn.grid(row=0, column=4, padx=(6,10))

        self.stop_requested = False
        self.cancel_btn = tk.Button(ctrl, text="Stop", bg=self.warn, fg="#111", state="disabled",
                                    command=self.request_stop)
        self.cancel_btn.grid(row=0, column=5, padx=(0,6))

        # Middle frame (left: logs, right: plot + results)
        middle = tk.Frame(root, bg=self.bg)
        middle.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0,12))

        # Left: logs
        left = tk.Frame(middle, bg=self.panel)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,8), pady=4)

        lbl_logs = tk.Label(left, text="Live Log:", bg=self.panel, fg=self.fg, anchor="w")
        lbl_logs.pack(fill=tk.X, padx=8, pady=(8,0))

        self.log_text = tk.Text(left, bg="#121213", fg=self.fg, insertbackground=self.fg,
                                wrap="word", height=22)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4,8))
        self.log_text.configure(state=tk.DISABLED)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(left, maximum=100.0, variable=self.progress_var)
        self.progress.pack(fill=tk.X, padx=8, pady=(0,8))

        # Right: plot + results
        right = tk.Frame(middle, bg=self.panel, width=380)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, pady=4)

        # Matplotlib Figure (empty initially)
        fig = Figure(figsize=(4.2,3.0), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_facecolor("#222223")
        self.ax.tick_params(colors=self.fg)
        self.ax.xaxis.label.set_color(self.fg)
        self.ax.yaxis.label.set_color(self.fg)
        self.ax.title.set_color(self.fg)

        self.canvas = FigureCanvasTkAgg(fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Result box
        res_frame = tk.Frame(right, bg=self.panel)
        res_frame.pack(fill=tk.X, padx=8, pady=(0,12))

        tk.Label(res_frame, text="Final Best Score:", bg=self.panel, fg=self.fg).grid(row=0, column=0, sticky="w")
        self.best_score_label = tk.Label(res_frame, text="—", bg=self.panel, fg=self.accent)
        self.best_score_label.grid(row=0, column=1, sticky="e", padx=8)

        tk.Label(res_frame, text="Saved Vector:", bg=self.panel, fg=self.fg).grid(row=1, column=0, sticky="w", pady=(6,0))
        self.vector_text = tk.Text(res_frame, height=5, width=32, bg="#121213", fg=self.fg)
        self.vector_text.grid(row=2, column=0, columnspan=2, pady=(4,0))

        # Footer instructions
        foot = tk.Label(root, text=f"Auto-loading .npy from:\n{BASE_PATH}", bg=self.bg, fg="#9aa0a6", justify="left")
        foot.pack(fill=tk.X, padx=12, pady=(0,10))

        # Queue & stdout redirection
        self.q = queue.Queue()
        self.stdout_orig = sys.stdout
        self.redirector = StdoutRedirector(self.q, self.stdout_orig)

        # Thread handler
        self.worker_thread = None
        self.history = None
        self.best_vector = None
        self.best_score = None

        # Polling loop to update GUI from stdout queue
        root.after(100, self.poll_stdout_queue)

    def request_stop(self):
        self.stop_requested = True
        self.append_log("Stop requested. TLBO will attempt to stop between iterations (if supported).\n")

    def append_log(self, text):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def poll_stdout_queue(self):
        """Read items from stdout redirection queue and write to log_text; parse iterations to update progress."""
        while True:
            try:
                s = self.q.get_nowait()
            except queue.Empty:
                break
            else:
                # s may contain partial lines; just write it
                self.append_log(s)

                # try parse iteration lines for progress e.g. "Iteration 20: Best Score = -1330.0"
                m = re.search(r"Iteration\s+(\d+)\s*:\s*Best Score\s*=\s*([-+]?\d*\.?\d+)", s)
                if m:
                    try:
                        iter_num = int(m.group(1))
                        # get requested max_iter from entry (safe fallback)
                        try:
                            max_iter = int(self.iter_entry.get())
                        except Exception:
                            max_iter = 400
                        percent = min(100.0, (iter_num / float(max_iter)) * 100.0)
                        self.progress_var.set(percent)
                    except Exception:
                        pass

        # if worker thread finished and we have history, ensure progress = 100
        if self.worker_thread and not self.worker_thread.is_alive() and self.history is not None:
            self.progress_var.set(100.0)

        self.root.after(100, self.poll_stdout_queue)

    def start_run_thread(self):
        # Basic checks
        if import_error:
            messagebox.showerror("Import Error", f"Could not import TLBO_Room_Allocation:\n{import_error}")
            return

        # Check .npy files existence
        pref_path = os.path.join(BASE_PATH, "Room_Preference_Matrix.npy")
        if not os.path.isfile(pref_path):
            messagebox.showerror("File Not Found", f"Room_Preference_Matrix.npy not found at:\n{pref_path}")
            return

        # Disable run button, enable cancel
        self.run_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.stop_requested = False

        # Clear previous logs/plot
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self.ax.clear()
        self.ax.set_facecolor("#222223")
        self.canvas.draw()

        # Start worker thread
        self.worker_thread = threading.Thread(target=self.worker_run_tlbo, daemon=True)
        self.worker_thread.start()

    def worker_run_tlbo(self):
        """Run TLBO while redirecting stdout to GUI via our redirector (so print() from TLBO shows live)."""
        # redirect stdout
        sys.stdout = self.redirector

        try:
            # load matrix automatically
            pref_path = os.path.join(BASE_PATH, "Room_Preference_Matrix.npy")
            room_pref_matrix = np.load(pref_path)

            # get params
            try:
                pop_size = int(self.pop_entry.get())
            except Exception:
                pop_size = 80
            try:
                max_iter = int(self.iter_entry.get())
            except Exception:
                max_iter = 400

            # instantiate and run TLBO
            print(f"Loaded preference matrix shape: {room_pref_matrix.shape}\n")
            print(f"Starting TLBO with population={pop_size}, max_iter={max_iter}\n")

            tlbo = TLBO(population_size=pop_size, max_iter=max_iter)

            # Run (this call may print iter logs — we capture them)
            best_vec, best_score, history = tlbo.run(room_pref_matrix)

            # Save results
            out_vec_path = os.path.join(os.path.dirname(BASE_PATH), "TLBO_Final_Vector.npy")
            out_score_path = os.path.join(os.path.dirname(BASE_PATH), "TLBO_Final_Score.npy")
            np.save(out_vec_path, best_vec)
            np.save(out_score_path, np.array([best_score]))

            print(f"\nSaved: {out_vec_path}")
            print(f"Saved: {out_score_path}\n")

            # store into instance variables for later plotting / display
            self.history = history
            self.best_vector = best_vec
            self.best_score = best_score

            # Update result widgets in GUI thread using after()
            self.root.after(0, self.on_run_complete)

        except Exception as e:
            # Show exception in GUI
            err = f"\nError while running TLBO:\n{type(e).__name__}: {e}\n"
            print(err)
            self.root.after(0, lambda: messagebox.showerror("Run Error", err))
        finally:
            # restore stdout
            sys.stdout = self.stdout_orig

    def on_run_complete(self):
        # enable run button again
        self.run_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")

        # display final score & vector
        if self.best_score is not None:
            self.best_score_label.config(text=str(self.best_score))
        if self.best_vector is not None:
            self.vector_text.delete(1.0, tk.END)
            self.vector_text.insert(tk.END, np.array2string(self.best_vector, max_line_width=60))

        # plot history if available (history expected to be list-like of best scores per iteration)
        if self.history is not None:
            try:
                iters = list(range(len(self.history)))
                scores = np.array(self.history)
                # If history contains dicts or tuples, try to extract values
                if scores.dtype == object:
                    # try to coerce numeric values
                    try:
                        scores = np.array([float(x) for x in self.history])
                    except Exception:
                        scores = None

                if scores is not None:
                    self.ax.clear()
                    self.ax.plot(iters, scores)
                    self.ax.set_title("Iteration vs Best Score")
                    self.ax.set_xlabel("Iteration")
                    self.ax.set_ylabel("Best Score")
                    self.ax.grid(True, linestyle="--", alpha=0.4)
                    self.canvas.draw()
            except Exception:
                pass

        self.append_log("\nTLBO run completed.\n")

# === Run the app ===
def main():
    root = tk.Tk()
    app = TLBOApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
