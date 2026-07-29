---
status: testing
phase: 08-system-tray-and-controls
source: main.py, tray.py, history.py, pipeline.py, settings.py, history_dialog.py
started: 2026-07-29T18:46:00Z
updated: 2026-07-29T18:46:00Z
---

## Current Test

number: 1
name: System tray icon appears
expected: |
  After launching the app, a blue tray icon (rounded square with "CC") appears in the system tray notification area.
awaiting: user response

## Tests

### 1. System Tray Icon Appears
expected: Blue "CC" icon appears in system tray notification area
result: [pending]

### 2. Right-Click Menu Shows Options
expected: Right-clicking the tray icon shows: Pause, Settings, History, Quit
result: [pending]

### 3. Pause/Resume Toggle
expected: Clicking "Pause" stops capture, shows ⏸ Paused overlay. Clicking "Resume" restarts capture, clears overlay
result: [pending]

### 4. Double-Click Tray Toggles
expected: Double-clicking the tray icon toggles between running and paused state
result: [pending]

### 5. Settings Opens with Language Selectors
expected: Right-click → Settings opens a dialog with dropdowns for Source Language and Target Language (10 languages each)
result: [pending]

### 6. Language Warning on Same Selection
expected: Selecting same source and target language shows an orange warning
result: [pending]

### 7. Settings Persists Language Choice
expected: Changing language in Settings, clicking OK, and reopening Settings shows the new language selection
result: [pending]

### 8. History Dialog Opens
expected: Right-click → History opens a dialog showing past sessions with timestamps, language pair, and subtitle count
result: [pending]

### 9. History Shows Subtitle Content
expected: Clicking a session in the left panel displays its subtitle text in the right panel
result: [pending]

### 10. Copy Subtitles from History
expected: Clicking "Copy Subtitles" copies the text to clipboard, button text changes to "Copied!" briefly
result: [pending]

### 11. Clear History Works
expected: Clicking "Clear History" shows confirmation dialog. Confirming deletes all history and refreshes the list
result: [pending]

### 12. Quit from Tray Menu
expected: Right-click → Quit shows "Are you sure?" confirmation. Confirming exits the app completely
result: [pending]

## Summary

total: 12
passed: 0
issues: 0
pending: 12
skipped: 0
blocked: 0

## Gaps

[none yet]
