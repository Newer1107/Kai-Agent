# Desktop Agent Task Examples

This project now supports desktop-first routing via os tools, with visual fallback.

## 1) Open Notepad and type text

Example command:

`open notepad`

Then type in chat:

`type hello from kai` (routes via existing agent fallback if direct tool is unavailable)

## 2) Create a folder in Documents

Example command:

`create folder C:\\Users\\Rishit Singh\\Documents\\KaiNotes`

## 3) Open Windows Settings

Example command:

`open settings`

Or specific page:

`open settings display`

## 4) Switch from Chrome to VS Code

Example commands:

`switch to chrome`

`switch to code`

## 5) Interact with File Explorer

Example commands:

`open explorer`

`list directory C:\\Users\\Rishit Singh\\Desktop`

`move file C:\\Users\\Rishit Singh\\Desktop\\a.txt to C:\\Users\\Rishit Singh\\Documents\\a.txt`

## Routing Behavior

1. Direct OS tools (open, switch, file ops, volume, window controls)
2. Desktop perception context snapshot
3. Existing universal router fallback
4. Existing visual agent fallback

## Safety Behavior

- Dangerous actions are blocked by default (shutdown/registry/format/uninstall patterns)
- Sensitive classes are marked approval-required in policy
