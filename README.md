- [Remote Apple Events](#remote-apple-events)
- [Root User](#root-user)
- [Safe Mode Boot](#safe-mode-boot)
- [Screenshots](#screenshots)
  * [Take Delayed Screenshot](#take-delayed-screenshot)
  * [Save Screenshots to Given Location](#save-screenshots-to-given-location)
  * [Save Screenshots in Given Format](#save-screenshots-in-given-format)
  * [Disable Shadow in Screenshots](#disable-shadow-in-screenshots)
  * [Set Default Screenshot Name](#set-default-screenshot-name)
- [Software Installation](#software-installation)
  * [Install PKG](#install-pkg)
- [Software Update](#software-update)
  * [Install All Available Software Updates](#install-all-available-software-updates)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

### Remote Apple Events
```bash
# Status
sudo systemsetup -getremoteappleevents

# Enable
sudo systemsetup -setremoteappleevents on

# Disable (Default)
sudo systemsetup -setremoteappleevents off
```

### Root User

```bash
# Enable
dsenableroot

# Disable
dsenableroot -d
```

### Safe Mode Boot

```bash
# Status
nvram boot-args

# Enable
sudo nvram boot-args="-x"

# Disable
sudo nvram boot-args=""
```

### Screenshots

#### Take Delayed Screenshot
Takes a screenshot as JPEG after 3 seconds and displays in Preview.
```bash
screencapture -T 3 -t jpg -P delayedpic.jpg
```

#### Save Screenshots to Given Location
Sets location to `~/Desktop`.
```bash
defaults write com.apple.screencapture location ~/Desktop && \
killall SystemUIServer
```

#### Save Screenshots in Given Format
Sets format to `png`. Other options are `bmp`, `gif`, `jpg`, `jpeg`, `pdf`, `tiff`.
```bash
defaults write com.apple.screencapture type -string "png"
```

#### Disable Shadow in Screenshots
```bash
defaults write com.apple.screencapture disable-shadow -bool true && \
killall SystemUIServer
```

#### Set Default Screenshot Name
Date and time remain unchanged.
```bash
defaults write com.apple.screencapture name "Example name" && \
killall SystemUIServer
```

### Software Installation

#### Install PKG
```bash
installer -pkg /path/to/installer.pkg -target /
```

### Software Update

#### Install All Available Software Updates
```bash
sudo softwareupdate -ia
```

