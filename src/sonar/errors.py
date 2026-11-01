"""Domain errors. Every error carries an actionable hint for the user."""

from __future__ import annotations

import platform


class SonarError(Exception):
    """Base error. ``hint`` tells the user how to fix the problem."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return self.message if not self.hint else f"{self.message}\n→ {self.hint}"


class ConfigError(SonarError):
    """Invalid or unsafe configuration."""


class DeviceNotFoundError(SonarError):
    """The requested audio device does not exist or lacks the needed channels."""


class UnsupportedSampleRateError(SonarError):
    """The device does not support the requested sample rate."""


class MicrophonePermissionError(SonarError):
    """The OS blocked microphone access (usually shows up as digital silence)."""


class ClippingError(SonarError):
    """The recording saturated the ADC, which distorts the matched filter."""


class LowSNRError(SonarError):
    """The emitted chirp could not be told apart from the background noise."""


class BandOutOfResponseError(SonarError):
    """The configured band is outside what the speaker/microphone can reproduce."""


class AudioIOError(SonarError):
    """Any other audio backend failure."""


class IncompatibleBackgroundError(SonarError):
    """A stored background profile does not match the current configuration."""


def microphone_permission_hint(system: str | None = None) -> str:
    """OS-specific instructions to grant microphone access to the terminal."""
    system = system or platform.system()
    if system == "Windows":
        return (
            "Windows: Configuración → Privacidad y seguridad → Micrófono. Activa "
            "'Acceso al micrófono' y 'Permitir que las aplicaciones de escritorio accedan "
            "al micrófono'. Revisa también que el micrófono no esté silenciado en "
            "Configuración → Sistema → Sonido → Entrada."
        )
    if system == "Darwin":
        return (
            "macOS: Ajustes del Sistema → Privacidad y seguridad → Micrófono, y habilita tu "
            "terminal (Terminal, iTerm, VS Code...). Reinicia la terminal después. Si no "
            "aparece en la lista, ejecuta `tccutil reset Microphone` y vuelve a intentar."
        )
    if system == "Linux":
        return (
            "Linux: verifica que la fuente no esté silenciada (`pactl list sources`, "
            "`pavucontrol`), que tu usuario pertenezca al grupo `audio` y, si usas "
            "Flatpak/Snap, que la app tenga permiso de micrófono (`snap connect ...:audio-record`)."
        )
    return "Revisa los permisos de micrófono del sistema operativo para esta terminal."
