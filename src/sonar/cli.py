"""Command line interface. Thin layer: parses options, calls the engine, renders output."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Annotated, NoReturn

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from sonar.audio import AudioDevice, SoundDeviceAudio
from sonar.config import AUDIBLE_BAND, NEAR_ULTRASONIC_BAND, SonarConfig
from sonar.diagnostics import recommend_band
from sonar.engine import Measurement, SonarEngine
from sonar.errors import (
    BandOutOfResponseError,
    ClippingError,
    LowSNRError,
    SonarError,
)
from sonar.processing import BackgroundProfile

DEFAULT_BACKGROUND = Path(".sonar") / "background.npz"
SAFETY_NOTE = (
    "Aviso: el casi-ultrasonido a alto volumen molesta a mascotas y puede dañar tweeters. "
    "Mantén el volumen del sistema moderado."
)

app = typer.Typer(
    help="Sonar acústico con el parlante y micrófono del computador.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


class Band(str, Enum):
    ultrasonic = "ultrasonic"
    audible = "audible"


def make_device(cfg: SonarConfig) -> AudioDevice:
    """Factory for the real device; tests replace it with a fake."""
    return SoundDeviceAudio(
        cfg.input_device, cfg.output_device, cfg.max_amplitude, cfg.host_api, cfg.exclusive
    )


def _fail(exc: SonarError) -> NoReturn:
    err_console.print(f"[bold red]Error:[/] {exc.message}")
    if exc.hint:
        err_console.print(f"[yellow]Cómo resolverlo:[/] {exc.hint}")
    raise typer.Exit(code=2)


@contextmanager
def _errors() -> Iterator[None]:
    try:
        yield
    except SonarError as exc:
        _fail(exc)


def _parse_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    return int(value) if value.strip().lstrip("-").isdigit() else value


def _cfg(ctx: typer.Context) -> SonarConfig:
    cfg = ctx.obj
    assert isinstance(cfg, SonarConfig)
    return cfg


@app.callback()
def main(
    ctx: typer.Context,
    band: Annotated[Band | None, typer.Option(help="Preset de banda: ultrasonic=18–22 kHz, audible=8–16 kHz.")] = None,
    f_start: Annotated[float | None, typer.Option(help="Frecuencia inicial del chirp (Hz).")] = None,
    f_end: Annotated[float | None, typer.Option(help="Frecuencia final del chirp (Hz).")] = None,
    fs: Annotated[int | None, typer.Option(help="Tasa de muestreo (Hz).")] = None,
    chirp_ms: Annotated[float | None, typer.Option(help="Duración del chirp (ms).")] = None,
    window: Annotated[str | None, typer.Option(help="Ventana del chirp: tukey | hann.")] = None,
    pings: Annotated[int | None, typer.Option(help="Pings a promediar coherentemente.")] = None,
    amplitude: Annotated[float | None, typer.Option(help="Amplitud digital (0–1).")] = None,
    max_amplitude: Annotated[float | None, typer.Option(help="Tope de seguridad de amplitud.")] = None,
    temperature: Annotated[float | None, typer.Option(help="Temperatura del aire (°C).")] = None,
    sound_speed: Annotated[float | None, typer.Option(help="Fuerza la velocidad del sonido (m/s).")] = None,
    direct_path: Annotated[float | None, typer.Option(help="Separación parlante–micrófono (m) para corregir el sesgo.")] = None,
    max_range: Annotated[float | None, typer.Option(help="Distancia máxima a buscar (m).")] = None,
    min_range: Annotated[float | None, typer.Option(help="Zona ciega cercana (m).")] = None,
    min_snr: Annotated[float | None, typer.Option(help="SNR mínimo de un eco (dB).")] = None,
    input_dev: Annotated[str | None, typer.Option("--input", help="Micrófono: índice o parte del nombre.")] = None,
    output_dev: Annotated[str | None, typer.Option("--output", help="Parlante: índice o parte del nombre.")] = None,
    host_api: Annotated[str | None, typer.Option(help="Busca --input/--output sólo en esta host API (p. ej. WASAPI).")] = None,
    exclusive: Annotated[bool, typer.Option(help="WASAPI exclusivo: evita el procesamiento de voz de Windows.")] = False,
    warmup: Annotated[int | None, typer.Option(help="Pings de calentamiento que no se promedian.")] = None,
    interval_ms: Annotated[float | None, typer.Option(help="Intervalo entre pings (ms).")] = None,
) -> None:
    preset = {Band.ultrasonic: NEAR_ULTRASONIC_BAND, Band.audible: AUDIBLE_BAND}.get(band) if band else None
    with _errors():
        ctx.obj = SonarConfig().with_overrides(
            f_start=f_start if f_start is not None else (preset[0] if preset else None),
            f_end=f_end if f_end is not None else (preset[1] if preset else None),
            fs=fs,
            chirp_duration=chirp_ms / 1e3 if chirp_ms is not None else None,
            window=window,
            n_pings=pings,
            amplitude=amplitude,
            max_amplitude=max_amplitude,
            temperature_c=temperature,
            speed_of_sound_override=sound_speed,
            direct_path_m=direct_path,
            max_range=max_range,
            min_range=min_range,
            min_snr_db=min_snr,
            input_device=_parse_device(input_dev),
            output_device=_parse_device(output_dev),
            host_api=host_api,
            exclusive=exclusive or None,
            warmup_pings=warmup,
            ping_interval=interval_ms / 1e3 if interval_ms is not None else None,
        )


# ------------------------------------------------------------------ helpers


def _load_background(cfg: SonarConfig, path: Path, explicit: bool, disabled: bool) -> BackgroundProfile | None:
    if disabled:
        return None
    if not path.exists():
        if explicit:
            raise SonarError(f"No existe el perfil de fondo {path}.", "Ejecuta `sonar calibrate` primero.")
        console.print("[dim]Sin perfil de fondo (ejecuta `sonar calibrate` para quitar el acoplamiento estático).[/]")
        return None
    return BackgroundProfile.load(path, cfg)


def _echo_table(m: Measurement) -> Table:
    p = m.profile
    table = Table(title=f"Ecos (c = {p.cfg.speed_of_sound:.1f} m/s, {p.n_averaged} pings, "
                        f"resolución {p.cfg.range_resolution * 100:.1f} cm)")
    table.add_column("#", justify="right")
    table.add_column("distancia (m)", justify="right")
    table.add_column("retardo (ms)", justify="right")
    table.add_column("SNR (dB)", justify="right")
    for i, e in enumerate(m.echoes, 1):
        table.add_row(str(i), f"{e.distance_m:.3f}", f"{e.delay_s * 1e3:.3f}", f"{e.snr_db:.1f}")
    return table


BackgroundOpt = Annotated[Path | None, typer.Option("--background", help=f"Perfil de fondo (por defecto {DEFAULT_BACKGROUND}).")]
NoBackgroundOpt = Annotated[bool, typer.Option("--no-background", help="No restar el perfil de fondo.")]


# ----------------------------------------------------------------- commands


@app.command()
def devices(ctx: typer.Context) -> None:
    """Lista dispositivos de entrada/salida y las tasas de muestreo soportadas."""
    cfg = _cfg(ctx)
    with _errors():
        infos = make_device(cfg).list_devices()
    table = Table(title="Dispositivos de audio")
    for col in ("idx", "nombre", "host API", "in", "out", "fs por defecto", "fs soportadas"):
        table.add_column(col)
    for d in infos:
        mark = (" [green](in)[/]" if d.is_default_input else "") + (" [green](out)[/]" if d.is_default_output else "")
        table.add_row(
            str(d.index), d.name + mark, d.hostapi, str(d.max_input_channels), str(d.max_output_channels),
            f"{d.default_samplerate:.0f}", ", ".join(map(str, d.supported_samplerates)) or "—",
        )
    console.print(table)


@app.command()
def diagnose(
    ctx: typer.Context,
    duration: Annotated[float, typer.Option(help="Duración del barrido (s).")] = 2.0,
    min_snr: Annotated[float, typer.Option(help="SNR mínimo para considerar usable una frecuencia (dB).")] = 15.0,
    plot: Annotated[Path | None, typer.Option(help="Guarda la gráfica de respuesta en este archivo.")] = None,
) -> None:
    """Barrido de frecuencias: mide la respuesta real y recomienda la mejor banda."""
    cfg = _cfg(ctx)
    console.print(f"[dim]{SAFETY_NOTE}[/]")
    console.print("Midiendo ruido ambiente y luego reproduciendo un barrido… guarda silencio.")
    with _errors():
        response = SonarEngine(make_device(cfg), cfg).diagnose(duration=duration)

    table = Table(title="Respuesta por banda de 1 kHz")
    table.add_column("banda (kHz)")
    table.add_column("SNR medio (dB)", justify="right")
    table.add_column("")
    top = int(response.freqs[-1] // 1000)
    for k in range(1, top):
        snr = response.mean_snr(k * 1e3, (k + 1) * 1e3)
        bar = "█" * max(0, min(40, int(snr / 2)))
        color = "green" if snr >= min_snr else "red"
        table.add_row(f"{k}–{k + 1}", f"[{color}]{snr:.1f}[/]", f"[{color}]{bar}[/]")
    console.print(table)

    coverage = response.coverage(cfg.f_start, cfg.f_end, min_snr)
    if coverage < 0.8:
        console.print(
            f"[yellow]La banda actual {cfg.f_start:.0f}–{cfg.f_end:.0f} Hz sólo es usable en un "
            f"{coverage:.0%}: está fuera de la respuesta del hardware.[/]"
        )
    recommendation = None
    try:
        recommendation = recommend_band(response, min_snr_db=min_snr)
    except BandOutOfResponseError as exc:
        err_console.print(f"[bold red]{exc.message}[/]\n[yellow]{exc.hint}[/]")
    if recommendation:
        r = recommendation
        console.print(
            f"[bold green]Banda recomendada:[/] {r.f_start:.0f}–{r.f_end:.0f} Hz ({r.kind}, "
            f"SNR medio {r.mean_snr_db:.0f} dB, resolución {r.range_resolution(cfg.speed_of_sound) * 100:.1f} cm)"
        )
        console.print(f"  Úsala con: [cyan]sonar --f-start {r.f_start:.0f} --f-end {r.f_end:.0f} ping[/]")
        if r.kind != "near-ultrasonic":
            console.print("  [yellow]El hardware no responde bien sobre 17 kHz; la banda será audible.[/]"
                          "  Fallback directo: [cyan]sonar --band audible ping[/]")
    if plot:
        from sonar.viz import frequency_response_figure

        fig = frequency_response_figure(response, min_snr, recommendation, (cfg.f_start, cfg.f_end))
        fig.savefig(plot, dpi=120)
        console.print(f"Gráfica guardada en {plot}")
    if recommendation is None:
        raise typer.Exit(code=2)


@app.command()
def calibrate(
    ctx: typer.Context,
    output: Annotated[Path, typer.Option("--out", help="Dónde guardar el perfil de fondo.")] = DEFAULT_BACKGROUND,
) -> None:
    """Graba el perfil de fondo (acoplamiento directo + reflexiones estáticas) y lo guarda."""
    cfg = _cfg(ctx)
    console.print(
        "Calibrando: deja el equipo en su posición de medida, apuntando a espacio libre "
        "(sin el objetivo delante) y no te muevas."
    )
    with _errors():
        background = SonarEngine(make_device(cfg), cfg).calibrate()
        background.save(output)
    console.print(f"[green]Perfil de fondo guardado en {output}[/]")


@app.command()
def ping(
    ctx: typer.Context,
    background: BackgroundOpt = None,
    no_background: NoBackgroundOpt = False,
    plot: Annotated[Path | None, typer.Option(help="Guarda el A-scan en este archivo (png/svg/pdf).")] = None,
    show: Annotated[bool, typer.Option(help="Muestra el A-scan en una ventana.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Salida JSON para scripts.")] = False,
    save_raw: Annotated[Path | None, typer.Option(help="Guarda la grabación cruda (.npy) para re-analizarla.")] = None,
) -> None:
    """Una medición: imprime los ecos más fuertes con distancia y SNR."""
    cfg = _cfg(ctx)
    with _errors():
        bg = _load_background(cfg, background or DEFAULT_BACKGROUND, background is not None, no_background)
        engine = SonarEngine(make_device(cfg), cfg, bg)
        recording = engine.acquire()
        if save_raw:
            np.save(save_raw, recording)
        m = engine.analyse(recording)

    if as_json:
        payload = {
            "speed_of_sound": cfg.speed_of_sound,
            "direct_snr_db": m.profile.direct_snr_db,
            "pings_averaged": m.profile.n_averaged,
            "echoes": [{"distance_m": e.distance_m, "snr_db": e.snr_db, "delay_s": e.delay_s} for e in m.echoes],
        }
        console.print_json(json.dumps(payload))
    elif m.echoes:
        console.print(_echo_table(m))
    else:
        console.print(
            f"[yellow]Sin ecos sobre {cfg.min_snr_db:.0f} dB entre {cfg.min_range} y {cfg.max_range} m.[/] "
            "Acércate a una superficie plana y dura, sube --pings o usa `sonar calibrate`."
        )
    if plot or show:
        import matplotlib.pyplot as plt

        from sonar.viz import profile_figure

        fig = profile_figure(m.profile, m.echoes)
        if plot:
            fig.savefig(plot, dpi=120)
            console.print(f"A-scan guardado en {plot}")
        if show:
            plt.show()


@app.command()
def live(
    ctx: typer.Context,
    background: BackgroundOpt = None,
    no_background: NoBackgroundOpt = False,
    history: Annotated[int, typer.Option(help="Filas del waterfall.")] = 120,
) -> None:
    """Perfil de eco en tiempo real (A-scan + waterfall). Cierra la ventana o Ctrl+C para salir."""
    import matplotlib.pyplot as plt

    from sonar.viz import LiveView

    cfg = _cfg(ctx)
    with _errors():
        bg = _load_background(cfg, background or DEFAULT_BACKGROUND, background is not None, no_background)
        engine = SonarEngine(make_device(cfg), cfg, bg)
        view = LiveView(cfg, history=history)
        try:
            while view.is_open:
                try:
                    m = engine.measure()
                    view.update(m.profile, m.echoes)
                except (LowSNRError, ClippingError) as exc:  # Transient: keep the loop alive.
                    view.show_status(exc.message)
                plt.pause(0.01)
        except KeyboardInterrupt:
            pass
    console.print("Fin del modo live.")


@app.command()
def sweep(
    ctx: typer.Context,
    background: BackgroundOpt = None,
    no_background: NoBackgroundOpt = False,
    out: Annotated[Path, typer.Option(help="Archivo del gráfico polar.")] = Path("sweep.png"),
    show: Annotated[bool, typer.Option(help="Muestra el gráfico al final.")] = False,
) -> None:
    """Barrido manual: gira el equipo, ingresa el ángulo y mide; al final dibuja un radar polar."""
    from sonar.viz import SweepScan, polar_sweep_figure

    cfg = _cfg(ctx)
    scans: list[SweepScan] = []
    with _errors():
        bg = _load_background(cfg, background or DEFAULT_BACKGROUND, background is not None, no_background)
        engine = SonarEngine(make_device(cfg), cfg, bg)
    console.print("Gira el equipo al ángulo deseado, escribe el ángulo y presiona Enter. Enter vacío termina.")
    while True:
        raw = typer.prompt("Ángulo (°)", default="", show_default=False).strip()
        if not raw:
            break
        try:
            angle = float(raw.replace(",", "."))
        except ValueError:
            err_console.print(f"[red]'{raw}' no es un número.[/]")
            continue
        try:
            m = engine.measure()
        except (LowSNRError, ClippingError) as exc:
            err_console.print(f"[yellow]{exc.message}[/] {exc.hint or ''} Repite este ángulo.")
            continue
        except SonarError as exc:
            _fail(exc)
        scans.append(SweepScan(angle, m.profile, tuple(m.echoes)))
        best = f"{m.echoes[0].distance_m:.2f} m ({m.echoes[0].snr_db:.0f} dB)" if m.echoes else "sin ecos"
        console.print(f"  {angle:.1f}° → {best}")
    if not scans:
        console.print("No se midió ningún ángulo.")
        return
    fig = polar_sweep_figure(scans)
    fig.savefig(out, dpi=120)
    console.print(f"[green]Gráfico polar guardado en {out}[/] ({len(scans)} ángulos)")
    if show:
        import matplotlib.pyplot as plt

        plt.show()
