"""ST3215 (Feetech STS series) control table.

Addresses follow the STS memory table, as also used by LeRobot's Feetech driver and the
NormaCore ST3215 driver. Multi-byte values are little-endian. Some signed values use
sign-magnitude encoding: the bit given by `sign_bit` is the sign, the bits below it the
magnitude.

EEPROM registers (address < 40) survive power-off, but only if the EEPROM is unlocked
(`LOCK = 0`) while writing. RAM registers reset on power-up.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Register:
    name: str
    addr: int
    size: int
    sign_bit: int | None = None
    eeprom: bool = False

    def decode(self, data: bytes) -> int:
        raw = int.from_bytes(data[: self.size], "little")
        if self.sign_bit is not None and raw & (1 << self.sign_bit):
            return -(raw & ((1 << self.sign_bit) - 1))
        return raw

    def encode(self, value: int) -> bytes:
        if self.sign_bit is not None:
            limit = (1 << self.sign_bit) - 1
            if abs(value) > limit:
                raise ValueError(f"{self.name}: |{value}| > {limit}")
            raw = abs(value) | ((1 << self.sign_bit) if value < 0 else 0)
        else:
            if not 0 <= value < (1 << (8 * self.size)):
                raise ValueError(f"{self.name}: {value} out of range for {self.size} byte(s)")
            raw = value
        return raw.to_bytes(self.size, "little")


def _r(name, addr, size, sign_bit=None):
    return Register(name, addr, size, sign_bit, eeprom=addr < 40)


# fmt: off
FIRMWARE_MAJOR      = _r("firmware_major", 0, 1)
FIRMWARE_MINOR      = _r("firmware_minor", 1, 1)
MODEL               = _r("model", 3, 2)
ID                  = _r("id", 5, 1)
BAUD_RATE           = _r("baud_rate", 6, 1)
RETURN_DELAY        = _r("return_delay", 7, 1)          # x2 us
RESPONSE_LEVEL      = _r("response_level", 8, 1)
MIN_POSITION_LIMIT  = _r("min_position_limit", 9, 2)
MAX_POSITION_LIMIT  = _r("max_position_limit", 11, 2)
MAX_TEMPERATURE     = _r("max_temperature", 13, 1)      # deg C
MAX_VOLTAGE         = _r("max_voltage", 14, 1)          # 0.1 V
MIN_VOLTAGE         = _r("min_voltage", 15, 1)          # 0.1 V
MAX_TORQUE          = _r("max_torque", 16, 2)           # 0.1 % of stall torque
PHASE               = _r("phase", 18, 1)
UNLOADING_CONDITION = _r("unloading_condition", 19, 1)
LED_ALARM_CONDITION = _r("led_alarm_condition", 20, 1)
P_COEF              = _r("p_coef", 21, 1)
D_COEF              = _r("d_coef", 22, 1)
I_COEF              = _r("i_coef", 23, 1)
MIN_STARTUP_FORCE   = _r("min_startup_force", 24, 2)
CW_DEAD_ZONE        = _r("cw_dead_zone", 26, 1)
CCW_DEAD_ZONE       = _r("ccw_dead_zone", 27, 1)
PROTECTION_CURRENT  = _r("protection_current", 28, 2)
ANGULAR_RESOLUTION  = _r("angular_resolution", 30, 1)
OFFSET              = _r("offset", 31, 2, sign_bit=11)  # reported = raw - offset
MODE                = _r("mode", 33, 1)                 # 0 = position servo
PROTECTIVE_TORQUE   = _r("protective_torque", 34, 1)
PROTECTION_TIME     = _r("protection_time", 35, 1)
OVERLOAD_TORQUE     = _r("overload_torque", 36, 1)
SPEED_P_COEF        = _r("speed_p_coef", 37, 1)
OVERCURRENT_TIME    = _r("overcurrent_time", 38, 1)
SPEED_I_COEF        = _r("speed_i_coef", 39, 1)

TORQUE_ENABLE       = _r("torque_enable", 40, 1)
ACCELERATION        = _r("acceleration", 41, 1)         # x100 steps/s^2
GOAL_POSITION       = _r("goal_position", 42, 2, sign_bit=15)
GOAL_TIME           = _r("goal_time", 44, 2)
GOAL_SPEED          = _r("goal_speed", 46, 2)           # steps/s
TORQUE_LIMIT        = _r("torque_limit", 48, 2)         # 0.1 %
LOCK                = _r("lock", 55, 1)                 # 0 = EEPROM writable
PRESENT_POSITION    = _r("present_position", 56, 2, sign_bit=15)
PRESENT_SPEED       = _r("present_speed", 58, 2, sign_bit=15)
PRESENT_LOAD        = _r("present_load", 60, 2, sign_bit=10)  # 0.1 %
PRESENT_VOLTAGE     = _r("present_voltage", 62, 1)      # 0.1 V
PRESENT_TEMPERATURE = _r("present_temperature", 63, 1)  # deg C
STATUS              = _r("status", 65, 1)               # error bits, see STATUS_BITS
MOVING              = _r("moving", 66, 1)
PRESENT_CURRENT     = _r("present_current", 69, 2, sign_bit=15)  # ~6.5 mA per unit
# fmt: on

ALL = [v for v in list(globals().values()) if isinstance(v, Register)]
BY_NAME = {r.name: r for r in ALL}

# One block read from 0 to the end of PRESENT_CURRENT covers the whole table.
TABLE_END = PRESENT_CURRENT.addr + PRESENT_CURRENT.size

STATUS_BITS = {
    0: "voltage",
    1: "angle sensor",
    2: "overheat",
    3: "overcurrent",
    5: "overload",
}

STEPS_PER_TURN = 4096
STS3215_MODEL = 777


def decode_block(start: int, data: bytes) -> dict[str, int]:
    """Decode every register that lies fully inside data read from address `start`."""
    out = {}
    for reg in ALL:
        lo = reg.addr - start
        if lo >= 0 and lo + reg.size <= len(data):
            out[reg.name] = reg.decode(data[lo : lo + reg.size])
    return out


def status_flags(status: int) -> list[str]:
    return [name for bit, name in STATUS_BITS.items() if status & (1 << bit)]
