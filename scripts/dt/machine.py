"""Process settings for time measurements on Windows hybrid CPUs.

Windows 11 may run a windowless background process at low quality of service (EcoQoS: efficiency cores, lower clock). On the
machine of docs/dt_extensions.md (Intel i7-13620H, 6 performance + 4 efficiency cores, Balanced power plan) this tripled the
measured round times of every running experiment at the same moment, in the middle of a run, with no stress injected.
prefer_performance() opts the process out of execution-speed throttling (SetProcessInformation, ProcessPowerThrottling) and
raises its priority class to ABOVE_NORMAL. Elsewhere it does nothing. It changes timing only, never a computed value.
"""
import sys


def prefer_performance():
    if sys.platform != 'win32':
        return {'platform': sys.platform, 'applied': False}
    import ctypes
    from ctypes import wintypes

    class PowerThrottlingState(ctypes.Structure):
        _fields_ = [('Version', wintypes.ULONG), ('ControlMask', wintypes.ULONG), ('StateMask', wintypes.ULONG)]

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    handle = kernel32.GetCurrentProcess()
    # version 1; control the execution-speed bit (0x1); state 0 = never throttle
    state = PowerThrottlingState(1, 0x1, 0x0)
    throttling_off = bool(kernel32.SetProcessInformation(handle, 4, ctypes.byref(state), ctypes.sizeof(state)))
    above_normal = bool(kernel32.SetPriorityClass(handle, 0x8000))
    return {'platform': 'win32', 'applied': throttling_off and above_normal, 'power_throttling_off': throttling_off,
            'priority_above_normal': above_normal}
