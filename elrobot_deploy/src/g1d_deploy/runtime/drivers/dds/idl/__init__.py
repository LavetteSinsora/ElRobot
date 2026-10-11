"""The DDS message types g1d_deploy speaks, vendored from unitree_sdk2py.

Exact equivalents of the IDL-generated classes in `unitree_sdk2_python`
(commit e4cd91f051aaa77a70600e3d2bf7f50889db1980, version 1.0.1): the same
DDS type names, the same field order, types and array sizes, the same
`@final` / `@autoid("sequential")` annotations. The wire bytes and the XTypes
type information are therefore identical, which `tests/test_wire_idl.py`
checks against the SDK itself wherever it is installed.

Only what g1d_deploy uses is here:

    unitree_hg.msg.dds_   LowState_ (+ IMUState_, MotorState_), LowCmd_ (+ MotorCmd_)
    unitree_go.msg.dds_   MotorCmds_ / MotorCmd_ (Dex-1, BrainCo commands),
                          MotorStates_ / MotorState_ (their state)
    geometry_msgs.msg.dds_  Point32_ (the lift column)

The one change from the generated files: a nested type is referenced by the
class itself rather than by the string `'unitree_sdk2py.idl....MotorCmd_'`,
which cyclonedds would resolve by importing unitree_sdk2py. The nested
types' own DDS names are unchanged, so nothing on the wire differs.

Importing these modules imports `cyclonedds`; they are loaded only by the
wire classes' `connect()`.

---------------------------------------------------------------------------
The generated type definitions are Unitree's, under this license:

BSD 3-Clause License

Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
