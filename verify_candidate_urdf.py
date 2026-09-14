"""One-off check: does candidate_orcahand_right_teleop.urdf satisfy the
adaptive_analytical retargeter's joint/frame naming requirements?

Uses only what `uv sync --extra test --extra mediapipe --extra adaptive`
already installed - doesn't install anything. Delete this file (and the
candidate URDF) once the real fix lands.
"""

from orca_teleop.retargeting.adaptive_analytical import AdaptiveAnalyticalRetargeter

retargeter = AdaptiveAnalyticalRetargeter.from_paths(
    model_path="../orca_core/orca_core/models/v2/orcahand-right/config.yaml",
    urdf_path="candidate_orcahand_right_teleop.urdf",
)
print("OK - retargeter initialized cleanly:", retargeter)



uv run python -c "
from orca_core import OrcaHand
hand = OrcaHand('orca_core/models/v2/orcahand-right/config.yaml')
print(hand.connect(interactive=False))
print(hand.get_motor_pos())
print(hand._motor_client.last_read_ok)
hand.disconnect()
""

uv run python -c "
from orca_teleop.pipeline import run_local
run_local(
        model_path='../orca_core/orca_core/models/v2/orcahand-right/config.yaml',
            urdf_path='candidate_orcahand_right_teleop.urdf',
                handedness='right',
                    show_video=True,)
"