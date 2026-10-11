"""Code that reads the files you write and resolves them into one settings object.

You never edit this package to change a deployment. You edit:

    deploy.toml                   beside your policy: rates, replan period, prompt,
                                  checkpoint, home pose    -> deploy_file.py
    a robot file                  robot/profiles/*.toml, or your own `extends` one:
                                  cameras, TCP, servo model -> robot_file.py
    machine.toml                  this checkout, gitignored: robot IP, NIC, board
                                  login                      -> machine_file.py
    the dashboard's Deploy settings   between rollouts       -> rollout_settings.py

Precedence: infra defaults < what the policy class declares (`trained_*`) <
deploy.toml < command line (`docs/deploying_a_policy/deploy_toml.md`, "How a
value is decided"). `resolved.py` holds the result every process reads
(`DeployConfig`, `Hardware`), its derived quantities and its consistency checks;
`json_transport.py` carries it to a child running under another interpreter.

Deliberately empty of imports: P5 imports parts of this package under the
policy's own interpreter, which holds only numpy.
"""
