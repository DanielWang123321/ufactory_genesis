# Security policy

## Supported version

Only the current `0.2.7` development line receives security fixes. This is Alpha robotics software and is not safety-certified.

## Reporting

Do not disclose a suspected vulnerability in a public issue when it could enable unintended robot motion, code execution, calibration/manifest bypass, or credential exposure. Use GitHub's private security advisory workflow for this repository and include a minimal reproduction, affected revision and impact. Do not include real robot IP addresses, serial numbers or proprietary calibration data.

## Trust boundaries

- Treat YAML, URDF, checkpoints, SDK feedback and inventory commands as untrusted.
- Never use the legacy pickle migration tool unless `--trusted-input` truthfully applies.
- Never weaken `ApprovedProgram` binding, collision checks, complete-serial calibration binding, or explicit real-motion confirmation.
- Repository assets and safety exemptions are integrity-checked; a mismatch fails closed.
- The public package contains no online real-policy execution, random-action deployment, policy session, or SDK policy-action adapter. `ufactory.deploy` was removed in v0.2.7.

Operators remain responsible for physical guarding, payload/tool validation, emergency-stop access, speed limits and site-specific risk assessment.
