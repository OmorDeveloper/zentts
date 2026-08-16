# Admin guide

How to control every ZenTTS install from one place. This is for the project
owner, not for users.

## The control file

Every install checks [`control.json`](control.json) in this repository, served
raw from:

```
https://raw.githubusercontent.com/OmorDeveloper/zentts/master/control.json
```

```json
{
  "enabled": true,
  "message": "",
  "min_version": "1.0.0",
  "blocked_versions": [],
  "require_package": true
}
```

| Field | Effect |
| --- | --- |
| `enabled` | `false` stops every install that can reach the file |
| `message` | shown to the user when it stops |
| `min_version` | anything older refuses to run and is told to upgrade |
| `blocked_versions` | exact versions that refuse to run |
| `require_package` | also require that `zentts` still exists on PyPI |

## Kill everything

Edit the file and push:

```bash
gh api repos/OmorDeveloper/zentts/contents/control.json \
  -X PUT -f message="Disable ZenTTS" \
  -f content="$(printf '{"enabled": false, "message": "ZenTTS has been withdrawn."}' | base64 -w0)" \
  -f sha="$(gh api repos/OmorDeveloper/zentts/contents/control.json -q .sha)"
```

Or simply edit `control.json`, commit and push. Installs pick it up within
12 hours, or immediately on their next first run of the day.

To bring it back, set `enabled` to `true` again — but note that any install
that already recorded the refusal keeps refusing until it can reach the file
again and read `enabled: true`.

## Kill by unpublishing

With `require_package: true`, deleting or fully yanking `zentts` on PyPI also
stops every install, because each one confirms the project still exists.

## Force an upgrade instead of a kill

```json
{"enabled": true, "min_version": "1.3.0"}
```

Anyone on an older version is told to run `pip install --upgrade zentts`.

## How it behaves on the user's machine

| Situation | Result |
| --- | --- |
| Online, `enabled: true` | runs, result cached for 12 hours |
| Online, `enabled: false` | stops, and the refusal is written to disk |
| Offline, checked in before | keeps running for 7 days, then stops |
| Offline, never checked in | refuses; it must reach the file once to activate |
| Already refused, now offline | stays refused |

The cached state lives at `<ZENTTS_HOME>/license.json`.

## Checking an install

```bash
zentts --license
```

```
ZenTTS 1.2.0
Status:  active
Reason:  licence check passed
Control: https://raw.githubusercontent.com/OmorDeveloper/zentts/master/control.json
State:   C:\Users\you\AppData\Local\zentts\license.json
```

The running server reports the same at `GET /v1/license`.

## What this does not do

Be clear-eyed about the limits:

- **It is not tamper-proof.** ZenTTS ships as readable Python. Anyone can set
  `ZENTTS_SKIP_LICENSE_CHECK=1`, edit the file, or pin an old version from
  their own cache. This stops honest users and casual copying, not a
  determined one.
- **It needs the network.** An install that has never checked in will not run,
  so a fully air-gapped machine cannot be served without lifting the check.
- **It can strand paying users.** If GitHub is unreachable for more than seven
  days, working installs stop. Raise `LICENSE_GRACE_DAYS` in `zentts.py` if
  that trade is wrong for you.
- **Users can see it.** The check is visible in the source and reaches out to
  GitHub and PyPI. Say so in your terms, or users will find it themselves.

## Testing the switch without affecting anyone

Point one install at a local file:

```bash
printf '{"enabled": false, "message": "Testing."}' > kill.json
python -m http.server 8399 &
ZENTTS_HOME=./tmp ZENTTS_LICENSE_URL=http://127.0.0.1:8399/kill.json zentts --license
```
