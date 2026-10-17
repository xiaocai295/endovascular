#!/usr/bin/env bash
# 服务器版环境激活脚本 —— 上传到 endovascular/activate.sh
# 所有路径都锚定在脚本所在目录(工程根)，不硬编码机器路径，不乱放环境。

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "source this file: source activate.sh" >&2
    exit 2
fi

_EV_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_EV_VENV="$_EV_PROJECT_ROOT/.venv"
_EV_SOFA="$_EV_PROJECT_ROOT/sofa/SOFA_v23.06.00_Linux"

for required in \
    "$_EV_VENV/bin/python" \
    "$_EV_SOFA/bin/runSofa" \
    "$_EV_SOFA/plugins/SofaPython3/lib/python3/site-packages/Sofa/__init__.py" \
    "$_EV_SOFA/plugins/BeamAdapter/lib/libBeamAdapter.so"; do
    if [[ ! -e "$required" ]]; then
        echo "missing runtime file: $required" >&2
        return 2
    fi
done

if [[ -z "${ENDOVASCULAR_ENV_ACTIVE:-}" ]]; then
    export _EV_OLD_PATH="${PATH:-}"
    export _EV_OLD_PYTHONPATH="${PYTHONPATH:-}"
    export _EV_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
fi

export ENDOVASCULAR_PROJECT_ROOT="$_EV_PROJECT_ROOT"
export ENDOVASCULAR_ENV_ACTIVE=1
export VIRTUAL_ENV="$_EV_VENV"
export SOFA_ROOT="$_EV_SOFA"
export SOFAPYTHON3_ROOT="$SOFA_ROOT/plugins/SofaPython3"
export SOFA_PLUGIN_PATH="$SOFA_ROOT/plugins"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/endovascular-matplotlib-${USER:-user}"

export PATH="$VIRTUAL_ENV/bin:$SOFA_ROOT/bin:${_EV_OLD_PATH:-}"
export PYTHONPATH="$SOFAPYTHON3_ROOT/lib/python3/site-packages:$SOFA_ROOT/plugins/STLIB/lib/python3/site-packages:$ENDOVASCULAR_PROJECT_ROOT/repositories/stEVE:$ENDOVASCULAR_PROJECT_ROOT/repositories/stEVE_bench:$ENDOVASCULAR_PROJECT_ROOT/repositories/stEVE_rl:$ENDOVASCULAR_PROJECT_ROOT"
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$SOFA_ROOT/lib:$SOFAPYTHON3_ROOT/lib:$SOFA_ROOT/plugins/BeamAdapter/lib"

mkdir -p "$MPLCONFIGDIR"
hash -r

deactivate_endovascular() {
    export PATH="${_EV_OLD_PATH:-}"
    export PYTHONPATH="${_EV_OLD_PYTHONPATH:-}"
    export LD_LIBRARY_PATH="${_EV_OLD_LD_LIBRARY_PATH:-}"
    unset ENDOVASCULAR_PROJECT_ROOT ENDOVASCULAR_ENV_ACTIVE VIRTUAL_ENV
    unset SOFA_ROOT SOFAPYTHON3_ROOT SOFA_PLUGIN_PATH
    unset PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE MPLCONFIGDIR
    unset _EV_OLD_PATH _EV_OLD_PYTHONPATH _EV_OLD_LD_LIBRARY_PATH
    unset _EV_PROJECT_ROOT _EV_VENV _EV_SOFA
    unset -f deactivate_endovascular
    hash -r
}

echo "endovascular environment: $VIRTUAL_ENV"
echo "SOFA runtime: $SOFA_ROOT"