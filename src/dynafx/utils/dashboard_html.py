"""Post-processor that makes Plotly dashboard HTML friendly to browsers.

Problem: dashboards embed many figures, several of which use WebGL trace
types (scattergl, scatterpolargl, scattermapbox, mesh3d, scatter3d,
surface, parcoords, heatmap, ...). Every WebGL trace allocates a WebGL
context, and browsers cap *active* contexts at ~16. The dashboards call
``Plotly.newPlot`` for **all** figures at page load (including the
figures inside hidden tab panes), blowing past the limit. The browser then
silently kills the oldest contexts and any later operation on those plots
(the ``Plotly.Plots.resize`` in the tab-switch handler) throws a terse,
message-less ``Error``.

Fix: defer every ``Plotly.newPlot`` call into a per-pane registry and only
render a pane's plots when that tab becomes visible; when leaving a tab we
``Plotly.purge`` its plots so their WebGL contexts are freed. At most one
pane's worth of WebGL contexts is ever alive, comfortably under the limit.

The transform is purely textual over the *already rendered* HTML string, so it
is agnostic to the brace-style (f-string vs ``.format``) each dashboard
used to build the page.
"""

from __future__ import annotations

import re

_PANE_MARKER = '<div class="pane'


def _find_newplot_calls(html: str):
    """Yield (start, end, args) for each top-level ``Plotly.newPlot(...)`` call.

    ``args`` is the source between the outer parentheses (exclusive). A
    quote-aware balanced-parenthesis scan is used so parentheses inside string
    literals (e.g. a title containing "(MW)") do not break matching.
    """
    calls: list[tuple[int, int, str]] = []
    n = len(html)
    i = 0
    while True:
        idx = html.find("Plotly.newPlot(", i)
        if idx == -1:
            break
        paren_open = idx + len("Plotly.newPlot(") - 1  # position of '('
        depth = 0
        in_str: str | None = None
        k = paren_open
        while k < n:
            c = html[k]
            if in_str is not None:
                if c == "\\":
                    k += 2
                    continue
                if c == in_str:
                    in_str = None
                k += 1
                continue
            if c in ('"', "'", "`"):
                in_str = c
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    k += 1
                    break
            k += 1
        args = html[paren_open + 1 : k - 1]
        calls.append((idx, k, args))
        i = k
    return calls


def _div_id(args: str) -> str:
    """Extract the first argument of ``newPlot`` as a JS expression (the div id)."""
    s = args.lstrip()
    if s and s[0] in ('"', "'", "`"):
        q = s[0]
        end = s.find(q, 1)
        if end != -1:
            return s[: end + 1]
    comma = s.find(",")
    return s[:comma].strip() if comma != -1 else s.strip()


def _inject_switchtab(html: str) -> str:
    """Insert a ``dynafxShowPane(param)`` call as the first statement of the
    tab-switch function (named either ``switchTab`` or ``showTab``)."""
    pat = re.compile(r"function (switchTab|showTab)\(([^)]*)\)\s*\{")

    def repl(m: re.Match) -> str:
        name = m.group(1)
        param = m.group(2).strip()
        return f"function {name}({param}){{ dynafxShowPane({param});"

    return pat.sub(repl, html)


_CONTROL_SCRIPT = """
<script>
window.addEventListener('error', function(e){
  console.error('[dashboard]', e.message, (e.error && e.error.stack) || '');
});
window.__dynafxPD = window.__dynafxPD || [];
window.__dynafxCur = -1;
function dynafxRenderPane(i){
  for(var k=0;k<window.__dynafxPD.length;k++){
    var x=window.__dynafxPD[k];
    if(x.p===i && !x.r){
      try{ x.f(); }
      catch(err){ console.error('[dashboard] plot init failed (pane '+i+')', err); }
      x.r=true;
    }
  }
}
function dynafxPurgePane(i){
  if(typeof Plotly==='undefined') return;
  for(var k=0;k<window.__dynafxPD.length;k++){
    var x=window.__dynafxPD[k];
    if(x.p===i && x.r){
      try{ Plotly.purge(x.id); }catch(err){}
      x.r=false;
    }
  }
}
function dynafxSafeResize(el){
  if(typeof Plotly==='undefined' || !el) return;
  requestAnimationFrame(function(){
    try{ Plotly.Plots.resize(el); }
    catch(err){ console.warn('[dashboard] resize skipped', err); }
  });
}
function dynafxShowPane(i){
  if(window.__dynafxCur===i) return;
  if(window.__dynafxCur>=0) dynafxPurgePane(window.__dynafxCur);
  window.__dynafxCur=i;
  dynafxRenderPane(i);
  setTimeout(function(){
    var ps=document.querySelectorAll('.pane');
    for(var t=0;t<ps.length;t++){
      if(!ps[t].classList.contains('hidden')){
        var pl=ps[t].querySelectorAll('.js-plotly-plot');
        for(var u=0;u<pl.length;u++){ dynafxSafeResize(pl[u]); }
      }
    }
  }, 80);
}
window.addEventListener('load', function(){ dynafxShowPane(0); });
</script>
"""


def make_lazy(html: str) -> str:
    """Transform a dashboard HTML string into lazy/per-pane Plotly rendering."""
    has_panes = _PANE_MARKER in html

    # 1) Neutralize existing resize calls so they route through the safe wrapper.
    html = html.replace("Plotly.Plots.resize(", "dynafxSafeResize(")

    # 2) Make switchTab render (and purge) the targeted pane.
    html = _inject_switchtab(html)

    # 3) Defer every Plotly.newPlot into the per-pane registry.
    calls = _find_newplot_calls(html)
    replacements: list[tuple[int, int, str]] = []
    for start, end, args in calls:
        if has_panes:
            pane_idx = html.count(_PANE_MARKER, 0, start) - 1
            if pane_idx < 0:
                pane_idx = 0
        else:
            pane_idx = 0
        divid = _div_id(args)
        repl = (
            "window.__dynafxPD.push({p:"
            + str(pane_idx)
            + ",id:"
            + divid
            + ",f:function(){Plotly.newPlot("
            + args
            + ")}});"
        )
        replacements.append((start, end, repl))
    # Apply in reverse order so earlier offsets stay valid.
    for start, end, repl in sorted(replacements, key=lambda t: t[0], reverse=True):
        html = html[:start] + repl + html[end:]

    # 4) Inject the control script just before </body>.
    if "</body>" in html:
        html = html.replace("</body>", _CONTROL_SCRIPT + "</body>", 1)
    else:
        html = html + _CONTROL_SCRIPT

    return html
