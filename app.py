from fastapi import FastAPI, Header, HTTPException
from typing import Optional, List, Dict
from datetime import date, timedelta

API_KEY = "change-me-please"

app = FastAPI(
    title="Procurement Contract Monitor API",
    version="1.0.0",
    description="Endpoints for contract monitoring and external risk scanning."
)

def demo_data():
    today = date.today()
    contracts = [
        {"contract_id":"PO-0045","supplier":"SteelOne Ltd","sku":"rebar-#4,#6",
         "spec":"ASTM A615 Gr60","due_date":(today+timedelta(days=3)).isoformat(),
         "qty_committed":40000,"qc_method":"MTC + visual"},
        {"contract_id":"MSA-CEM-2025","supplier":"SolidCem S.A.","sku":"cement-42.5R",
         "spec":"EN197-1 42.5R","due_date":(today+timedelta(days=1)).isoformat(),
         "qty_committed":500,"qc_method":"CoA + moisture"},
        {"contract_id":"SLA-RMX-001","supplier":"UrbanMix","sku":"readymix-C30/37",
         "spec":"S3 slump 100–150mm","due_date":(today+timedelta(days=2)).isoformat(),
         "qty_committed":200,"qc_method":"slump + cylinders"}
    ]
    shipments = [
        {"po_id":"PO-0045","sku":"rebar-#4","qty_received":21000,"date_received":(today+timedelta(days=2)).isoformat(),"qc_result":"pass","spec_variance":""},
        {"po_id":"PO-0045","sku":"rebar-#6","qty_received":0,"date_received":"","qc_result":"","spec_variance":"pending"},
        {"po_id":"MSA-CEM-2025","sku":"cement-42.5R","qty_received":0,"date_received":"","qc_result":"","spec_variance":"delayed"},
        {"po_id":"SLA-RMX-001","sku":"readymix-C30/37","qty_received":0,"date_received":"","qc_result":"","spec_variance":"scheduled"}
    ]
    risks = [
        {"time": today.isoformat(), "event":"Port berth maintenance 24–36h delays",
         "affects":"cement-42.5R", "likelihood":"high",
         "impact":"3-day slip risk for MSA-CEM-2025",
         "recommendation":"allocate buffer; consider Altamira trucking"},
        {"time": today.isoformat(), "event":"Heavy rain forecast",
         "affects":"readymix-C30/37", "likelihood":"medium",
         "impact":"dispatch window compression; pour reschedule",
         "recommendation":"shift pour to AM; add covers"}
    ]
    return contracts, shipments, risks

def sum_received(shipments, po_id):
    return sum(s["qty_received"] for s in shipments if s["po_id"] == po_id)

def link_risk(risks, sku):
    hits = [r["event"] for r in risks if r["affects"] in sku]
    return "; ".join(hits) if hits else ""

@app.post("/contract_status", summary="Get status/exceptions", tags=["monitoring"])
def contract_status(body: Dict[str, str], x_api_key: Optional[str] = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    user_input = (body.get("user_input") or "").lower()
    today = date.today().isoformat()
    contracts, shipments, risks = demo_data()

    exceptions = []
    qc_fail = 0
    spec_mismatch = 0

    for c in contracts:
        received = sum_received(shipments, c["contract_id"])
        if received < c["qty_committed"] and c["due_date"] <= today:
            exceptions.append({
                "contract_id": c["contract_id"],
                "sku": c["sku"],
                "issue": "Delayed",
                "due": c["due_date"],
                "eta": None,
                "impact": ("foundation pour risk" if "cement" in c["sku"]
                           else "dispatch/pour window risk" if "readymix" in c["sku"]
                           else "schedule risk") + (f" – {link_risk(risks, c['sku'])}" or "")
            })
        for s in shipments:
            if s["po_id"] == c["contract_id"] and s.get("qc_result") == "fail":
                qc_fail += 1
                exceptions.append({
                    "contract_id": c["contract_id"], "sku": c["sku"],
                    "issue": "QC Failure", "due": c["due_date"], "eta": None,
                    "impact": "quality nonconformance"
                })
            if s["po_id"] == c["contract_id"] and s.get("spec_variance") in ("out_of_spec","mismatch"):
                spec_mismatch += 1
                exceptions.append({
                    "contract_id": c["contract_id"], "sku": c["sku"],
                    "issue": "Spec Mismatch", "due": c["due_date"], "eta": None,
                    "impact": "specification mismatch"
                })

    if user_input:
        filt = [e for e in exceptions if user_input in e["contract_id"].lower() or user_input in e["sku"].lower()]
        if filt:
            exceptions = filt

    delayed = sum(1 for e in exceptions if e["issue"] == "Delayed")
    total = max(1, len(demo_data()[0]))
    on_time_pct = round(100 * (total - delayed) / total)

    kpis = {
        "as_of": today,
        "on_time_pct": on_time_pct,
        "delayed": delayed,
        "qc_failed": qc_fail,
        "spec_mismatch": spec_mismatch
    }

    def line(e):
        return f"- **{e['contract_id']}** · `{e['sku']}` · **{e['issue']}** (due {e['due']}, ETA {'—'}) — _{e['impact']}_ → **Next:** expedite or alternate source" \
               if e["issue"] == "Delayed" else \
               f"- **{e['contract_id']}** · `{e['sku']}` · **{e['issue']}** (due {e['due']}) — _{e['impact']}_ → **Next:** investigate / raise NCR"

    md_ex = "\n".join(line(e) for e in exceptions) if exceptions else "_No exceptions detected._"
    markdown = (
        f"**Contract Status – {today}**\n\n"
        f"**KPIs:** On-time **{kpis['on_time_pct']}%**, Delayed **{kpis['delayed']}**, "
        f"QC failures **{kpis['qc_failed']}**, Spec mismatches **{kpis['spec_mismatch']}**\n\n"
        f"**Exceptions**\n{md_ex}"
    )

    return {"markdown": markdown, "kpis": kpis, "exceptions": exceptions}

@app.post("/risk_scan", summary="Scan external risks", tags=["risk"])
def risk_scan(body: Dict[str, str], x_api_key: Optional[str] = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    q = (body.get("query") or "").lower()
    _, _, risks = demo_data()
    alerts = [r for r in risks if q in r["affects"].lower() or q in r["event"].lower()] if q else risks
    md = "**Supply Chain Risk Alerts**\n\n" + "\n".join(
        f"- **{a['event']}** · affects `{a['affects']}` · Likelihood: **{a['likelihood']}** → _{a['impact']}_ · **Mitigation:** {a['recommendation']}"
        for a in alerts
    )
    return {"markdown": md, "alerts": alerts}
