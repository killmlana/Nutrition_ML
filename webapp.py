# webapp.py
"""Nutrition ML - Child Malnutrition Assessment"""

from pathlib import Path
import base64
import json

import joblib
import pandas as pd
from nicegui import ui, events

from src.database import (
    init_db, add_child, update_child, get_child, get_all_children,
    delete_child, add_assessment, get_assessments, get_latest_assessments,
    get_children_with_latest_risk, get_stats, get_all_face_data,
)
from src.face import encode_face, match_face, FACE_AVAILABLE
from src.rule_engine import check_anemia, check_muac
from src.recommend import acute_text, muac_text, risk_level, build_recommendation

# ── Paths ──

BASE = Path(__file__).resolve().parent
ACUTE_PATH = BASE / "models" / "acute_model.pkl"
STUNT_PATH = BASE / "models" / "stunting_model.pkl"
SCALER_PATH = BASE / "data" / "processed" / "scaler.pkl"
FONTS_DIR = BASE / "fonts"

RISK_COLORS = {"critical": "negative", "high": "warning", "moderate": "info", "low": "positive"}
RISK_ICONS = {"critical": "emergency", "high": "warning", "moderate": "info", "low": "check_circle"}
PRIORITY_COLORS = {"urgent": "negative", "important": "warning", "normal": "positive"}

# ── Theme CSS ──

THEME_CSS = r"""
<style>
@font-face {
  font-family: 'Geist';
  src: url('/fonts/GeistVF.woff2') format('woff2');
  font-weight: 100 900; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'Geist Mono';
  src: url('/fonts/GeistMonoVF.woff2') format('woff2');
  font-weight: 100 900; font-style: normal; font-display: swap;
}

:root {
  --background: 0 0% 100%;
  --foreground: 0 0% 3.9%;
  --primary: 0 0% 9%;
  --primary-foreground: 0 0% 98%;
  --secondary: 0 0% 80.1%;
  --secondary-foreground: 0 0% 9%;
  --muted: 0 0% 80.1%;
  --muted-foreground: 0 0% 45.1%;
  --accent: 0 0% 80.1%;
  --accent-foreground: 0 0% 9%;
  --additive: 112 50% 36%;
  --additive-foreground: 0 0% 9%;
  --destructive: 0 84.2% 60.2%;
  --destructive-foreground: 0 0% 98%;
  --border: 0 0% 89.8%;
  --ring: 0 0% 3.9%;
  --q-primary: hsl(var(--primary));
  --q-secondary: hsl(var(--secondary));
  --q-accent: hsl(var(--accent));
  --q-positive: hsl(var(--additive));
  --q-negative: hsl(var(--destructive));
  --q-info: hsl(var(--muted-foreground));
  --q-warning: hsl(38, 92%, 50%);
}

html.dark, body.body--dark {
  --background: 240 22.7% 8.6%;
  --foreground: 160 100% 45%;
  --primary: 0 0% 98%;
  --primary-foreground: 0 0% 9%;
  --secondary: 0 0% 14.9%;
  --secondary-foreground: 160 100% 45%;
  --muted: 0 0% 14.9%;
  --muted-foreground: 0 0% 63.9%;
  --accent: 0 0% 14.9%;
  --accent-foreground: 0 0% 98%;
  --additive: 112 50% 36%;
  --additive-foreground: 0 0% 9%;
  --destructive: 0 62.8% 30.6%;
  --destructive-foreground: 0 0% 98%;
  --border: 0 0% 14.9%;
  --ring: 0 0% 83.1%;
  --q-primary: hsl(var(--primary));
  --q-secondary: hsl(var(--secondary));
  --q-accent: hsl(var(--accent));
  --q-positive: hsl(var(--additive));
  --q-negative: hsl(var(--destructive));
  --q-info: hsl(var(--muted-foreground));
  --q-warning: hsl(38, 92%, 50%);
}

html, body {
  background: hsl(var(--background));
  color: hsl(var(--foreground));
  font-family: Geist, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}
.q-header {
  background: hsl(var(--background)) !important;
  color: hsl(var(--foreground)) !important;
  border-bottom: 1px solid hsl(var(--border)) !important;
}
.q-card {
  background: hsl(var(--background)) !important;
  color: hsl(var(--foreground)) !important;
  border: 1px solid hsl(var(--border)) !important;
  border-radius: 16px !important;
}
.q-field--outlined .q-field__control { border-color: hsl(var(--border)) !important; }
.q-btn--unelevated { border-radius: 12px !important; }
.q-chip { border-radius: 999px !important; border: 1px solid hsl(var(--border)) !important; }
.nav-link { color: hsl(var(--muted-foreground)); text-decoration: none; font-weight: 500; transition: color 0.15s; }
.nav-link:hover { color: hsl(var(--foreground)); }
.nav-link-active { color: hsl(var(--foreground)) !important; }
.risk-banner { border-left: 4px solid !important; }
.risk-banner-critical { border-left-color: hsl(0 84.2% 60.2%) !important; }
.risk-banner-high { border-left-color: hsl(38, 92%, 50%) !important; }
.risk-banner-moderate { border-left-color: hsl(var(--muted-foreground)) !important; }
.risk-banner-low { border-left-color: hsl(var(--additive)) !important; }
.child-row { border-bottom: 1px solid hsl(var(--border)); transition: background 0.15s; }
.child-row:hover { background: hsl(var(--muted) / 0.3); }
</style>
"""

# ── Initialize ──

init_db()

acute_model = stunting_model = scaler = None
load_error = None
try:
    acute_model = joblib.load(str(ACUTE_PATH))
    stunting_model = joblib.load(str(STUNT_PATH))
    scaler = joblib.load(str(SCALER_PATH))
except Exception as e:
    load_error = str(e)

ui.add_head_html(THEME_CSS, shared=True)

try:
    if FONTS_DIR.exists():
        from fastapi.staticfiles import StaticFiles
        ui.get_app().mount("/fonts", StaticFiles(directory=str(FONTS_DIR)), name="fonts")
except Exception:
    pass


# ── Navigation helper ──

def goto(url):
    try:
        ui.navigate.to(url)
    except AttributeError:
        ui.open(url)


# ── ML Assessment ──

def assess(age, sex, weight, height, muac, hb):
    bmi = weight / ((height / 100.0) ** 2)
    sample_df = pd.DataFrame(
        [[age, sex, weight, height, muac, hb, bmi]],
        columns=["age", "sex", "weight", "height", "muac", "hb", "bmi"],
    )
    sample_scaled = scaler.transform(sample_df)
    wasting_pred = int(acute_model.predict(sample_scaled)[0])
    stunting_pred = int(stunting_model.predict(sample_scaled)[0])
    anemia_flag = int(check_anemia(hb))
    muac_status = check_muac(muac)
    risk = risk_level(wasting_pred, stunting_pred, anemia_flag, muac_status)
    recs = build_recommendation(wasting_pred, stunting_pred, anemia_flag, muac_status)
    return {
        "wasting_pred": wasting_pred,
        "wasting_text": acute_text(wasting_pred),
        "stunting_pred": stunting_pred,
        "anemia_flag": anemia_flag,
        "muac_status": muac_status,
        "muac_text": muac_text(muac_status),
        "bmi": bmi,
        "risk_level": risk,
        "recommendations": recs,
    }


# ── Shared Header ──

def build_header(active="/"):
    ui.page_title("Nutrition ML")
    dm = ui.dark_mode(value=True)

    def set_dark(v):
        dm.value = v
        ui.run_javascript(f"document.documentElement.classList.toggle('dark', {str(v).lower()});")

    ui.run_javascript("document.documentElement.classList.add('dark');")

    with ui.header().classes("items-center justify-between q-px-md"):
        with ui.row().classes("items-center gap-4"):
            ui.icon("health_and_safety").classes("text-h6")
            ui.label("Nutrition ML").classes("text-h6")
            ui.separator().props("vertical inset")
            cls_assess = "nav-link nav-link-active" if active == "/" else "nav-link"
            cls_admin = "nav-link nav-link-active" if active == "/admin" else "nav-link"
            cls_live = "nav-link nav-link-active" if active == "/live" else "nav-link"
            ui.link("Assess", "/").classes(cls_assess)
            ui.link("Admin", "/admin").classes(cls_admin)
            ui.link("Live", "/live").classes(cls_live)
        with ui.row().classes("items-center gap-2"):
            face_status = "on" if FACE_AVAILABLE else "off"
            ui.icon("face").classes("opacity-50").tooltip(f"Face recognition: {face_status}")
            ui.switch("Dark", value=True, on_change=lambda e: set_dark(bool(e.value))).props("dense dark")


# ── Render recommendation cards ──

def render_recs(recs, container):
    container.clear()
    with container:
        for rec in recs:
            pcolor = PRIORITY_COLORS.get(rec["priority"], "info")
            with ui.card().classes("w-full q-mt-sm"):
                with ui.row().classes("items-center gap-2"):
                    ui.chip(rec["priority"].upper(), icon="flag").props(f"dense color={pcolor}")
                    ui.label(rec["category"]).classes("text-subtitle1 font-bold")
                ui.label(rec["text"]).classes("q-mt-xs")
                if rec.get("details"):
                    with ui.column().classes("q-mt-xs q-pl-md gap-1"):
                        for d in rec["details"]:
                            with ui.row().classes("items-start gap-1"):
                                ui.icon("arrow_right").classes("text-sm opacity-70 q-mt-xs")
                                ui.label(d).classes("text-body2")


# ════════════════════════════════════════════════════════════════
#  ASSESSMENT PAGE
# ════════════════════════════════════════════════════════════════

@ui.page("/")
def assessment_page(child: int = 0):
    build_header(active="/")

    if load_error:
        with ui.column().classes("w-full max-w-3xl mx-auto q-pa-md"):
            with ui.card().classes("w-full"):
                ui.icon("error").classes("text-negative text-h4")
                ui.label("Model files not found").classes("text-h6")
                ui.separator()
                ui.markdown(f"```\n{load_error}\n```")
        return

    state = {"child_id": None, "photo_bytes": None, "face_encoding": None}

    with ui.column().classes("w-full max-w-3xl mx-auto q-pa-md gap-4"):

        # ── Identify Child ──
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("face").classes("text-h6")
                ui.label("Identify Child").classes("text-h6")
            ui.separator()

            photo_display = ui.row().classes("hidden q-mt-sm")
            match_status = ui.label("").classes("hidden q-mt-sm")

            children_list = get_all_children()
            profile_options = {0: "-- New child --"}
            for c in children_list:
                profile_options[c["id"]] = c["name"]

            async def handle_upload(e: events.UploadEventArguments):
                content = await e.file.read()

                # Reset form state for new upload
                state["child_id"] = None
                state["photo_bytes"] = content
                state["face_encoding"] = None
                name_in.value = ""
                sex_in.value = 0
                guardian_in.value = ""
                contact_in.value = ""
                profile_sel.value = 0
                result_container.classes(add="hidden")
                upload_el.reset()

                b64 = base64.b64encode(content).decode()
                photo_display.classes(remove="hidden")
                photo_display.clear()
                with photo_display:
                    ui.image(f"data:image/jpeg;base64,{b64}").classes("w-32 h-32 rounded")

                if FACE_AVAILABLE:
                    encoding = encode_face(content)
                    if encoding is not None:
                        state["face_encoding"] = encoding
                        face_data = get_all_face_data()
                        if face_data:
                            match_id, similarity = match_face(encoding, face_data)
                            if match_id is not None:
                                child = get_child(match_id)
                                state["child_id"] = match_id
                                name_in.value = child["name"]
                                sex_in.value = child["sex"]
                                guardian_in.value = child.get("guardian_name", "")
                                contact_in.value = child.get("guardian_contact", "")
                                # Update dropdown options to include this child before setting value
                                if match_id not in profile_sel.options:
                                    profile_sel.options[match_id] = child["name"]
                                    profile_sel.update()
                                profile_sel.value = match_id
                                acount = len(get_assessments(match_id))
                                match_status.text = f"Matched: {child['name']} (confidence: {similarity:.0%}, {acount} prior assessments)"
                                match_status.classes(remove="hidden")
                                ui.notify(f"Face matched: {child['name']}", type="positive")
                                return
                        match_status.text = "No match found. Fill in details for a new profile."
                        match_status.classes(remove="hidden")
                    else:
                        match_status.text = "No face detected in image. Photo saved."
                        match_status.classes(remove="hidden")
                else:
                    match_status.text = "Face recognition unavailable. Photo saved for records."
                    match_status.classes(remove="hidden")

            with ui.row().classes("w-full gap-4 items-start"):
                upload_el = ui.upload(
                    on_upload=handle_upload,
                    auto_upload=True,
                    max_files=1,
                    max_file_size=10_000_000,
                    label="Upload child photo (max 10MB)",
                ).props("accept=image/*").classes("w-64")
                photo_display

            match_status

            ui.label("-- or select existing profile --").classes("text-center opacity-50 q-mt-sm")

            def on_profile_select(e):
                cid = e.value
                if cid and cid != 0:
                    child = get_child(cid)
                    if child:
                        state["child_id"] = child["id"]
                        name_in.value = child["name"]
                        sex_in.value = child["sex"]
                        guardian_in.value = child.get("guardian_name", "")
                        contact_in.value = child.get("guardian_contact", "")
                        if child.get("photo"):
                            b64 = base64.b64encode(child["photo"]).decode()
                            photo_display.classes(remove="hidden")
                            photo_display.clear()
                            with photo_display:
                                ui.image(f"data:image/jpeg;base64,{b64}").classes("w-32 h-32 rounded")
                        acount = len(get_assessments(cid))
                        match_status.text = f"Selected: {child['name']} ({acount} prior assessments)"
                        match_status.classes(remove="hidden")
                else:
                    state["child_id"] = None
                    name_in.value = ""
                    sex_in.value = 0
                    guardian_in.value = ""
                    contact_in.value = ""
                    match_status.classes(add="hidden")
                    photo_display.classes(add="hidden")

            profile_sel = ui.select(
                profile_options,
                value=0,
                label="Select existing profile",
                on_change=on_profile_select,
            ).props("outlined dense").classes("w-full max-w-sm q-mt-xs")

        # ── Child Information ──
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("person").classes("text-h6")
                ui.label("Child Information").classes("text-h6")
            ui.separator()

            with ui.row().classes("w-full gap-3"):
                name_in = ui.input("Name", value="").props("outlined dense").classes("flex-1")
                sex_in = ui.radio({0: "Female", 1: "Male"}, value=0).props("inline dense color=primary").classes("q-mt-sm")

            with ui.row().classes("w-full gap-3"):
                guardian_in = ui.input("Guardian Name", value="").props("outlined dense").classes("flex-1")
                contact_in = ui.input("Contact", value="").props("outlined dense").classes("flex-1")

        # ── Measurements ──
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("straighten").classes("text-h6")
                ui.label("Measurements").classes("text-h6")
            ui.separator()

            with ui.row().classes("w-full gap-3"):
                age_in = ui.number("Age (months)", min=6, max=59, step=1, value=24).props("outlined dense").classes("w-44")
                with age_in.add_slot("prepend"):
                    ui.icon("calendar_month")
                wt_in = ui.number("Weight (kg)", min=0, step=0.1, value=12.0).props("outlined dense").classes("w-44")
                with wt_in.add_slot("prepend"):
                    ui.icon("monitor_weight")

            with ui.row().classes("w-full gap-3"):
                ht_in = ui.number("Height (cm)", min=0, step=0.1, value=85.0).props("outlined dense").classes("w-44")
                with ht_in.add_slot("prepend"):
                    ui.icon("straighten")
                muac_in = ui.number("MUAC (mm)", min=0, step=0.1, value=130.0).props("outlined dense").classes("w-44")
                with muac_in.add_slot("prepend"):
                    ui.icon("fitness_center")

            with ui.row().classes("w-full gap-3"):
                hb_in = ui.number("Hemoglobin (g/dL)", min=0, step=0.1, value=11.5).props("outlined dense").classes("w-44")
                with hb_in.add_slot("prepend"):
                    ui.icon("bloodtype")

        # ── Results container (hidden until assessed) ──
        result_container = ui.column().classes("w-full gap-3 hidden")

        def show_results(r):
            result_container.classes(remove="hidden")
            result_container.clear()

            risk = r["risk_level"]
            rcolor = RISK_COLORS.get(risk, "info")
            ricon = RISK_ICONS.get(risk, "info")

            with result_container:
                # Risk banner
                with ui.card().classes(f"w-full risk-banner risk-banner-{risk}"):
                    with ui.row().classes("items-center gap-3"):
                        ui.icon(ricon).classes(f"text-h4 text-{rcolor}")
                        with ui.column().classes("gap-0"):
                            ui.label(f"Risk Level: {risk.upper()}").classes("text-h6 font-bold")
                            if risk in ("critical", "high"):
                                ui.label("Immediate attention required").classes("text-negative opacity-90")
                            elif risk == "moderate":
                                ui.label("Monitor closely and follow up").classes("opacity-70")
                            else:
                                ui.label("Continue routine monitoring").classes("opacity-70")

                    ui.separator()

                    with ui.row().classes("w-full gap-2 q-mt-sm flex-wrap"):
                        wc = "negative" if r["wasting_pred"] else "positive"
                        ui.chip(f"Wasting: {r['wasting_text']}", icon="warning").props(f"outline color={wc}")
                        mc = "negative" if r["muac_status"] != "normal" else "positive"
                        ui.chip(f"MUAC: {r['muac_text']}", icon="fitness_center").props(f"outline color={mc}")
                        sc = "negative" if r["stunting_pred"] else "positive"
                        ui.chip(f"Stunting: {'Yes' if r['stunting_pred'] else 'No'}", icon="height").props(f"outline color={sc}")
                        ac = "negative" if r["anemia_flag"] else "positive"
                        ui.chip(f"Anemia: {'Yes' if r['anemia_flag'] else 'No'}", icon="bloodtype").props(f"outline color={ac}")
                        ui.chip(f"BMI: {r['bmi']:.1f}", icon="insights").props("outline color=accent")

                # Recommendations
                with ui.card().classes("w-full"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("restaurant_menu").classes("text-h6")
                        ui.label("Recommendations").classes("text-h6")
                    ui.separator()
                    recs_container = ui.column().classes("w-full")
                    render_recs(r["recommendations"], recs_container)

        def on_submit():
            try:
                age = int(age_in.value)
                sex = int(sex_in.value)
                weight = float(wt_in.value)
                height = float(ht_in.value)
                muac = float(muac_in.value)
                hb = float(hb_in.value)

                if height <= 0 or weight <= 0:
                    ui.notify("Height and weight must be > 0", type="negative")
                    return

                r = assess(age, sex, weight, height, muac, hb)

                # Save profile if new child
                if state["child_id"] is None:
                    name = name_in.value.strip()
                    if not name:
                        ui.notify("Enter child name for new profile", type="warning")
                        return
                    face_enc = state.get("face_encoding")
                    enc_bytes = face_enc.tobytes() if face_enc is not None else None
                    state["child_id"] = add_child(
                        name=name,
                        sex=sex,
                        guardian_name=guardian_in.value.strip(),
                        guardian_contact=contact_in.value.strip(),
                        face_encoding=enc_bytes,
                        photo=state.get("photo_bytes"),
                    )
                    # Add new child to dropdown so future face matches can set it
                    profile_sel.options[state["child_id"]] = name
                    profile_sel.update()

                add_assessment(
                    child_id=state["child_id"],
                    age_months=age,
                    weight=weight,
                    height=height,
                    muac=muac,
                    hb=hb,
                    bmi=r["bmi"],
                    wasting_pred=r["wasting_pred"],
                    stunting_pred=r["stunting_pred"],
                    anemia_flag=r["anemia_flag"],
                    risk=r["risk_level"],
                    recommendations=r["recommendations"],
                )

                show_results(r)
                ui.notify("Assessment complete and saved", type="positive")
            except Exception as e:
                ui.notify(f"Error: {e}", type="negative")

        def on_reset():
            state["child_id"] = None
            state["photo_bytes"] = None
            state["face_encoding"] = None
            name_in.value = ""
            sex_in.value = 0
            guardian_in.value = ""
            contact_in.value = ""
            age_in.value = 24
            wt_in.value = 12.0
            ht_in.value = 85.0
            muac_in.value = 130.0
            hb_in.value = 11.5
            profile_sel.value = 0
            upload_el.reset()
            result_container.classes(add="hidden")
            photo_display.classes(add="hidden")
            match_status.classes(add="hidden")

        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Reset", on_click=on_reset, icon="restart_alt").props("outline color=secondary")
            ui.button("Assess", on_click=on_submit, icon="analytics").props("unelevated color=primary")

        result_container

        # Pre-fill from query param (e.g. navigated from Live page)
        if child and child > 0:
            prefill = get_child(child)
            if prefill:
                state["child_id"] = prefill["id"]
                name_in.value = prefill["name"]
                sex_in.value = prefill["sex"]
                guardian_in.value = prefill.get("guardian_name", "")
                contact_in.value = prefill.get("guardian_contact", "")
                if child not in profile_sel.options:
                    profile_sel.options[child] = prefill["name"]
                    profile_sel.update()
                profile_sel.value = child


# ════════════════════════════════════════════════════════════════
#  ADMIN DASHBOARD
# ════════════════════════════════════════════════════════════════

@ui.page("/admin")
def admin_page():
    build_header(active="/admin")

    with ui.column().classes("w-full max-w-5xl mx-auto q-pa-md gap-4"):

        # ── Summary Stats ──
        stats = get_stats()
        with ui.row().classes("w-full gap-3 flex-wrap"):
            for label, value, color, icon in [
                ("Total Children", stats["total"], "primary", "groups"),
                ("Critical", stats["critical"], "negative", "emergency"),
                ("High Risk", stats["high"], "warning", "warning"),
                ("Moderate", stats["moderate"], "info", "info"),
                ("Low Risk", stats["low"], "positive", "check_circle"),
            ]:
                with ui.card().classes("flex-1 min-w-[140px]"):
                    with ui.column().classes("items-center gap-1 q-pa-sm"):
                        ui.icon(icon).classes(f"text-h4 text-{color}")
                        ui.label(str(value)).classes("text-h4 font-bold")
                        ui.label(label).classes("opacity-70 text-caption")

        # ── High Risk Children ──
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("emergency").classes("text-h6 text-negative")
                ui.label("High Risk Children").classes("text-h6")
            ui.separator()

            high_risk = get_latest_assessments(risk_filter=["critical", "high"])

            if not high_risk:
                ui.label("No high-risk children.").classes("opacity-70 q-pa-sm")
            else:
                for entry in high_risk:
                    risk = entry.get("risk_level", "unknown")
                    rcolor = RISK_COLORS.get(risk, "info")

                    with ui.card().classes(f"w-full q-mt-sm risk-banner risk-banner-{risk} cursor-pointer").on(
                        "click", lambda _, cid=entry["child_id"]: goto(f"/admin/child/{cid}")
                    ):
                        with ui.row().classes("items-center justify-between w-full"):
                            with ui.row().classes("items-center gap-3"):
                                if entry.get("photo"):
                                    b64 = base64.b64encode(entry["photo"]).decode()
                                    ui.image(f"data:image/jpeg;base64,{b64}").classes("w-10 h-10 rounded-full")
                                else:
                                    ui.icon("person").classes("text-h5 opacity-50")

                                with ui.column().classes("gap-0"):
                                    ui.label(entry["name"]).classes("font-bold")
                                    age_text = f"Age: {entry.get('age_months', '?')}m"
                                    date_text = entry.get("assessed_at", "?")[:10]
                                    ui.label(f"{age_text} | Last assessed: {date_text}").classes("opacity-70 text-caption")

                            with ui.row().classes("gap-2 items-center"):
                                ui.chip(risk.upper()).props(f"dense color={rcolor}")
                                if entry.get("wasting_pred"):
                                    ui.chip("Wasted", icon="warning").props("dense outline color=negative")
                                if entry.get("stunting_pred"):
                                    ui.chip("Stunted", icon="height").props("dense outline color=warning")
                                if entry.get("anemia_flag"):
                                    ui.chip("Anemic", icon="bloodtype").props("dense outline color=negative")
                                ui.icon("chevron_right").classes("opacity-50")

        # ── All Profiles ──
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("people").classes("text-h6")
                ui.label("All Profiles").classes("text-h6")
            ui.separator()

            all_children = get_children_with_latest_risk()

            if not all_children:
                ui.label("No profiles yet. Assess a child to create one.").classes("opacity-70 q-pa-sm")
            else:
                list_container = ui.column().classes("w-full gap-0")

                def render_children(filter_text=""):
                    list_container.clear()
                    filtered = all_children
                    if filter_text:
                        filtered = [c for c in all_children if filter_text.lower() in c["name"].lower()]

                    with list_container:
                        if not filtered:
                            ui.label("No matches.").classes("opacity-70 q-pa-sm")
                        else:
                            for child in filtered:
                                risk = child.get("risk_level") or None
                                rcolor = RISK_COLORS.get(risk, "grey") if risk else "grey"

                                with ui.row().classes(
                                    "w-full items-center justify-between q-pa-sm child-row cursor-pointer"
                                ).on("click", lambda _, cid=child["id"]: goto(f"/admin/child/{cid}")):
                                    with ui.row().classes("items-center gap-3"):
                                        ui.icon("person").classes("opacity-50")
                                        ui.label(child["name"]).classes("font-bold")
                                        ui.label("F" if child["sex"] == 0 else "M").classes("opacity-50 text-caption")
                                        if child.get("guardian_name"):
                                            ui.label(f"({child['guardian_name']})").classes("opacity-40 text-caption")

                                    with ui.row().classes("items-center gap-2"):
                                        count = child.get("assessment_count", 0)
                                        if count:
                                            ui.label(f"{count} assessments").classes("opacity-50 text-caption")
                                        if risk:
                                            ui.chip(risk.upper()).props(f"dense color={rcolor}")
                                        else:
                                            ui.chip("NEW").props("dense outline color=grey")
                                        ui.icon("chevron_right").classes("opacity-50")

                ui.input(
                    label="Search by name",
                    on_change=lambda e: render_children(e.value or ""),
                ).props("outlined dense clearable").classes("w-full max-w-sm q-mb-sm")

                render_children()


# ════════════════════════════════════════════════════════════════
#  CHILD DETAIL PAGE
# ════════════════════════════════════════════════════════════════

@ui.page("/admin/child/{child_id}")
def child_detail_page(child_id: int):
    build_header(active="/admin")

    child = get_child(child_id)
    if not child:
        with ui.column().classes("w-full max-w-3xl mx-auto q-pa-md"):
            ui.label("Child not found.").classes("text-h6")
            ui.link("Back to Admin", "/admin")
        return

    assessments_list = get_assessments(child_id)
    latest = assessments_list[0] if assessments_list else None

    with ui.column().classes("w-full max-w-3xl mx-auto q-pa-md gap-4"):

        # ── Back link ──
        ui.link("< Back to Admin", "/admin").classes("nav-link")

        # ── Profile Card ──
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center justify-between w-full"):
                with ui.row().classes("items-center gap-4"):
                    if child.get("photo"):
                        b64 = base64.b64encode(child["photo"]).decode()
                        ui.image(f"data:image/jpeg;base64,{b64}").classes("w-24 h-24 rounded-full")
                    else:
                        ui.icon("person").classes("text-h2 opacity-30")

                    with ui.column().classes("gap-1"):
                        ui.label(child["name"]).classes("text-h5 font-bold")
                        ui.label(f"Sex: {'Female' if child['sex'] == 0 else 'Male'}").classes("opacity-70")
                        if child.get("guardian_name"):
                            ui.label(f"Guardian: {child['guardian_name']}").classes("opacity-70")
                        if child.get("guardian_contact"):
                            ui.label(f"Contact: {child['guardian_contact']}").classes("opacity-70")

                with ui.column().classes("items-end gap-2"):
                    if latest:
                        risk = latest.get("risk_level", "unknown")
                        rcolor = RISK_COLORS.get(risk, "info")
                        ui.chip(f"Risk: {risk.upper()}").props(f"color={rcolor}")

                    with ui.row().classes("gap-1"):
                        # Edit button
                        def open_edit():
                            with ui.dialog() as dlg, ui.card().classes("w-96"):
                                ui.label("Edit Profile").classes("text-h6 q-mb-sm")
                                ne = ui.input("Name", value=child["name"]).props("outlined dense").classes("w-full")
                                se = ui.radio({0: "Female", 1: "Male"}, value=child["sex"]).props("inline dense")
                                ge = ui.input("Guardian", value=child.get("guardian_name", "")).props("outlined dense").classes("w-full")
                                ce = ui.input("Contact", value=child.get("guardian_contact", "")).props("outlined dense").classes("w-full")

                                def save_edit():
                                    update_child(child_id, name=ne.value, sex=se.value,
                                                 guardian_name=ge.value, guardian_contact=ce.value)
                                    dlg.close()
                                    goto(f"/admin/child/{child_id}")

                                with ui.row().classes("justify-end gap-2 q-mt-md"):
                                    ui.button("Cancel", on_click=dlg.close).props("flat")
                                    ui.button("Save", on_click=save_edit).props("unelevated color=primary")
                            dlg.open()

                        ui.button(icon="edit", on_click=open_edit).props("flat dense round")

                        # Delete button
                        def open_delete():
                            with ui.dialog() as dlg, ui.card():
                                ui.label(f"Delete {child['name']}?").classes("text-h6")
                                ui.label("This will remove all assessments permanently.").classes("opacity-70 q-mt-sm")
                                with ui.row().classes("justify-end gap-2 q-mt-md"):
                                    ui.button("Cancel", on_click=dlg.close).props("flat")
                                    ui.button(
                                        "Delete",
                                        on_click=lambda: (delete_child(child_id), dlg.close(), goto("/admin")),
                                    ).props("unelevated color=negative")
                            dlg.open()

                        ui.button(icon="delete", on_click=open_delete).props("flat dense round color=negative")

        # ── Current Recommendations ──
        if latest:
            try:
                recs = json.loads(latest["recommendations"]) if isinstance(latest["recommendations"], str) else latest["recommendations"]
            except (json.JSONDecodeError, TypeError):
                recs = []

            if recs:
                with ui.card().classes("w-full"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("restaurant_menu").classes("text-h6")
                        ui.label("Current Recommendations").classes("text-h6")
                    ui.separator()
                    recs_box = ui.column().classes("w-full")
                    render_recs(recs, recs_box)

        # ── Assessment History ──
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("history").classes("text-h6")
                ui.label(f"Assessment History ({len(assessments_list)})").classes("text-h6")
            ui.separator()

            if not assessments_list:
                ui.label("No assessments yet.").classes("opacity-70 q-pa-sm")
            else:
                columns = [
                    {"name": "date", "label": "Date", "field": "date", "align": "left", "sortable": True},
                    {"name": "age", "label": "Age (m)", "field": "age", "align": "center"},
                    {"name": "weight", "label": "Wt (kg)", "field": "weight", "align": "center"},
                    {"name": "height", "label": "Ht (cm)", "field": "height", "align": "center"},
                    {"name": "bmi", "label": "BMI", "field": "bmi", "align": "center"},
                    {"name": "wasting", "label": "Wasting", "field": "wasting", "align": "center"},
                    {"name": "stunting", "label": "Stunting", "field": "stunting", "align": "center"},
                    {"name": "anemia", "label": "Anemia", "field": "anemia", "align": "center"},
                    {"name": "risk", "label": "Risk", "field": "risk", "align": "center"},
                ]
                rows = []
                for a in assessments_list:
                    rows.append({
                        "date": (a.get("assessed_at") or "")[:10],
                        "age": a.get("age_months", ""),
                        "weight": f"{a['weight']:.1f}",
                        "height": f"{a['height']:.1f}",
                        "bmi": f"{a['bmi']:.1f}",
                        "wasting": "Yes" if a["wasting_pred"] else "No",
                        "stunting": "Yes" if a["stunting_pred"] else "No",
                        "anemia": "Yes" if a["anemia_flag"] else "No",
                        "risk": (a.get("risk_level") or "").upper(),
                    })

                ui.table(columns=columns, rows=rows, row_key="date").props("dense flat bordered").classes("w-full q-mt-sm")

                # Trend summary
                if len(assessments_list) >= 2:
                    prev = assessments_list[1]
                    curr = assessments_list[0]
                    with ui.card().classes("w-full q-mt-sm"):
                        ui.label("Trend (vs. previous assessment)").classes("text-subtitle2 font-bold")
                        with ui.row().classes("gap-4 q-mt-xs flex-wrap"):
                            wt_diff = curr["weight"] - prev["weight"]
                            wt_icon = "trending_up" if wt_diff > 0 else "trending_down" if wt_diff < 0 else "trending_flat"
                            with ui.row().classes("items-center gap-1"):
                                ui.icon(wt_icon).classes("text-sm")
                                ui.label(f"Weight: {wt_diff:+.1f} kg").classes("text-caption")

                            ht_diff = curr["height"] - prev["height"]
                            ht_icon = "trending_up" if ht_diff > 0 else "trending_down" if ht_diff < 0 else "trending_flat"
                            with ui.row().classes("items-center gap-1"):
                                ui.icon(ht_icon).classes("text-sm")
                                ui.label(f"Height: {ht_diff:+.1f} cm").classes("text-caption")

                            risk_order = {"low": 0, "moderate": 1, "high": 2, "critical": 3}
                            prev_r = risk_order.get(prev.get("risk_level", ""), 0)
                            curr_r = risk_order.get(curr.get("risk_level", ""), 0)
                            if curr_r < prev_r:
                                ui.chip("Risk improving", icon="trending_down").props("dense outline color=positive")
                            elif curr_r > prev_r:
                                ui.chip("Risk worsening", icon="trending_up").props("dense outline color=negative")
                            else:
                                ui.chip("Risk stable", icon="trending_flat").props("dense outline")


# ════════════════════════════════════════════════════════════════
#  LIVE RECOGNITION PAGE
# ════════════════════════════════════════════════════════════════

WEBCAM_INIT_JS = """
(async () => {
    window._camError = null;
    window._lastFrame = null;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        window._camError = 'Camera API unavailable (HTTPS required on non-localhost)';
        return;
    }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
        });

        const slot = document.querySelector('.webcam-slot');
        if (!slot) { window._camError = 'mount point not found'; return; }

        const wrap = document.createElement('div');
        wrap.style.cssText = 'position:relative;display:inline-block;width:100%;max-width:640px;';

        const video = document.createElement('video');
        video.autoplay = true;
        video.playsInline = true;
        video.muted = true;
        video.style.cssText = 'width:100%;border-radius:8px;background:#111;display:block;';
        video.srcObject = stream;
        wrap.appendChild(video);

        const box = document.createElement('div');
        box.id = 'guide-box';
        box.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:55%;height:70%;border:2px dashed hsl(112,50%,36%);border-radius:8px;pointer-events:none;transition:border-color .3s;';
        wrap.appendChild(box);

        slot.appendChild(wrap);
        await video.play();

        window._webcamVideo = video;
        window._facingMode = 'user';

        const canvas = document.createElement('canvas');
        setInterval(() => {
            if (video.readyState < 2) return;
            const vw = video.videoWidth, vh = video.videoHeight;
            if (!vw || !vh) return;
            const cw = Math.round(vw * 0.55), ch = Math.round(vh * 0.7);
            const cx = Math.round((vw - cw) / 2), cy = Math.round((vh - ch) / 2);
            canvas.width = cw; canvas.height = ch;
            canvas.getContext('2d').drawImage(video, cx, cy, cw, ch, 0, 0, cw, ch);
            window._lastFrame = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
        }, 1500);

        window.switchCamera = async function() {
            try {
                const v = window._webcamVideo;
                if (v && v.srcObject) v.srcObject.getTracks().forEach(t => t.stop());
                window._facingMode = window._facingMode === 'user' ? 'environment' : 'user';
                const s = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: window._facingMode }
                });
                if (v) { v.srcObject = s; await v.play(); }
                window._lastFrame = null;
            } catch(e) { window._camError = e.message; }
        };

    } catch (err) {
        window._camError = err.message;
    }
})();
"""

CAPTURE_JS = """
(() => {
    if (window._camError) return 'err:' + window._camError;
    return window._lastFrame || 'loading';
})()
"""


@ui.page("/live")
def live_page():
    build_header(active="/live")

    if not FACE_AVAILABLE:
        with ui.column().classes("w-full max-w-3xl mx-auto q-pa-md"):
            with ui.card().classes("w-full"):
                ui.icon("error").classes("text-negative text-h4")
                ui.label("Face recognition is not available.").classes("text-h6")
        return

    state = {"polling": False, "last_match": None}

    with ui.column().classes("w-full max-w-3xl mx-auto q-pa-md gap-4"):

        # ── Webcam Card ──
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center justify-between w-full"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("videocam").classes("text-h6")
                    ui.label("Live Recognition").classes("text-h6")
                with ui.row().classes("items-center gap-1"):
                    assess_btn = ui.button(
                        "New Assessment", icon="add_circle",
                        on_click=lambda: goto(f"/?child={state['last_match']}"),
                    ).props("unelevated dense color=primary").classes("hidden")
                    ui.button(
                        icon="flip_camera_ios",
                        on_click=lambda: ui.run_javascript("if(window.switchCamera)switchCamera();"),
                    ).props("flat dense round").tooltip("Switch camera")
            ui.separator()

            ui.element("div").classes("webcam-slot")

            status_label = ui.label("Starting camera...").classes("q-mt-sm opacity-70")

            ui.run_javascript(WEBCAM_INIT_JS)

        # ── Result Cards (hidden initially) ──
        child_card = ui.card().classes("w-full hidden")
        recs_card = ui.card().classes("w-full hidden")

        async def poll_face():
            if state["polling"]:
                return
            state["polling"] = True
            try:
                result = await ui.run_javascript(CAPTURE_JS, timeout=5.0)

                if not result or result == "loading":
                    status_label.text = "Starting camera..."
                    return
                if isinstance(result, str) and result.startswith("err:"):
                    status_label.text = f"Camera error: {result[4:]}"
                    await ui.run_javascript(
                        "var b=document.getElementById('guide-box');if(b)b.style.borderColor='hsl(0,84%,60%)';"
                    )
                    return

                image_bytes = base64.b64decode(result)
                encoding = encode_face(image_bytes)

                if encoding is None:
                    status_label.text = "No face detected — position face inside the box"
                    await ui.run_javascript(
                        "var b=document.getElementById('guide-box');if(b)b.style.borderColor='hsl(0,84%,60%)';"
                    )
                    if state["last_match"] is not None:
                        child_card.classes(add="hidden")
                        recs_card.classes(add="hidden")
                        assess_btn.classes(add="hidden")
                        state["last_match"] = None
                    return

                face_data = get_all_face_data()
                if not face_data:
                    status_label.text = "Face detected — no profiles in database"
                    await ui.run_javascript(
                        "var b=document.getElementById('guide-box');if(b)b.style.borderColor='hsl(38,92%,50%)';"
                    )
                    return

                match_id, similarity = match_face(encoding, face_data)

                if match_id is None:
                    status_label.text = "Face detected — no match found"
                    await ui.run_javascript(
                        "var b=document.getElementById('guide-box');if(b)b.style.borderColor='hsl(38,92%,50%)';"
                    )
                    if state["last_match"] is not None:
                        child_card.classes(add="hidden")
                        recs_card.classes(add="hidden")
                        assess_btn.classes(add="hidden")
                        state["last_match"] = None
                    return

                # ── Match found ──
                await ui.run_javascript(
                    "var b=document.getElementById('guide-box');if(b)b.style.borderColor='hsl(112,50%,36%)';"
                )

                # Skip re-render if same child is still in frame
                if state["last_match"] == match_id:
                    status_label.text = f"Matched: {get_child(match_id)['name']} (confidence: {similarity:.0%})"
                    return

                state["last_match"] = match_id
                child = get_child(match_id)
                assessments_list = get_assessments(match_id)
                latest = assessments_list[0] if assessments_list else None

                status_label.text = f"Matched: {child['name']} (confidence: {similarity:.0%})"
                assess_btn.classes(remove="hidden")

                # ── Show child info ──
                child_card.classes(remove="hidden")
                child_card.clear()
                with child_card:
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("person").classes("text-h6")
                        ui.label("Child Profile").classes("text-h6")
                    ui.separator()

                    with ui.row().classes("items-center gap-4 q-mt-sm"):
                        if child.get("photo"):
                            b64p = base64.b64encode(child["photo"]).decode()
                            ui.image(f"data:image/jpeg;base64,{b64p}").classes("w-16 h-16 rounded-full")
                        else:
                            ui.icon("person").classes("text-h3 opacity-30")

                        with ui.column().classes("gap-0"):
                            ui.label(child["name"]).classes("text-h6 font-bold")
                            ui.label(f"{'Female' if child['sex'] == 0 else 'Male'}").classes("opacity-70")
                            if child.get("guardian_name"):
                                ui.label(f"Guardian: {child['guardian_name']}").classes("opacity-70 text-caption")

                        with ui.column().classes("items-end gap-1 ml-auto"):
                            if latest:
                                risk = latest.get("risk_level", "unknown")
                                rcolor = RISK_COLORS.get(risk, "info")
                                ui.chip(f"Risk: {risk.upper()}").props(f"color={rcolor}")
                                ui.label(f"{len(assessments_list)} assessments").classes("opacity-50 text-caption")
                            else:
                                ui.chip("No assessment").props("outline color=grey")

                    ui.link("View full profile →", f"/admin/child/{match_id}").classes("nav-link q-mt-sm")

                # ── Show recommendations ──
                recs_card.classes(remove="hidden")
                recs_card.clear()
                with recs_card:
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("restaurant_menu").classes("text-h6")
                        ui.label("Recommended Diet").classes("text-h6")
                    ui.separator()

                    if latest:
                        try:
                            recs = json.loads(latest["recommendations"]) if isinstance(latest["recommendations"], str) else latest["recommendations"]
                        except (json.JSONDecodeError, TypeError):
                            recs = []
                        if recs:
                            recs_box = ui.column().classes("w-full")
                            render_recs(recs, recs_box)
                        else:
                            ui.label("No recommendations on record.").classes("opacity-70 q-mt-sm")
                    else:
                        ui.label("No assessment on record yet.").classes("opacity-70 q-mt-sm")
                        ui.link("Go to Assess →", "/").classes("nav-link")

            except Exception as e:
                status_label.text = f"Error: {e}"
            finally:
                state["polling"] = False

        ui.timer(2.0, poll_face)


# ════════════════════════════════════════════════════════════════

ui.run(host="0.0.0.0", port=8081, dark=True, storage_secret="nutrition_ml")
