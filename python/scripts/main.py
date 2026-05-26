import flet as ft
import os
import requests
from core_logic.vision_engine import get_recommendations # We only need the JSON logic here, not the ML

# The Tailscale IP of your MacBook M2
OPS_SERVER_URL = "http://100.124.57.75:8000/analyze_biometrics"
UPLOAD_PATH = os.path.join(os.path.dirname(__file__), "assets", "uploads")
os.makedirs(UPLOAD_PATH, exist_ok=True)

TRANSPARENT_PIXEL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

def main(page: ft.Page):
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = ft.Colors.BLACK
    page.window.width = 400
    page.window.height = 850
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.padding = 20

    header = ft.Text("CYBERPUNK HUD // THIN CLIENT", color=ft.Colors.GREEN_ACCENT, weight=ft.FontWeight.BOLD)
    status_text = ft.Text("STATUS: AWAITING INPUT", color=ft.Colors.WHITE70)

    # Simplified display since OpenCV is removed from the client
    feed_image = ft.Image(src=TRANSPARENT_PIXEL, width=350, height=350, fit=ft.BoxFit.COVER)
    display_container = ft.Container(
        content=feed_image,
        width=350, height=350, alignment=ft.Alignment.CENTER, border=ft.border.all(1, ft.Colors.WHITE24)
    )

    result_display = ft.Text("", size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN_400, text_align=ft.TextAlign.CENTER)

    hairline_toggle = ft.SegmentedButton(
        selected=["SOLID"],
        allow_empty_selection=False, allow_multiple_selection=False,
        selected_icon=ft.Icon(ft.Icons.RADAR),
        segments=[
            ft.Segment(value="SOLID", label=ft.Text("SOLID", weight=ft.FontWeight.BOLD)),
            ft.Segment(value="RECEDING", label=ft.Text("RECEDING", weight=ft.FontWeight.BOLD)),
        ],
    )

    gallery_row = ft.Row(wrap=False, scroll=ft.ScrollMode.ALWAYS, visible=False, height=160, spacing=15)
    modal_image = ft.Image(src=TRANSPARENT_PIXEL, fit=ft.BoxFit.CONTAIN)
    interactive_viewer = ft.InteractiveViewer(content=modal_image, max_scale=5, boundary_margin=ft.margin.all(20))
    
    def close_modal(e):
        blueprint_modal.open = False
        page.update()

    blueprint_modal = ft.AlertDialog(
        content=ft.Container(interactive_viewer, width=350, height=450),
        actions=[ft.TextButton("CLOSE TERMINAL", on_click=close_modal)],
        actions_alignment=ft.MainAxisAlignment.END,
        bgcolor="#111111"
    )
    page.overlay.append(blueprint_modal)

    def process_server_response(shape):
        operator_hairline = list(hairline_toggle.selected)[0]
        result_display.value = f"TARGET LOCKED: {shape}\nHAIRLINE: {operator_hairline}"
        
        recommendations = get_recommendations(shape.lower(), operator_hairline)
        gallery_row.controls.clear()
        
        if recommendations:
            for rec in recommendations:
                def create_open_modal_func(path_360):
                    def open_modal(e):
                        modal_image.src = path_360
                        blueprint_modal.open = True
                        page.update()
                    return open_modal

                thumb = ft.Container(
                    content=ft.Image(src=rec["front"], width=130, height=130, fit=ft.BoxFit.COVER, border_radius=10),
                    border=ft.border.all(2, ft.Colors.CYAN_900),
                    border_radius=12,
                    on_click=create_open_modal_func(rec["360"])
                )
                gallery_row.controls.append(thumb)
            gallery_row.visible = True

        status_text.value = "STATUS: RENDER COMPLETE"
        page.update()

    # ==========================================
    # UPLINK ENGINE: HTTP BRIDGE
    # ==========================================
    def on_upload_progress(e: ft.FilePickerUploadEvent):
        if e.progress == 1.0: 
            filepath = os.path.join(UPLOAD_PATH, e.file_name)
            feed_image.src = filepath # Show the local image instantly
            
            status_text.value = "STATUS: TRANSMITTING TO OPS SERVER..."
            page.update()
            
            try:
                # Fire the asset over the Tailscale tunnel
                with open(filepath, 'rb') as f:
                    response = requests.post(OPS_SERVER_URL, files={'file': f}, timeout=10)
                
                data = response.json()
                if data["status"] == "success":
                    process_server_response(data["shape"])
                else:
                    status_text.value = "STATUS: SERVER FAILED TO ACQUIRE LOCK"
                    page.update()
                    
            except Exception as ex:
                status_text.value = f"STATUS: UPLINK FAILED ({str(ex)})"
                page.update()

    file_picker = ft.FilePicker(on_upload=on_upload_progress)
    page.overlay.append(file_picker)
    page.update()
    
    async def trigger_scan(e):
        files = await file_picker.pick_files(allow_multiple=False, allowed_extensions=["png", "jpg", "jpeg"])
        if not files: return
        status_text.value = "STATUS: ACQUIRING OPTICAL DATA..."
        page.update()
        upload_url = page.get_upload_url(files[0].name, 60)
        await file_picker.upload([ft.FilePickerUploadFile(name=files[0].name, upload_url=upload_url)])

    scan_btn = ft.Button("UPLOAD TO OPS SERVER", icon=ft.Icons.CLOUD_UPLOAD, color=ft.Colors.BLACK, bgcolor=ft.Colors.CYAN_ACCENT, on_click=trigger_scan)
    
    page.add(header, status_text, display_container, result_display, hairline_toggle, scan_btn, gallery_row)

if __name__ == "__main__":
    ft.app(target=main,
           assets_dir=os.path.join(os.path.dirname(__file__), "assets"),
           upload_dir=os.path.join(os.path.dirname(__file__), "assets", "uploads"))
