# USER DEFINED FUNCTIONS FOR USER INTERFACE 

def display_img(image_file, preview_width=400):

    ext = os.path.splitext(image_file)[1].lower()

    if ext in [".jpg", ".jpeg"]:
        mime = "jpeg"
    elif ext == ".png":
        mime = "png"
    else:
        mime = "jpeg"   # safe fallback

    with open(image_file, "rb") as f:
        data = f.read()

    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/{mime};base64,{encoded}" 
                 style="max-width:90%; width:{preview_width}px; 
                        border:4px solid #ccc; 
                        box-shadow:5px 5px 15px rgba(0,0,0,0.3); 
                        border-radius:8px;">
        </div>
        """,
        unsafe_allow_html=True
    )
