from app import create_dash_app
app = create_dash_app()

# app = app.server

app.run_server(debug=True)