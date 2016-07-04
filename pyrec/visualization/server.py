from app import app
import sys
port = int(sys.argv[1])
app.run(host='0.0.0.0', port=port, debug=True)