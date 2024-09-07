from flask import Flask, render_template
from flask_bootstrap import Bootstrap
app = Flask(__name__)
Bootstrap(app)

class Benchmark:
    def __init__(self):
        self.data = {
            "CPU": None,
            "Memory": None,
            "Disk": None,
        }
    
    def run_benchmarks(self):
        import os, psutil
        
        # cpu information
        print('CPU count:', os.cpu_count())
        self.data["CPU"] = str(os.cpu_count())

        # memory information
        mem = psutil.virtual_memory()
        print("Total Memory", mem.total)
        self.data["Memory"] = str(mem.total)
        
        # disk information.
        disk = psutil.disk_usage('/')
        print("Disk usage: ", disk.percent)
        self.data["Disk"] = str(disk.percent)

@app.route("/")
def benchmark():
    bm = Benchmark()
    bm.run_benchmarks()
    return render_template('index.html', data=bm.data)

if __name__ == "__main__":
    app.run(debug=True)
Now, for creating a UI with Flask 