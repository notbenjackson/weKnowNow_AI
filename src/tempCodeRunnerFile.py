
def print_network_info():
    """Print network information on startup"""
    local_ip = get_local_ip()
    hostname = socket.gethostname()
    
    print("\n" + "="*50)
    print(f"FastAPI Server Running!")
    print(f"Local access: http://localhost:8000")
    print(f"Network access: http://{local_ip}:8000")
    print(f"Hostname: {hostname}")
    print("="*50 + "\n")