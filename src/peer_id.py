class PeerID: 
    def __init__(self, host: str, port: int, name: str | None = None):
        self.host = host
        self.port = port
        self.name = name

    @property
    def address(self):
        return (self.host, self.port)

    def __eq__(self, other):
        if not isinstance(other, PeerID):
            return NotImplemented
        return self.address == other.address
    
    def __hash__(self):
        return hash(self.address)
    
    def __str__(self):
        return f"{self.host}:{self.port}"
