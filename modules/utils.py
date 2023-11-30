import secrets

def generate_token():
    token = secrets.token_hex(7)
    print(f"Received custom request with token {token}")
    return token
