import requests
import json
from pathlib import Path
from typing import Optional
from datetime import datetime


SYNC_CONFIG = {
    "enabled": True,
    # TODO: change before uploading
    "server_url": "http://100.73.166.22:8080", # This is the Rust server so 8080 used here
    #"http://100.101.102.103:0000", # this is a placeholder to protect my privacy I have the actual server and it works
    "auto_sync_on_save": True, # sync when autosave triggers
    "manual_sync_only": False, # if true, only sync with /sync
}

class ConversationSync:
    """
    Handles syncing conversations to remote server.
    """

    def __init__(self, server_url: str, enabled: bool = True) -> None:
        self.server_url = server_url.rstrip('/')
        self.enabled = enabled
        # self.use_dedicated = use_dedicated

    def sync_file(self, filepath: str) -> bool:
        """"""
        if not self.enabled:
            return False
        
        try:
            file_path = Path(filepath)
            if not file_path.exists():
                print(f"File not found: {filepath}")
                return False

            with open(file_path, "rb") as f:
                # print(f"DEBUG: Posting to: {self.server_url}/conversations/upload")
                files = {
                    "file": (file_path.name, f, "application/json")
                    }
                response = requests.post(
                    # "http://100.101.102.103:0000/conversations",
                    f"{self.server_url}/conversations/upload",
                    files = files,
                    timeout = 30
                )
                # print(f"[debug]\n[CoversationSync.sync_file]\n{__file__} \nfilepath : {file_path}\nresponse status code{response.status_code}\n")
            
            return response.status_code == 200

        except:
            return False
    
    def sync_all(self, conversations_dir: str) -> dict:
        """
        """
        if not self.enabled:
            return {
                "Success":0,
                "Failed":0,
                "Total":0,
                }
        
        conversations_path = Path(conversations_dir)
        if not conversations_path.exists():
            return {
                "Success":0,
                "Failed":0,
                "Total":0,
                }
        
        json_files = list(conversations_path.glob("*.json"))
        success = 0
        failed = 0

        for file in json_files:
            if self.sync_file(str(file)):
                success += 1
            else:
                failed += 1
        return {
            "success": success,
            "failed": failed,
            "total": len(json_files),
        }

    def list_remote_conversations(self) -> list[dict]:
        """
        """
        if not self.enabled:
            return []
        
        try:
            response = requests.get(f"{self.server_url}/conversations", timeout = 10)
            if response.status_code == 200:
                return response.json()
            return []
        except:
            return []
    
    def download_conversation(self, filename: str, save_path: str) -> bool:
        """
        """
        if not self.enabled:
            return False
        
        try: 
            url = f"{self.server_url}/conversations/download/{filename}"
            response = requests.get(url, timeout = 30)
            
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return True
            return False
        except:
            return False
        
    def test_connection(self) -> bool:
        """
        """
        try:
            response = requests.get(
                f"{self.server_url}/conversations",
                timeout = 5
            )
            return response.status_code == 200
        except:
            return False



syncer = ConversationSync(
    server_url = SYNC_CONFIG["server_url"],
    enabled = SYNC_CONFIG["enabled"],
    # use_dedicated = SYNC_CONFIG.get("use_conversations_endpoint", True)
)

def sync_on_save(filepath: str) -> bool:
    """
    """
    if not SYNC_CONFIG["auto_sync_on_save"] or SYNC_CONFIG["manual_sync_only"]:
        return False
    
    return syncer.sync_file(filepath)

def manual_sync_all() -> dict:
    """
    """
    conversations_dir = Path(__file__).parent / "conversations"
    print(conversations_dir)
    return syncer.sync_all(str(conversations_dir))

def list_server_conversations() -> list[dict]:
    """
    """
    return syncer.list_remote_conversations()
 
def download_from_server(filename: str, local_dir: str) -> bool:
    """
    """
    save_path = Path(local_dir) / filename
    
    return syncer.download_conversation(filename, str(save_path))

def check_server_status() -> bool:
    """
    """
    return syncer.test_connection()