"""
This file is not yet fully implemented.
"""


import threading
from .session import Session


class SessionManager:
    __lock = threading.Lock()
    __session_map: dict[str,Session] = {}
    __target_key: str = None
    
    def __init__(self):
        pass
    
    # def __del__(self):
    #     if SessionManager.__lock is not None:
    #         with SessionManager.__lock:
    #             # Finalize the session map
    #             SessionManager.__target_key = None
    #             for key, session in SessionManager.__session_map.items():
    #                 del SessionManager.__session_map[key]
    #                 del session
    #             SessionManager.__session_map = None
            
    #         # Delete the mutex
    #         del SessionManager.__lock
    #         SessionManager.__lock = None

    @staticmethod
    def open_session(session_name: str, url: str) -> Session or None:
        """
        | Open session, need to input 'session name', 'url'
        | This function return Python class.

        >>> open_session( session_name, url )
        return Session
        """

        # Check the validation
        with SessionManager.__lock:
            result = SessionManager.__session_map.get(session_name, None)
            
        if result is not None:
            print(f"failed: Session name '{session_name}' is already in use.")
            return None
        
        # Open a session
        with SessionManager.__lock:
            new_session = Session(url)
            SessionManager.__session_map[session_name] = new_session
            SessionManager.__target_key = session_name
        
        return new_session
    
    @staticmethod
    def close_session(session_name: str) -> bool:
        """
        | Close session, need to input 'session name'.
        | If session name is not registered, return false.
        | If Session successfully delete, return true.
        | This function return Python bool object.

        >>> close_session( session_name )
        return bool
        """

        # Get the target session
        with SessionManager.__lock:
            session = SessionManager.__session_map.get(session_name, None)
        
        # Check the validation  
        if session is None:
            print(f"failed: Session '{session_name}' is not registered.")
            return False
        
        # Delete the session from map and close it
        with SessionManager.__lock:
            if SessionManager.__target_key == session_name:
                SessionManager.__target_key = None
            
            del SessionManager.__session_map[session_name]
            del session
        
        return True

    @staticmethod
    def append_session(session_name: str, session: Session) -> bool:
        # Check the validation
        with SessionManager.__lock:
            result = SessionManager.__session_map.get(session_name, None)
            
        if result is not None:
            print(f"failed: Session name '{session_name}' is already in use.")
            return False
        
        # append the session
        with SessionManager.__lock:
            SessionManager.__session_map[session_name] = session
            SessionManager.__target_key = session_name
            
        return True

    @staticmethod
    def get_session_map() -> dict[str,Session]:
        with SessionManager.__lock:
            result = SessionManager.__session_map
        return result
    
    @staticmethod
    def get_session() -> Session or None:
        with SessionManager.__lock:
            result = SessionManager.__session_map.get(SessionManager.__target_key, None)
        
        if result is None:
            print(f"failed: There are no open sessions.")
            
        return result

    @staticmethod
    def set_session(session_name: str) -> bool:
        # Get the target session
        with SessionManager.__lock:
            session = SessionManager.__session_map.get(session_name, None)
        
        # Check the validation  
        if session is None:
            print(f"failed: Session '{session_name}' is not registered.")
            return False
        
        # Change the target key
        with SessionManager.__lock:
            SessionManager.__target_key = session_name
        
        return True
