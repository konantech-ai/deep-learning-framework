import threading


class GlobalVariables:
    """
    This class manage GlobalVariables.
    """

    __session_ptr = None
    __lock = threading.Lock()

    @staticmethod
    def get_session_ptr():
        """
        Get session pointer
        """

        with GlobalVariables.__lock:
            return GlobalVariables.__session_ptr
    
    @staticmethod
    def set_session_ptr(session_ptr):
        """
        Set session pointer

        Parameters
        ----------
        `session_ptr`: ctypes.c_void_p
            session pointer( C++ engine ).
        """

        with GlobalVariables.__lock:
            GlobalVariables.__session_ptr = session_ptr

