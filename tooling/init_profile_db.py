"""Create the dedicated profile table in the explicitly configured MySQL DB."""
from dotenv import load_dotenv
from memory.profile_store import MySQLProfileStore


def main():
    load_dotenv()
    store = MySQLProfileStore()
    try:
        store.create_schema()
        store.check_ready()
        print("echomind_user_profiles schema ready")
    finally:
        store.close()


if __name__ == "__main__":
    main()
