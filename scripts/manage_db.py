"""
Generate Fake Face Vectors for Testing

This script adds synthetic face embeddings to the database so you can test
the webcam recognition against fake/unknown faces.

Options:
1. Add random face embeddings (should NOT match you)
2. Add noisy copies of existing user (should match as same person)
3. Clear and reset database
"""

import json
import numpy as np
import os
from datetime import datetime
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILE = os.path.join(BASE_DIR, "data", "face_database.json")
EMBEDDING_DIM = 478 * 3  # 1434 dimensions


def load_database() -> dict:
    """Load existing database."""
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    return {"users": []}


def save_database(data: dict):
    """Save database."""
    with open(DB_FILE, 'w') as f:
        json.dump(data, f, indent=4)


def get_next_id(data: dict) -> int:
    """Get next available user ID."""
    if not data["users"]:
        return 1
    return max(u["id"] for u in data["users"]) + 1


def generate_random_embedding() -> List[float]:
    """Generate random unit-normalized embedding."""
    emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb.tolist()


def add_random_faces(n: int = 5):
    """Add N random fake faces to database."""
    print(f"\n📥 Adding {n} random fake faces...")
    
    data = load_database()
    
    fake_names = [
        "FakePerson_A", "FakePerson_B", "FakePerson_C", 
        "FakePerson_D", "FakePerson_E", "FakePerson_F",
        "FakePerson_G", "FakePerson_H", "FakePerson_I",
        "FakePerson_J", "Stranger_1", "Stranger_2",
        "Unknown_X", "Unknown_Y", "Unknown_Z"
    ]
    
    for i in range(n):
        user_id = get_next_id(data)
        name = fake_names[i % len(fake_names)] + f"_{user_id}"
        
        user = {
            "id": user_id,
            "name": name,
            "embeddings": {
                "front": generate_random_embedding(),
                "right": generate_random_embedding(),
                "left": generate_random_embedding()
            },
            "created_at": datetime.now().isoformat(),
            "is_fake": True  # Mark as fake for reference
        }
        
        data["users"].append(user)
        print(f"   ✓ Added: {name} (ID: {user_id})")
    
    save_database(data)
    print(f"\n✅ Database now has {len(data['users'])} users")


def add_celebrity_names(n: int = 10):
    """Add fake faces with celebrity names for realistic testing."""
    print(f"\n📥 Adding {n} fake celebrity faces...")
    
    data = load_database()
    
    celebrity_names = [
        "John_Smith", "Jane_Doe", "Bob_Johnson", "Alice_Williams",
        "Charlie_Brown", "Diana_Ross", "Edward_Norton", "Fiona_Apple",
        "George_Clooney", "Helen_Mirren", "Ivan_Drago", "Julia_Roberts",
        "Kevin_Hart", "Lisa_Simpson", "Michael_Scott", "Nancy_Drew",
        "Oscar_Isaac", "Patricia_Arquette", "Quentin_Tarantino", "Rachel_Green"
    ]
    
    for i in range(min(n, len(celebrity_names))):
        user_id = get_next_id(data)
        name = celebrity_names[i]
        
        user = {
            "id": user_id,
            "name": name,
            "embeddings": {
                "front": generate_random_embedding(),
                "right": generate_random_embedding(),
                "left": generate_random_embedding()
            },
            "created_at": datetime.now().isoformat(),
            "is_fake": True
        }
        
        data["users"].append(user)
        print(f"   ✓ Added: {name} (ID: {user_id})")
    
    save_database(data)
    print(f"\n✅ Database now has {len(data['users'])} users")


def list_users():
    """List all users in database."""
    data = load_database()
    
    print(f"\n📋 Database Contents ({len(data['users'])} users):")
    print("-" * 50)
    
    for u in data["users"]:
        is_fake = u.get("is_fake", False)
        fake_tag = " [FAKE]" if is_fake else " [REAL]"
        print(f"   ID {u['id']}: {u['name']}{fake_tag}")
    
    if not data["users"]:
        print("   (empty)")


def clear_fake_users():
    """Remove all fake users, keep real ones."""
    data = load_database()
    
    original_count = len(data["users"])
    data["users"] = [u for u in data["users"] if not u.get("is_fake", False)]
    removed_count = original_count - len(data["users"])
    
    save_database(data)
    print(f"\n🗑️ Removed {removed_count} fake users")
    print(f"   Remaining: {len(data['users'])} real users")


def clear_all_users():
    """Clear entire database."""
    save_database({"users": []})
    print("\n🗑️ Database cleared!")


def show_menu():
    """Show interactive menu."""
    print("\n" + "="*60)
    print("  MySIMOKA - Database Manager")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("  1. Add 5 random fake faces")
        print("  2. Add 10 celebrity-named fake faces")
        print("  3. Add 20 fake faces")
        print("  4. Add 50 fake faces")
        print("  5. List all users")
        print("  6. Remove only fake users (keep real)")
        print("  7. Clear entire database")
        print("  8. Quick setup (clear + add 10 fakes)")
        print("  0. Exit")
        
        choice = input("\nEnter choice: ").strip()
        
        if choice == "1":
            add_random_faces(5)
        elif choice == "2":
            add_celebrity_names(10)
        elif choice == "3":
            add_random_faces(20)
        elif choice == "4":
            add_random_faces(50)
        elif choice == "5":
            list_users()
        elif choice == "6":
            clear_fake_users()
        elif choice == "7":
            confirm = input("Are you sure? (y/n): ").strip().lower()
            if confirm == 'y':
                clear_all_users()
        elif choice == "8":
            add_celebrity_names(10)
            print("\n✅ Added 10 fake faces! Your existing users are preserved.")
            print("   Run demo_webcam.py to test recognition.")
        elif choice == "0":
            print("\n👋 Bye!")
            break
        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "add5":
            add_random_faces(5)
        elif cmd == "add10":
            add_celebrity_names(10)
        elif cmd == "add20":
            add_random_faces(20)
        elif cmd == "add50":
            add_random_faces(50)
        elif cmd == "list":
            list_users()
        elif cmd == "clear-fake":
            clear_fake_users()
        elif cmd == "clear-all":
            clear_all_users()
        elif cmd == "setup":
            add_celebrity_names(10)
        else:
            print(f"Unknown command: {cmd}")
            print("Available: add5, add10, add20, add50, list, clear-fake, clear-all, setup")
    else:
        show_menu()
