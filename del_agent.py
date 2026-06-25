#!/usr/bin/env python3
# list_agents.py - List reasoning engines and delete by resource ID

import sys
import vertexai
from vertexai.preview import reasoning_engines

# Initialize
PROJECT_ID = "ca-app-shared-prd-444"
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)

print("=" * 80)
print("📋 LISTING ALL REASONING ENGINES")
print("=" * 80)

try:
    # List all reasoning engines
    engines = reasoning_engines.ReasoningEngine.list(
        project=PROJECT_ID,
        location=LOCATION
    )
    
    if not engines:
        print("\n✅ No reasoning engines found.")
        exit(0)
    
    print(f"\n📊 Found {len(engines)} reasoning engine(s):\n")
    
    engine_dict = {}
    for i, engine in enumerate(engines, 1):
        # Extract just the ID number from the full resource name
        resource_id = engine.name.split('/')[-1]
        engine_dict[resource_id] = engine
        
        print(f"{i}. {engine.display_name}")
        print(f"   Full Resource: {engine.name}")
        print(f"   Resource ID: {resource_id}")
        print(f"   Created: {engine.create_time if hasattr(engine, 'create_time') else 'Unknown'}")
        print()

except Exception as e:
    print(f"❌ Error listing engines: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Ask if user wants to delete
print("=" * 80)
print("🗑️  DELETE ENGINE")
print("=" * 80)

resource_id_to_delete = input("\nEnter Resource ID to delete (or press Enter to skip): ").strip()

if not resource_id_to_delete:
    print("\n✅ No deletion. Exiting.")
    exit(0)

# Check if resource ID exists
if resource_id_to_delete not in engine_dict:
    print(f"\n❌ Resource ID '{resource_id_to_delete}' not found!")
    print(f"\nAvailable Resource IDs:")
    for rid in engine_dict.keys():
        print(f"  - {rid}")
    exit(1)

# Confirm deletion
engine_to_delete = engine_dict[resource_id_to_delete]
print(f"\n⚠️  Are you sure you want to delete:")
print(f"   Name: {engine_to_delete.display_name}")
print(f"   Resource ID: {resource_id_to_delete}")

confirm = input("\nType 'yes' to confirm deletion: ").strip().lower()

if confirm != "yes":
    print("\n✅ Deletion cancelled.")
    exit(0)

# Delete the engine
try:
    print(f"\n🗑️  Deleting {engine_to_delete.display_name}...")
    engine_to_delete.delete()
    print(f"✅ Successfully deleted!")
    
except Exception as e:
    print(f"❌ Error deleting engine: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("Done!")
print("=" * 80)
