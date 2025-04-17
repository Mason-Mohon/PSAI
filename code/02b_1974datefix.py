import json
import re
import os

def fix_dates_in_json(json_file_path):
    """
    Fix incomplete dates in a consolidated JSON file by extracting date from source_file.
    
    Args:
        json_file_path (str): Path to the consolidated JSON file to fix
    """
    print(f"Processing file: {json_file_path}")
    
    # Load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Count of updated chunks
    updated_count = 0
    
    # Get the year from the data
    year = data.get("year", "1974")
    
    # Process each chunk
    for chunk in data.get("chunks", []):
        metadata = chunk.get("metadata", {})
        
        # Check if date is incomplete
        if metadata.get("date") == f", {year}" or not metadata.get("date"):
            source_file = metadata.get("source_file", "")
            
            # Try to extract date from source_file
            date_match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', source_file)
            
            if date_match:
                # Get month and day
                _, month, day = date_match.groups()
                
                # Convert month number to month name
                month_names = [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ]
                
                try:
                    month_num = int(month)
                    day_num = int(day)
                    if 1 <= month_num <= 12:
                        month_name = month_names[month_num - 1]
                        # Update the date
                        metadata["date"] = f"{month_name} {day_num}, {year}"
                        updated_count += 1
                except ValueError:
                    # If month or day couldn't be converted to int
                    print(f"  Could not parse date parts in: {source_file}")
            
            # If no match found using the format YYYY-MM-DD, try PSR YYYY-MM
            elif not date_match:
                psr_match = re.search(r'PSR (\d{4})-(\d{1,2})', source_file)
                if psr_match:
                    _, month = psr_match.groups()
                    try:
                        month_num = int(month)
                        if 1 <= month_num <= 12:
                            month_name = month_names[month_num - 1]
                            # In this case, we don't have a day, so we'll use 1 as default
                            metadata["date"] = f"{month_name} 1, {year}"
                            updated_count += 1
                    except ValueError:
                        print(f"  Could not parse date parts in: {source_file}")
            
            # If still no match, try any occurrence of month number
            elif not date_match and not psr_match:
                any_month_match = re.search(r'-(\d{1,2})', source_file)
                if any_month_match:
                    month = any_month_match.group(1)
                    try:
                        month_num = int(month)
                        if 1 <= month_num <= 12:
                            month_name = month_names[month_num - 1]
                            # Default day to 1
                            metadata["date"] = f"{month_name} 1, {year}"
                            updated_count += 1
                    except ValueError:
                        print(f"  Could not parse month in: {source_file}")
            
            if metadata.get("date") == f", {year}" or not metadata.get("date"):
                # If we still couldn't fix it, set a default date
                metadata["date"] = f"January 1, {year}"
                print(f"  Using default date for: {source_file}")
                updated_count += 1
    
    # Save the updated JSON back to the same file
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Updated {updated_count} dates in {json_file_path}")
    print(f"File has been saved.")

# Main execution
if __name__ == "__main__":
    # File path
    json_file_path = r"D:\Technical_projects\PSAI\code\consolidated_output\psc_1974_all_chunks.json"
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"Error: File not found: {json_file_path}")
    else:
        fix_dates_in_json(json_file_path)