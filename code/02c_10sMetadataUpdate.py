import json
import os
import glob
import re
import pandas as pd
import chardet
from datetime import datetime, timedelta

# Define paths
CONSOLIDATED_DIR = r"D:\Technical_projects\PSAI\code\consolidated_output"
CSV_FILE = r"D:\Technical_projects\PSAI\raw_data\PSC\000_LIMBFILES\10s.csv"

def detect_encoding(file_path):
    """Detect the encoding of a file"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(1024 * 1024)  # Read up to 1MB
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    
    # If confidence is low, use common fallbacks
    if not encoding or result['confidence'] < 0.7:
        for enc in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    f.read(1000)  # Try reading a bit
                return enc
            except UnicodeDecodeError:
                continue
        return 'latin1'  # Last resort fallback
    return encoding

def parse_date(date_str):
    """Parse date string to datetime object"""
    # Try different formats
    formats = [
        '%Y_%m_%d',  # YYYY_MM_DD
        '%Y-%m-%d',  # YYYY-MM-DD
        '%Y/%m/%d'   # YYYY/MM/DD
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None

def load_metadata_from_csv(csv_file):
    """Load metadata from the 10s.csv file"""
    print(f"Loading metadata from: {csv_file}")
    
    # Detect encoding
    encoding = detect_encoding(csv_file)
    print(f"Detected encoding: {encoding}")
    
    # Read CSV with proper encoding
    try:
        df = pd.read_csv(csv_file, encoding=encoding)
    except Exception as e:
        print(f"Error reading CSV with detected encoding: {e}")
        # Try common fallbacks
        for alt_encoding in ['latin1', 'cp1252', 'iso-8859-1', 'utf-8']:
            try:
                print(f"Trying encoding: {alt_encoding}")
                df = pd.read_csv(csv_file, encoding=alt_encoding)
                print(f"Successfully read with {alt_encoding}")
                break
            except:
                continue
        else:
            print("Failed to read CSV with any encoding")
            return {}
    
    # Print column info
    print(f"Columns in CSV: {df.columns.tolist()}")
    
    # Map to standardize column names
    column_mapping = {}
    standard_columns = {'date': 'date', 'title': 'title', 'subject': 'subject'}
    
    for std_col, aliases in standard_columns.items():
        if std_col not in df.columns:
            # Look for matching column names
            for col in df.columns:
                if std_col.lower() in col.lower():
                    column_mapping[col] = std_col
                    break
    
    # Rename columns if needed
    if column_mapping:
        print(f"Renaming columns: {column_mapping}")
        df = df.rename(columns=column_mapping)
    
    # Create metadata mapping
    metadata_map = {}
    date_objects = {}  # Store datetime objects for date flexibility
    
    for _, row in df.iterrows():
        try:
            date = row.get('date')
            title = row.get('title')
            subject = row.get('subject')
            
            # Skip if missing essential data
            if pd.isna(date) or pd.isna(title):
                continue
            
            # Clean up values
            date = str(date).strip()
            title = str(title).strip()
            
            # Parse date to datetime for flexible matching
            date_obj = parse_date(date)
            
            # If date couldn't be parsed, skip this entry
            if not date_obj:
                print(f"Warning: Could not parse date '{date}', skipping this entry")
                continue
            
            # Format subjects
            subject_list = []
            if subject and not pd.isna(subject):
                subject = str(subject)
                if ';' in subject:
                    subject_list = [s.strip() for s in subject.split(';')]
                elif ',' in subject:
                    subject_list = [s.strip() for s in subject.split(',')]
                else:
                    subject_list = [subject.strip()]
            
            # Create metadata entry
            metadata_entry = {
                'title': title,
                'subjects': subject_list
            }
            
            # Store the date object for flexible matching
            date_key = date.replace('-', '_')  # Standardize to YYYY_MM_DD
            date_objects[date_key] = date_obj
            
            # Store with various key formats
            # Original format
            metadata_map[date_key] = metadata_entry
            
            # With PSC_ prefix
            metadata_map[f"PSC_{date_key}"] = metadata_entry
            
            # Hyphen format
            hyphen_date = date_key.replace('_', '-')
            metadata_map[hyphen_date] = metadata_entry
            metadata_map[f"PSC_{hyphen_date}"] = metadata_entry
            
        except Exception as e:
            print(f"Error processing row: {e}")
    
    # Store the date objects dictionary alongside the metadata map
    return metadata_map, date_objects

def generate_adjacent_dates(date_obj):
    """Generate date strings for the day before and after"""
    dates = []
    
    # Original date
    dates.append(date_obj.strftime('%Y_%m_%d'))
    dates.append(date_obj.strftime('%Y-%m-%d'))
    
    # Day before
    prev_day = date_obj - timedelta(days=1)
    dates.append(prev_day.strftime('%Y_%m_%d'))
    dates.append(prev_day.strftime('%Y-%m-%d'))
    
    # Day after
    next_day = date_obj + timedelta(days=1)
    dates.append(next_day.strftime('%Y_%m_%d'))
    dates.append(next_day.strftime('%Y-%m-%d'))
    
    # Return with and without PSC_ prefix
    all_dates = []
    for d in dates:
        all_dates.append(d)
        all_dates.append(f"PSC_{d}")
    
    return all_dates

def update_json_file(json_file_path, metadata_map, date_objects):
    """Update a JSON file with metadata"""
    print(f"Processing: {os.path.basename(json_file_path)}")
    
    # Read JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Track updates
    updated_count = 0
    total_chunks = len(data.get("chunks", []))
    
    # First pass: collect unique source files
    source_files = {}
    for i, chunk in enumerate(data.get("chunks", [])):
        metadata = chunk.get("metadata", {})
        source_file = metadata.get("source_file", "")
        
        if source_file:
            if source_file not in source_files:
                source_files[source_file] = []
            source_files[source_file].append(i)
    
    print(f"Found {len(source_files)} unique source files in {total_chunks} chunks")
    
    # Second pass: match and update
    matched_files = 0
    unmatched_files = 0
    
    for source_file, chunk_indices in source_files.items():
        # Try multiple matching strategies
        metadata_entry = None
        
        # Extract base name without extension
        base_name = os.path.splitext(source_file)[0]
        
        # Strategy 1: Direct match
        if base_name in metadata_map:
            metadata_entry = metadata_map[base_name]
        
        # Strategy 2: Extract date pattern with flexible matching
        if not metadata_entry:
            # Try YYYY_MM_DD or YYYY-MM-DD pattern
            date_pattern = re.search(r'(\d{4})[\-_](\d{2})[\-_](\d{2})', source_file)
            if date_pattern:
                year, month, day = date_pattern.groups()
                
                # Create date object for this file
                try:
                    file_date = datetime(int(year), int(month), int(day))
                    
                    # Look for exact date match
                    date_str = file_date.strftime('%Y_%m_%d')
                    if date_str in metadata_map:
                        metadata_entry = metadata_map[date_str]
                    elif date_str in date_objects:
                        # Found the date in our objects, use it directly
                        metadata_entry = metadata_map[date_str]
                    else:
                        # Try flexible date matching (±1 day)
                        # Check each date in the metadata
                        for meta_date, date_obj in date_objects.items():
                            # Calculate difference in days
                            diff = abs((file_date - date_obj).days)
                            if diff <= 1:  # Within 1 day
                                metadata_entry = metadata_map[meta_date]
                                print(f"Flexible date match: {source_file} → {meta_date} (off by {diff} day(s))")
                                break
                                
                except ValueError:
                    pass  # Invalid date format
        
        # Strategy 3: Try all possible adjacent dates
        if not metadata_entry:
            date_pattern = re.search(r'(\d{4})[\-_](\d{2})[\-_](\d{2})', source_file)
            if date_pattern:
                year, month, day = date_pattern.groups()
                
                try:
                    file_date = datetime(int(year), int(month), int(day))
                    possible_dates = generate_adjacent_dates(file_date)
                    
                    for date_format in possible_dates:
                        if date_format in metadata_map:
                            metadata_entry = metadata_map[date_format]
                            print(f"Adjacent date match: {source_file} → {date_format}")
                            break
                except ValueError:
                    pass  # Invalid date
        
        # Strategy 4: Try partial matches
        if not metadata_entry:
            # Look for any keys that contain the year_month part
            year_month_pattern = re.search(r'(\d{4})[\-_](\d{2})', source_file)
            if year_month_pattern:
                year_month = year_month_pattern.group(0)
                for key in metadata_map.keys():
                    if year_month in key:
                        metadata_entry = metadata_map[key]
                        print(f"Partial date match: {source_file} → {key}")
                        break
        
        # Update chunks if metadata was found
        if metadata_entry:
            matched_files += 1
            for idx in chunk_indices:
                chunk = data["chunks"][idx]
                chunk_metadata = chunk.get("metadata", {})
                
                # Only update if fields are empty
                if not chunk_metadata.get("title") or chunk_metadata.get("title") == "":
                    chunk_metadata["title"] = metadata_entry["title"]
                
                if not chunk_metadata.get("subjects") or chunk_metadata.get("subjects") == []:
                    chunk_metadata["subjects"] = metadata_entry["subjects"]
                
                updated_count += 1
        else:
            unmatched_files += 1
            print(f"No metadata match found for: {source_file}")
    
    # Save updated JSON
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Updated {updated_count} chunks in {json_file_path}")
    print(f"Matched {matched_files}/{len(source_files)} source files")
    return updated_count

def process_years(start_year=2010, end_year=2014):
    """Process all years in the specified range"""
    # Load metadata
    metadata_map, date_objects = load_metadata_from_csv(CSV_FILE)
    
    total_updated = 0
    
    for year in range(start_year, end_year + 1):
        year_str = str(year)
        json_file = os.path.join(CONSOLIDATED_DIR, f"psc_{year_str}_all_chunks.json")
        
        if os.path.exists(json_file):
            print(f"\nProcessing year {year_str}...")
            updated = update_json_file(json_file, metadata_map, date_objects)
            total_updated += updated
        else:
            print(f"\nNo file found for year {year_str}")
    
    print(f"\nTotal chunks updated: {total_updated}")

# Run the code
if __name__ == "__main__":
    try:
        import chardet
    except ImportError:
        import subprocess
        import sys
        print("Installing chardet...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "chardet"])
        import chardet
    
    process_years(2010, 2014)