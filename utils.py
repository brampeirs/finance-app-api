import re
from fastapi import HTTPException


def validate_date_format(date_str: str, field_name: str = "date") -> None:
    """
    Validate that a date string is in YYYY-MM format.
    
    Args:
        date_str: The date string to validate
        field_name: Name of the field for error messages (default: "date")
        
    Raises:
        HTTPException: If the date format is invalid
    """
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", date_str):
        raise HTTPException(
            status_code=422, 
            detail=f"{field_name} must be in 'YYYY-MM' format"
        )


def validate_date_range(start: str, end: str) -> None:
    """
    Validate a date range (start and end dates).
    
    Args:
        start: Start date in YYYY-MM format
        end: End date in YYYY-MM format
        
    Raises:
        HTTPException: If either date format is invalid or start >= end
    """
    # Validate both date formats
    validate_date_format(start, "start")
    validate_date_format(end, "end")
    
    # Validate start < end
    if start > end:
        raise HTTPException(
            status_code=422, 
            detail="start must be before end"
        )


def calculate_months_between(start: str, end: str) -> int:
    """
    Calculate the number of months between two YYYY-MM date strings.
    
    Args:
        start: Start date in YYYY-MM format
        end: End date in YYYY-MM format
        
    Returns:
        Number of months between the dates (inclusive of start, exclusive of end)
    """
    from datetime import datetime
    
    start_date = datetime.strptime(start, "%Y-%m")
    end_date = datetime.strptime(end, "%Y-%m")
    
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
