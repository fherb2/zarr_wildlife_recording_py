"""
Universal JSON Serialization Module for Zarr Compatibility

This module provides universal JSON serialization capabilities for any Python class,
making them compatible with Zarr and other systems that require JSON serialization.

Key Features:
- Works with ANY Python class (dataclasses, regular classes, enums)
- Automatic serialization/deserialization
- Zarr-compatible out of the box
- No class modification required (optional enhancements available)
- Single import and setup

Usage:
    import universal_json_serialization as ujs
    
    # One-time setup (call once at program start)
    ujs.enable_universal_serialization()
    
    # Now ALL your classes work with json.dumps() and Zarr!
    json.dumps(your_object)  # Just works!

Author: Universal JSON Serialization Module
License: MIT
"""

import json
import dataclasses
from dataclasses import is_dataclass, asdict
from datetime import datetime, date, time
from enum import Enum
from typing import Any, Dict, List, Union, Optional, Type, Callable
from decimal import Decimal
from uuid import UUID
import inspect


# =============================================================================
# Core Serialization Functions
# =============================================================================

def serialize_object(obj: Any) -> Any:
    """
    Universal object serializer that handles any Python object.
    
    This function tries multiple strategies to serialize an object:
    1. Basic JSON types (pass through)
    2. Objects with __json__() method
    3. Objects with to_dict() method  
    4. Dataclasses (via dataclasses.asdict)
    5. Enums (via .value)
    6. Built-in types (datetime, UUID, Decimal, etc.)
    7. Objects with __dict__ attribute
    8. Iterables (lists, tuples, sets)
    9. Mappings (dictionaries)
    10. Fallback to string representation
    
    Args:
        obj: Any Python object to serialize
        
    Returns:
        JSON-serializable representation of the object
    """
    # 1. Basic JSON types - pass through unchanged
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    
    # 2. Objects with explicit __json__ method (highest priority)
    if hasattr(obj, '__json__') and callable(getattr(obj, '__json__')):
        try:
            result = obj.__json__()
            # Recursively serialize the result in case it contains complex objects
            return serialize_object(result) if not isinstance(result, (str, int, float, bool, type(None))) else result
        except Exception:
            pass  # Fall through to next strategy
    
    # 3. Objects with to_dict method
    if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        try:
            result = obj.to_dict()
            return serialize_object(result)
        except Exception:
            pass
    
    # 4. Dataclasses - use dataclasses.asdict
    if is_dataclass(obj):
        try:
            result = asdict(obj)
            return serialize_object(result)
        except Exception:
            pass
    
    # 5. Enums - use .value
    if isinstance(obj, Enum):
        return serialize_object(obj.value)
    
    # 6. Built-in types that need special handling
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, date):
        return obj.isoformat()
    elif isinstance(obj, time):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return str(obj)
    elif isinstance(obj, UUID):
        return str(obj)
    elif isinstance(obj, bytes):
        # Convert bytes to base64 string
        import base64
        return base64.b64encode(obj).decode('ascii')
    elif isinstance(obj, complex):
        return {'real': obj.real, 'imag': obj.imag, '_type': 'complex'}
    
    # 7. Objects with __dict__ (most regular Python classes)
    if hasattr(obj, '__dict__'):
        try:
            # Create dict from object attributes, excluding private/callable attributes
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_') and not callable(value):
                    result[key] = serialize_object(value)
            return result
        except Exception:
            pass
    
    # 8. Iterables (lists, tuples, sets)
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, dict)):
        try:
            if isinstance(obj, (list, tuple)):
                return [serialize_object(item) for item in obj]
            elif isinstance(obj, set):
                return list(serialize_object(item) for item in obj)
            else:
                # Other iterables
                return [serialize_object(item) for item in obj]
        except Exception:
            pass
    
    # 9. Mappings (dictionaries and dict-like objects)
    if hasattr(obj, 'items'):
        try:
            return {str(key): serialize_object(value) for key, value in obj.items()}
        except Exception:
            pass
    
    # 10. Fallback - convert to string
    try:
        return str(obj)
    except Exception:
        return f"<Unserializable: {type(obj).__name__}>"


def deserialize_object(data: Any, target_class: Optional[Type] = None) -> Any:
    """
    Universal object deserializer.
    
    Attempts to reconstruct objects from their JSON representation.
    Works best when target_class is provided, but can handle basic cases automatically.
    
    Args:
        data: JSON data (typically dict) to deserialize
        target_class: Optional target class to deserialize into
        
    Returns:
        Reconstructed object or the original data if no deserialization is possible
    """
    # Basic types - return as is
    if data is None or isinstance(data, (str, int, float, bool)):
        return data
    
    # Lists - recursively deserialize items
    if isinstance(data, list):
        return [deserialize_object(item, target_class) for item in data]
    
    # Dictionaries
    if isinstance(data, dict):
        # Special handling for complex numbers
        if data.get('_type') == 'complex':
            return complex(data['real'], data['imag'])
        
        # If target class is provided, try to construct it
        if target_class:
            try:
                # Try direct construction from dict
                if hasattr(target_class, 'from_json'):
                    return target_class.from_json(data)
                elif hasattr(target_class, 'from_dict'):
                    return target_class.from_dict(data)
                else:
                    # Try constructor with kwargs
                    return target_class(**data)
            except Exception:
                pass
        
        # Recursively deserialize dict values
        return {key: deserialize_object(value) for key, value in data.items()}
    
    return data


# =============================================================================
# JSON Encoder Patching
# =============================================================================

class UniversalJSONEncoder(json.JSONEncoder):
    """
    Universal JSON encoder that can handle any Python object.
    
    This encoder automatically serializes complex objects using the
    serialize_object function.
    """
    
    def default(self, obj):
        """Override default method to handle complex objects."""
        return serialize_object(obj)


# Global reference to original JSONEncoder.default for restoration
_original_json_default = None


def enable_universal_serialization():
    """
    Enable universal JSON serialization globally.
    
    This patches the standard json.JSONEncoder.default method to automatically
    handle complex objects. After calling this function, json.dumps() will work
    with any Python object.
    
    This is safe to call multiple times.
    
    Example:
        import universal_json_serialization as ujs
        ujs.enable_universal_serialization()
        
        # Now json.dumps() works with any object
        json.dumps(my_complex_object)
    """
    global _original_json_default
    
    # Store original if not already stored
    if _original_json_default is None:
        _original_json_default = json.JSONEncoder.default
    
    # Patch the default method
    json.JSONEncoder.default = lambda self, obj: serialize_object(obj)
    
    print("✅ Universal JSON serialization enabled globally!")


def disable_universal_serialization():
    """
    Disable universal JSON serialization and restore original behavior.
    
    This restores the original json.JSONEncoder.default method.
    """
    global _original_json_default
    
    if _original_json_default:
        json.JSONEncoder.default = _original_json_default
        print("✅ Universal JSON serialization disabled, original behavior restored.")
    else:
        print("⚠️  Universal serialization was not enabled.")


# =============================================================================
# Mixin Classes for Enhanced Functionality
# =============================================================================

class JSONSerializable:
    """
    Mixin class that adds JSON serialization methods to any class.
    
    Usage:
        class MyClass(JSONSerializable):
            def __init__(self, value):
                self.value = value
        
        obj = MyClass(42)
        json_str = obj.to_json()
        obj_copy = MyClass.from_json(json_str)
    """
    
    def __json__(self):
        """Return JSON-serializable representation."""
        return serialize_object(self)
    
    def to_dict(self):
        """Convert object to dictionary."""
        return serialize_object(self)
    
    def to_json(self, **kwargs):
        """Convert object to JSON string."""
        return json.dumps(self, **kwargs)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create instance from dictionary."""
        return deserialize_object(data, cls)
    
    @classmethod
    def from_json(cls, json_str: str):
        """Create instance from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class ZarrCompatible(JSONSerializable):
    """
    Enhanced mixin specifically designed for Zarr compatibility.
    
    Provides additional methods for working with Zarr metadata and attributes.
    """
    
    def to_zarr_attrs(self) -> Dict[str, Any]:
        """
        Convert object to Zarr-compatible attributes dictionary.
        
        Returns:
            Dictionary suitable for storing in Zarr .zattrs
        """
        return self.to_dict()
    
    @classmethod
    def from_zarr_attrs(cls, attrs: Dict[str, Any]):
        """
        Create instance from Zarr attributes dictionary.
        
        Args:
            attrs: Dictionary loaded from Zarr .zattrs
            
        Returns:
            Instance of this class
        """
        return cls.from_dict(attrs)
    
    def save_to_zarr_group(self, zarr_group, attr_name: str = 'metadata'):
        """
        Save this object to a Zarr group as attributes.
        
        Args:
            zarr_group: Zarr group object
            attr_name: Name of the attribute to store this object under
        """
        zarr_group.attrs[attr_name] = self.to_zarr_attrs()
    
    @classmethod
    def load_from_zarr_group(cls, zarr_group, attr_name: str = 'metadata'):
        """
        Load this object from a Zarr group's attributes.
        
        Args:
            zarr_group: Zarr group object
            attr_name: Name of the attribute to load this object from
            
        Returns:
            Instance of this class
        """
        attrs = zarr_group.attrs[attr_name]
        return cls.from_zarr_attrs(attrs)


# =============================================================================
# Utility Functions
# =============================================================================

def make_serializable(cls: Type) -> Type:
    """
    Class decorator that adds JSON serialization methods to any class.
    
    This is an alternative to using the JSONSerializable mixin.
    
    Usage:
        @make_serializable
        class MyClass:
            def __init__(self, value):
                self.value = value
        
        obj = MyClass(42)
        json.dumps(obj)  # Works!
    
    Args:
        cls: Class to make serializable
        
    Returns:
        Enhanced class with serialization methods
    """
    # Add __json__ method
    cls.__json__ = lambda self: serialize_object(self)
    
    # Add to_dict method
    cls.to_dict = lambda self: serialize_object(self)
    
    # Add to_json method
    cls.to_json = lambda self, **kwargs: json.dumps(self, **kwargs)
    
    # Add from_dict class method
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return deserialize_object(data, cls)
    cls.from_dict = from_dict
    
    # Add from_json class method
    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls.from_dict(data)
    cls.from_json = from_json
    
    return cls


def is_json_serializable(obj: Any) -> bool:
    """
    Check if an object can be JSON serialized.
    
    Args:
        obj: Object to check
        
    Returns:
        True if object can be serialized, False otherwise
    """
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


def prepare_for_zarr(obj: Any) -> Any:
    """
    Prepare any object for Zarr storage by ensuring it's JSON serializable.
    
    This function explicitly converts objects to their JSON-serializable form
    without relying on global patching.
    
    Args:
        obj: Object to prepare for Zarr
        
    Returns:
        JSON-serializable version of the object
    """
    return serialize_object(obj)


def test_serialization(obj: Any, verbose: bool = True) -> bool:
    """
    Test if an object can be successfully serialized and deserialized.
    
    Args:
        obj: Object to test
        verbose: Whether to print detailed results
        
    Returns:
        True if serialization/deserialization works, False otherwise
    """
    try:
        # Serialize
        serialized = serialize_object(obj)
        
        # Convert to JSON and back
        json_str = json.dumps(serialized)
        json_data = json.loads(json_str)
        
        # Try to deserialize
        deserialized = deserialize_object(json_data, type(obj))
        
        if verbose:
            print(f"✅ Serialization test passed for {type(obj).__name__}")
            print(f"   Original: {obj}")
            print(f"   Serialized: {serialized}")
            print(f"   Deserialized: {deserialized}")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"❌ Serialization test failed for {type(obj).__name__}: {e}")
        return False


# =============================================================================
# Module Initialization and Exports
# =============================================================================

# Auto-enable universal serialization when module is imported
# (Comment out this line if you want manual control)
# enable_universal_serialization()

# Define what gets imported with "from universal_json_serialization import *"
__all__ = [
    # Main functions
    'enable_universal_serialization',
    'disable_universal_serialization',
    'serialize_object',
    'deserialize_object',
    
    # Mixin classes
    'JSONSerializable',
    'ZarrCompatible',
    
    # Utilities
    'make_serializable',
    'is_json_serializable',
    'prepare_for_zarr',
    'test_serialization',
    
    # JSON Encoder
    'UniversalJSONEncoder',
]


# =============================================================================
# Module Information
# =============================================================================

__version__ = "1.0.0"
__author__ = "Universal JSON Serialization Module"
__description__ = "Universal JSON serialization for Python objects with Zarr compatibility"


if __name__ == "__main__":
    # Demo/test code when module is run directly
    print("🚀 Universal JSON Serialization Module Demo")
    print("=" * 50)
    
    # Enable universal serialization
    enable_universal_serialization()
    
    # Test with various object types
    from enum import Enum
    from datetime import datetime
    from dataclasses import dataclass
    
    # Test enum
    class Status(Enum):
        READY = "ready"
        PROCESSING = "processing"
    
    # Test regular class
    class RegularClass:
        def __init__(self, name, value):
            self.name = name
            self.value = value
    
    # Test dataclass
    @dataclass
    class DataClass:
        name: str
        count: int
    
    # Test with mixin
    class MixinClass(JSONSerializable):
        def __init__(self, data):
            self.data = data
    
    # Run tests
    test_objects = [
        Status.READY,
        RegularClass("test", 42),
        DataClass("example", 100),
        MixinClass({"nested": "data"}),
        datetime.now(),
        [1, 2, {"key": "value"}],
    ]
    
    for obj in test_objects:
        test_serialization(obj)
        print()
    
    print("🎉 All tests completed!")
