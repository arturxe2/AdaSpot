# Global imports

# Local imports
from util.io import load_text
from util.constants import F3SET_ELEMENTS

def load_classes(file_name):
    return {x: i + 1 for i, x in enumerate(load_text(file_name))}

def load_elements(file_name):
    elements = []
    elements_text = load_text(file_name)
    j = 0
    for category_length in F3SET_ELEMENTS:
        category_start = j
        category_end = j + category_length
        elements.append({elements_text[i]: i - category_start for i in range(category_start, category_end)})
        j += category_length
    
    return elements