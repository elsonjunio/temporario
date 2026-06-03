from src.search.list_files import handle_list_files
from src.search.search_files import handle_search_files
from src.search.file_reader import handle_paginated_read


def main():
    # Test case for listing files in the current directory (non-recursive)
    print('--- Testing list_files (Non-Recursive) ---')
    try:
        result = handle_list_files(
            path='.', page=1, page_size=5, recursive=False
        )
        if 'error' not in result:
            print(
                f"Successfully retrieved file listing. Total items found: {result['total']}"
            )
            # Print a summary of the first few items for verification
            for item in result['items']:
                print(f"- {item['name']} ({item['type']})")
        else:
            print(f"Error during test: {result['error']}")

    except Exception as e:
        print(f'An unexpected error occurred during testing: {e}')

    # Test case for searching files in the current directory (recursive)
    print('\n--- Testing search_files (Recursive, *.py) ---')
    try:
        result = handle_search_files(
            pattern='*.py', path='.', page=1, page_size=5, recursive=True
        )
        if 'error' not in result:
            print(
                f"Successfully retrieved file search results. Total items found: {result['total']}"
            )
            # Print a summary of the first few items for verification
            for item in result['items']:
                print(f"- {item['name']} ({item['type']})")
        else:
            print(f"Error during test: {result['error']}")

    except Exception as e:
        print(f'An unexpected error occurred during testing: {e}')


if __name__ == '__main__':
    main()

# Example usage (optional, for testing)
if __name__ == "__main__":
    # Replace with a real file path for local testing
    test_file = "src/main.py" 
    result = handle_paginated_read(test_file, page=1, page_size=50, show_line_number=True)
    print("--- Test Result ---")
    if result['status'] == 'success':
        print(f"Total Words: {result['total_words']}")
        print(f"Total Pages: {result['total_pages']}")
        print("\nPage 1 Content (first 50 words):")
        print("----------------------------------------")
        print(result['current_page_content'][:1000]) # Print up to 1000 chars for preview
    else:
        print(f"Error: {result.get('error')}")
