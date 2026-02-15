/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem using a two-pass, storage-first approach.
 * @details The problem requires abbreviating words that are longer than 10 characters.
 * This implementation first reads all words into an array and then iterates through
 * the array to process and print each one.
 *
 * @section algorithm Algorithm
 * The algorithm uses a two-pass approach:
 * 1.  **Storage Phase:**
 *     a. Read the number of words, `n`.
 *     b. Declare an array of strings to hold all `n` words. Note that this code
 *        uses a Variable-Length Array (VLA), which is not standard C++.
 *     c. Loop `n` times to read each word from input and store it in the array.
 * 2.  **Processing Phase:**
 *     a. Loop `n` times again, iterating through the stored words.
 *     b. For each word, check its length.
 *     c. If the length is 10 or less, print the word as is.
 *     d. If the length is greater than 10, construct and print an abbreviation:
 *        the first character, followed by the number of middle characters (length - 2),
 *        followed by the last character.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N * L), where N is the number of words and L is the max length of a word, dominated by the I/O operations.
 * - Space Complexity: O(N * L) to store all words in the array before processing.
 *   (Note: A more space-optimal O(L) solution exists by processing words as they are read).
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  int n, i, l;
  string word;
  cin >> n;
  
  // This program uses a Variable-Length Array (VLA), which is a non-standard C++ extension.
  // A std::vector<string> would be the standard C++ way to handle a dynamic array.
  string wordArray[n];
  
  // First pass: Read all words from input and store them in the array.
  for (i = 0; i < n; i++)
    cin >> wordArray[i];

  // Second pass: Process each word from the array and print the result.
  for (i = 0; i < n; i++) {
    word = wordArray[i];
    l = word.length();
    
    // Check if the word is "too long" (length > 10).
    if (l < 11) { // Equivalent to l <= 10
      cout << word << endl;
    } else {
      // The original code reuses 'l' which can be confusing.
      // The logic is to print the first char, then length-2, then the last char.
      // word.at(l - 1) would access the last character.
      cout << word.at(0) << (l - 2) << word.at(l - 1) << endl;
    }
  }

  return 0;
}