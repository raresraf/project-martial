/**
 * @file 71A.CPP
 * @brief Solution for the "Way Too Long Words" problem.
 * @details This program shortens words that are longer than 10 characters by
 * replacing the middle characters with a count of their number. This implementation
 * first reads all words into an array and then processes them.
 *
 * @section algorithm Algorithm
 * 1. Read an integer `n`, the total number of words to process.
 * 2. Declare a string array `wordArray` of size `n`. Note: this is a
 *    Variable Length Array (VLA), which is a non-standard C++ extension.
 * 3. A loop runs `n` times to read all the words from the input and store them
 *    in `wordArray`.
 * 4. A second loop runs `n` times to process each word stored in the array.
 *    a. For each word, its length `l` is determined.
 *    b. If the length is 10 or less, the word is printed as is.
 *    c. If the length is greater than 10, an abbreviation is constructed and printed.
 *       The abbreviation consists of the first character, the number of middle
 *       characters (`length - 2`), and the last character.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N * L), where N is the number of words and L is the
 *   maximum length of a word. This is dominated by the I/O operations.
 * - Space Complexity: O(N * L) to store all the words in the `wordArray` before
 *   processing. This is less memory-efficient than an O(L) approach that
 *   processes each word as it is read.
 */
#include <iostream>
#include <string>
#include <vector>

using namespace std;

/**
 * @brief Main function to process and abbreviate long words.
 * @return 0 on successful execution.
 */
int main() {
  int n;
  cin >> n;
  
  // Using a std::vector is a standard C++ way to handle a dynamic array
  // instead of a non-standard VLA.
  vector<string> wordArray(n);
  for (int i = 0; i < n; i++) {
    cin >> wordArray[i];
  }

  // Process each word that was stored in the array.
  for (int i = 0; i < n; i++) {
    string word = wordArray[i];
    int l = word.length();
    
    // If the word is not "too long", print it directly.
    if (l <= 10) {
      cout << word << endl;
    } else {
      // Otherwise, construct and print the abbreviation.
      cout << word[0] << (l - 2) << word.back() << endl;
    }
  }

  return 0;
}