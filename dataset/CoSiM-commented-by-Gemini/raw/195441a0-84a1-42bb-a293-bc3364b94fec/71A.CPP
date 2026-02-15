/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" competitive programming problem.
 * @details The problem requires abbreviating words that are longer than 10 characters.
 * Words with 10 or fewer characters are printed as is. Longer words are replaced by a
 * special abbreviation.
 *
 * @section algorithm Algorithm
 * The algorithm processes a given number of words one by one:
 * 1. Read the number of words, `n`.
 * 2. Loop `n` times to process each word.
 * 3. In each iteration, read a `word`.
 * 4. Check the length of the `word`.
 * 5. If the length is 10 or less, the word is printed unchanged.
 * 6. If the length is greater than 10, an abbreviation is constructed and printed.
 *    The abbreviation consists of:
 *    a. The first character of the word.
 *    b. The count of characters between the first and last character (i.e., length - 2),
 *       converted to a string.
 *    c. The last character of the word.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N * L), where N is the number of words and L is the maximum
 *   length of a word. This is primarily dominated by the I/O operations.
 * - Space Complexity: O(L) to store the longest word being processed.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  int n; // The number of words to process.
  cin >> n;
  for (int i = 0; i < n; i++) {
    string word;
    cin >> word;
    // Check if the word's length is greater than 10.
    if (word.size() <= 10) {
      // If the word is not "too long", print it as is.
      cout << word << endl;
    } else {
      // If the word is "too long", create and print its abbreviation.
      
      // Calculate the number of characters between the first and last letters.
      int middle_char_count = word.size() - 2;
      
      // Convert the count to its string representation.
      string count_str = to_string(middle_char_count);
      
      // Construct the abbreviation: first char + count_str + last char.
      string abbreviation = word[0] + count_str + word[word.size() - 1];
      cout << abbreviation << endl;
    }
  }
  return 0;
}
