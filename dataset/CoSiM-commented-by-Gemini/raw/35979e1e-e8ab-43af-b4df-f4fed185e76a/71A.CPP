/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem using C-style strings.
 *
 * This program reads a number of words and abbreviates any word that is longer
 * than 10 characters. The abbreviation consists of the first letter, the count
 * of the middle letters, and the last letter. This implementation uses a
 * character array (C-style string) instead of std::string.
 *
 * Algorithm:
 * 1. Read the number of test cases, n.
 * 2. Loop 'n' times:
 *    a. Read the word into a character array.
 *    b. Calculate its length using `strlen`.
 *    c. If the length is greater than 10, print the abbreviation.
 *    d. Otherwise, print the original word.
 *
 * Time Complexity: O(T * L), where T is the number of test cases and L is the
 * maximum length of a word. `strlen` takes O(L) time for each word.
 * Space Complexity: O(L_max), where L_max is the size of the character array (101).
 */
#include <cstring> // For strlen
#include <iostream>

using namespace std;

int main() {
  int n; // Number of test cases.
  char c[101]; // Buffer to store words as C-style strings.
  
  cin >> n; // Read the number of words to process.
  
  /**
   * Block Logic: Main processing loop.
   * Iterates 'n' times, processing one word in each iteration.
   */
  while (n) {
    cin >> c; // Read the word.
    
    int len = strlen(c); // Calculate the length of the C-style string.
    
    /**
     * Block Logic: Abbreviation conditional.
     * Checks if the word's length is greater than the threshold of 10.
     */
    if (len > 10) {
      // If the word is too long, print the abbreviation.
      // c[0]: first character.
      // len - 2: count of characters between the first and last.
      // c[len - 1]: last character.
      cout << c[0] << len - 2 << c[len - 1] << endl;
    } else {
      // If the word is not too long, print it as is.
      cout << c << endl;
    }
    n--; // Decrement the test case counter.
  }
  return 0;
}
