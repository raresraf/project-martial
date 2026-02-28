/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces (71A).
 * @details The problem requires shortening long words. If a word is longer than 10 characters,
 * it should be replaced by an abbreviation composed of its first letter, the number of
 * letters between the first and the last, and its last letter. Shorter words remain unchanged.
 *
 * @algorithm
 * The solution first reads an integer `n`, the number of words to process. It then
 * loops `n` times. In each iteration, it reads a word into a character array.
 * It calculates the length of the word. If the length is greater than 10, it prints
 * the special abbreviation. Otherwise (if the length is 10 or less), it prints the
 * word as is.
 *
 * @complexity
 * Time Complexity: O(N * L), where N is the number of words and L is the maximum
 * length of a word. This is dominated by the I/O operations of reading the words.
 * The logic for each word is O(L) due to `strlen`.
 * Space Complexity: O(L) to store the longest word.
 */
#include <iostream>
#include <string.h>
using namespace std;

int main() {
  /**
   * @variable n: The number of words to process.
   * @variable i: Stores the length of the current word.
   * @variable str: A character array to hold the current word.
   */
  int n, i;
  char str[1000];

  // Read the number of test cases.
  cin >> n;

  // The check `if (n >= 1)` is redundant as the loop condition handles n=0.
  if (n >= 1) {
    /**
     * @block
     * @brief This loop processes each of the `n` words.
     */
    while (n--) {
      cin >> str;
      // Calculate the length of the word.
      i = strlen(str);
      
      // Pre-condition: Check if the word's length is 10 or less.
      if ((i >= 1) && (i <= 10))
        // If so, print the word as is.
        cout << str << endl;
      // Pre-condition: The word's length is greater than 10.
      else if (i <= 100) // The upper bound check is also redundant given the problem constraints.
        // Print the abbreviation: first char, count of middle chars, last char.
        cout << str[0] << (i - 2) << str[i - 1] << endl;
    }
  }

  return 0;
}
