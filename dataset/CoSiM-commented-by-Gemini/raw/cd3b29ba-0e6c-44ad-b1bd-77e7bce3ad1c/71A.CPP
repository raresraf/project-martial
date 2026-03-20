/**
 * @file 71A.CPP
 * @brief Solution for the "Way Too Long Words" problem (Codeforces 71A).
 *
 * @details
 * This program abbreviates words that are longer than 10 characters.
 * The abbreviation consists of the first letter, the number of letters
 * between the first and last, and the last letter. Words with 10 or fewer
 * characters are left unchanged.
 *
 * @section algorithm Algorithm
 * 1. Read the number of words, `n`.
 * 2. Store all `n` words in an array.
 * 3. Iterate through the array of words:
 *    a. If a word's length is greater than 10, construct and print the
 *       abbreviation.
 *    b. Otherwise, print the word as is.
 *
 * @section complexity Complexity
 * - Time Complexity: O(N*L) where N is the number of words and L is their max length,
 *   dominated by the time to read the input strings.
 * - Space Complexity: O(N*L) to store all words in memory.
 */
#include <iostream>
#include <string.h>
#include <string>

using namespace std;

int main() {
  int n;
  cin >> n;

  // Dynamically allocate an array to store the words.
  // Note: Using std::vector<string> is more idiomatic and safer in modern C++.
  string *words = new string[n];
  for (int i = 0; i < n; i++) {
    cin >> words[i];
  }

  /**
   * @brief Main processing loop.
   * Iterates through each word to decide whether to abbreviate it.
   */
  for (int i = 0; i < n; i++) {
    if (words[i].length() <= 10)
      cout << words[i] << endl;
    else {
      // Abbreviate the word if it's longer than 10 characters.
      string firstL(1, words[i].at(0));

      // Calculate the number of middle characters.
      // Note: The expression `(1, words[i].length() - 2)` uses the comma
      // operator, which evaluates to the second operand. So, it correctly
      // calculates `words[i].length() - 2`, though this syntax is unusual.
      string middleN = to_string((1, words[i].length() - 2));
      string lastL(1, words[i].at(words[i].length() - 1));

      // Print the constructed abbreviation.
      cout << firstL + middleN + lastL << endl;
    }
  }

  // Deallocate the memory used for the words array to prevent memory leaks.
  delete[] words;
  return 0;
}
