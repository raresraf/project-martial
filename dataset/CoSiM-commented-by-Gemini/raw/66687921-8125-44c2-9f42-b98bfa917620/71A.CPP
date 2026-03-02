#include <iostream>
#include <string>
using namespace std;

/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem from Codeforces.
 * @details This program implements a word abbreviation logic. If a word is longer
 * than 10 characters, it is shortened; otherwise, it is printed as is.
 *
 * Algorithm:
 * The program first reads an integer `n`, which is the number of words to process.
 * It then enters a loop that runs `n` times. In each iteration, it reads a word.
 * It checks if the length of the word is greater than 10.
 * If it is, the word is abbreviated to a format consisting of its first character,
 * followed by the number of characters between the first and last character (i.e., length - 2),
 * and finally its last character.
 * If the word's length is 10 or less, it is printed without modification.
 *
 * Time Complexity: O(n), where `n` is the number of words, as each word is processed once.
 * The string operations (length, character access) take time proportional to the word length,
 * but within typical constraints, this is effectively constant time per word.
 * Space Complexity: O(L), where `L` is the maximum length of a single word, to store the word in memory.
 */
int main() {
  int n;
  // Read the number of words to process.
  cin >> n;
  /**
   * @brief Main loop to process each of the `n` words.
   * Pre-condition: `n` contains the total count of words.
   * Invariant: The loop processes one word per iteration.
   */
  while (n--) {
    string s;
    // Read the next word.
    cin >> s;
    int x = s.length();
    /**
     * @brief Check if the word's length is greater than 10.
     * This is the condition for abbreviation.
     */
    if (x > 10) {
      // Print the abbreviated word: first char, length-2, last char.
      cout << s[0] << x - 2 << s[x - 1] << endl;
    } else
      // If the word is not too long, print it as is.
      cout << s << endl;
  }
}