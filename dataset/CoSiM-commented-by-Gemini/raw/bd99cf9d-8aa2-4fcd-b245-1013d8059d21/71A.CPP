/**
 * @file 71A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 71A "Way Too Long Words".
 *
 * @details
 * This program processes a list of words. For each word, if its length is
 * greater than 10 characters, it is replaced by a special abbreviation.
 * The abbreviation consists of the first character of the word, followed by
 * the number of characters between the first and last characters, and finally
 * the last character of the word. Words with 10 or fewer characters remain
 * unchanged.
 *
 * @algorithm
 * 1. Read the total number of words to process.
 * 2. Read all words from standard input and store them in a std::vector.
 * 3. Iterate through the vector of words.
 * 4. For each word, check if its length exceeds 10.
 * 5. If it does, construct and print the abbreviation.
 * 6. Otherwise, print the word as is.
 *
 * @note The implementation first stores all words in a vector before processing,
 * which is less memory-efficient than processing each word as it is read.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the total number of characters across all words,
 *   dominated by the input reading phase.
 * - Space Complexity: O(N), as all words are stored in a vector before being processed.
 */

#include <iostream>
#include <string>
#include <vector>
using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
int main() {
  int w{0}; // Variable to store the number of words.
  vector<string> v; // Vector to store all input words.
  
  cin >> w;
  
  /**
   * @brief Input phase.
   * Reads all `w` words from standard input and stores them in a vector.
   */
  for (int i{0}; i < w; i++) {
    string s;
    cin >> s;
    v.push_back(s);
  }
  
  /**
   * @brief Processing and output phase.
   * Iterates through each word stored in the vector.
   */
  for (auto s : v) {
    // Pre-condition: Check if the word is "too long".
    if (s.length() > 10) {
      /**
       * @brief Constructs and prints the abbreviation for a long word.
       * The format is: [first letter][number of middle letters][last letter].
       * The number of middle letters is calculated as `length - 2`.
       * Note: `s.substr(1, s.length() - 2).length()` is an unnecessarily
       * complex way to compute `s.length() - 2`.
       */
      cout << s.at(0) << s.substr(1, s.length() - 2).length()
           << s.at(s.length() - 1) << endl;
    } else {
      // If the word is not too long, it is printed without modification.
      cout << s << endl;
    }
  }
  return 0;
}
