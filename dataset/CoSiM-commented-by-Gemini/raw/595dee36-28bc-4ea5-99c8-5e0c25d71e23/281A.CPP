/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem from Codeforces (281A).
 * @details The task is to capitalize the first letter of a given word. If the first
 * letter is already capitalized, no change is needed.
 *
 * @algorithm
 * The solution reads the input word into a C++ string. It then checks if the
 * first character of the string falls within the ASCII range of lowercase 'a'
 * to 'z'. If it does, the character is converted to its uppercase equivalent by
 * subtracting 32 from its ASCII value (the fixed offset between lowercase and
 * uppercase letters). Finally, the potentially modified string is printed.
 *
 * @complexity
 * Time Complexity: O(L) where L is the length of the string, dominated by the I/O operations.
 * The core logic of checking and modifying a single character is O(1).
 * Space Complexity: O(L) to store the input string.
 */
#include <cstring>
#include <iostream>

using namespace std;

int main() {
  /**
   * @variable word: Stores the input word to be capitalized.
   */
  string word;

  cin >> word;

  /**
   * @block
   * @brief Capitalizes the first letter of the word if it is lowercase.
   */
  // Pre-condition: Check if the first character is a lowercase letter.
  if (word[0] >= 'a' && word[0] <= 'z') {
    // Convert to uppercase using ASCII arithmetic.
    word[0] = word[0] - 32;
  }

  // Output the resulting word.
  cout << word << endl;

  return 0;
}
