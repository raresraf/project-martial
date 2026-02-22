/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A: Way Too Long Words.
 * @details This program processes a list of words and abbreviates those that exceed a certain length (10 characters).
 * For words longer than 10 characters, the abbreviation consists of the first letter, followed by the count
 * of characters between the first and last letters, and then the last letter. Words 10 characters or shorter
 * are printed as they are. This implementation includes input validation for the number of words.
 * @algorithm String manipulation and conditional formatting.
 * Time Complexity: $O(L_{avg} \cdot N)$ where N is the number of words and $L_{avg}$ is the average length of words.
 * Space Complexity: $O(L_{max})$ to store the longest word, where $L_{max}$ is the maximum word length.
 * @author Sagindyk Urazayev
 * @see codeforces.com/problemset/problem/71/A
 */
#include <iostream>
#include <string>

using namespace std;

/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A: Way Too Long Words.
 * @details This program processes a list of words and abbreviates those that exceed a certain length (10 characters).
 * For words longer than 10 characters, the abbreviation consists of the first letter, followed by the count
 * of characters between the first and last letters, and then the last letter. Words 10 characters or shorter
 * are printed as they are. This implementation includes input validation for the number of words.
 * @algorithm String manipulation and conditional formatting.
 * Time Complexity: $O(L_{avg} \cdot N)$ where N is the number of words and $L_{avg}$ is the average length of words.
 * Space Complexity: $O(L_{max})$ to store the longest word, where $L_{max}$ is the maximum word length.
 * @author Sagindyk Urazayev
 * @see codeforces.com/problemset/problem/71/A
 */
#include <iostream> // Required for input/output operations (std::cin, std::cout).
#include <string>   // Required for string manipulation (std::string, size, at, substr).

using namespace std; // Using the standard namespace to avoid prefixing std::.

int main() {
  /**
   * @brief Main function: orchestrates the word abbreviation logic.
   */
  int i, // @var i: Loop counter.
      n; // @var n: The total number of words to process.

  cin >> n; // Reads the total number of words.

  /**
   * @brief Validates the input value for `n`.
   * @block_logic Checks if `n` is within the valid range [1, 100].
   * @invariant If `n` is outside the range, prints an error message. The program continues,
   *            but invalid input handling is noted.
   */
  if ((n < 1) || (n > 100)) {
    cout << "Wrong value!" << endl; // Prints error message for invalid 'n'.
  }

  string word, // @var word: Stores the current word read from input.
         werd; // @var werd: Declared but not used in this specific implementation.

  /**
   * @brief Processes each of the 'n' words.
   * @block_logic Iterates `n` times, processing one word in each iteration.
   * @invariant For each iteration, `word` holds the current word, which is either abbreviated or printed in full.
   */
  for (i = 0; i < n; i++) {
    cin >> word; // Reads the current word.

    /**
     * @brief Checks if the word needs abbreviation.
     * @block_logic If `word.size()` is greater than 10, abbreviates the word; otherwise, prints it as is.
     * @invariant Words are either abbreviated or printed in full based on their length.
     */
    if (word.size() > 10) {
      // Abbreviate the word: first character + (length - 2) + last character.
      cout << word.at(0)                  // First character.
           << word.size() - 2             // Number of characters between first and last.
           << word.at(word.size() - 1)    // Last character.
           << endl;                       // Newline.
    } else {
      // Print the word as is if its length is 10 or less.
      cout << word << endl;               // The original word followed by a newline.
    }
  }
  return 0; // Indicates successful execution.
}
