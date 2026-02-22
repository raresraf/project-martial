/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A: Way Too Long Words.
 * @details This program processes a list of words and abbreviates those that exceed a certain length (10 characters).
 * For words longer than 10 characters, the abbreviation consists of the first letter, followed by the count
 * of characters between the first and last letters, and then the last letter. Words 10 characters or shorter
 * are printed as they are.
 * @algorithm String manipulation and conditional formatting.
 * Time Complexity: $O(L_{avg} \cdot N)$ where N is the number of words and $L_{avg}$ is the average length of words.
 * Space Complexity: $O(L_{max})$ to store the longest word, where $L_{max}$ is the maximum word length.
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
 * are printed as they are.
 * @algorithm String manipulation and conditional formatting.
 * Time Complexity: $O(L_{avg} \cdot N)$ where N is the number of words and $L_{avg}$ is the average length of words.
 * Space Complexity: $O(L_{max})$ to store the longest word, where $L_{max}$ is the maximum word length.
 */
#include <iostream> // Required for input/output operations (std::cin, std::cout).
#include <string>   // Required for string manipulation (std::string, substr, size).

using namespace std; // Using the standard namespace to avoid prefixing std::.

int main() {
  /**
   * @brief Main function: orchestrates the word abbreviation logic.
   */
  string palabra; // @var palabra: Stores the current word read from input.
  int t,          // @var t: The total number of words to process.
      tam;        // @var tam: Stores the size (length) of the current word.

  cin >> t; // Reads the total number of words.

  /**
   * @brief Processes each of the 't' words.
   * @block_logic Iterates `t` times, processing one word in each iteration.
   * @invariant `t` decreases by 1 in each iteration until it becomes 0.
   */
  while (t--) {
    cin >> palabra; // Reads the current word.
    tam = palabra.size(); // Calculates the length of the current word.

    /**
     * @brief Checks if the word needs abbreviation.
     * @block_logic If `palabra.size()` is greater than 10, abbreviates the word; otherwise, prints it as is.
     * @invariant Words are either abbreviated or printed in full based on their length.
     */
    if (palabra.size() > 10) {
      // Abbreviate the word: first character + (length - 2) + last character.
      cout << palabra.substr(0, 1)        // First character.
           << tam - 2                     // Number of characters between first and last.
           << palabra.substr(palabra.size() - 1, 1) // Last character.
           << endl;                       // Newline.
    } else {
      // Print the word as is if its length is 10 or less.
      cout << palabra << endl;            // The original word followed by a newline.
    }
  }
  return 0; // Indicates successful execution.
}
