/**
 * @file 266A.CPP
 * @brief This program counts the number of adjacent identical characters in a given string.
 * It iterates through the string and increments a counter each time it finds a character
 * that is the same as its immediate successor.
 *
 * Algorithm: Linear scan for adjacent character comparison.
 * 1. Read an integer (which is ignored by the logic) and the input string.
 * 2. Initialize a counter for identical adjacent pairs to zero.
 * 3. Iterate from the first character up to the second-to-last character of the string.
 * 4. In each iteration, compare the current character with the next character.
 * 5. If they are identical, increment the counter.
 * 6. Print the final count.
 *
 * Time Complexity: O(L), where L is the length of the input string.
 * The algorithm performs a single pass through the string.
 *
 * Space Complexity: O(L) for storing the input string. Additional space is O(1) for variables.
 */

#include <iostream> // Required for input/output operations (cin, cout).
#include <string>   // Required for using the std::string class.

using namespace std;

/**
 * @brief Main function of the program.
 * Reads an integer and a string, then calculates and prints the count of
 * adjacent identical characters in the string.
 * @pre Standard input consists of an integer `n` followed by a string `s`.
 *      The integer `n` is provided but not used by the current logic.
 *      The string `s` can contain any characters.
 * @post A single integer representing the total count of adjacent identical
 *       character pairs is printed to standard output.
 * @return 0 upon successful program execution.
 */
int main() {
  string s; // Functional Utility: Declares a string variable to store the input string.
  int n;    // Functional Utility: Declares an integer variable to store the input integer (not used).
  // Functional Utility: Reads the integer `n` and the string `s` from standard input.
  cin >> n >> s;
  // Functional Utility: Stores the length of the input string for efficient loop iteration.
  int len = s.length();
  // Functional Utility: Initializes a counter to keep track of adjacent identical character pairs.
  int count = 0;

  /**
   * Block Logic: Iterates through the input string `s` to find adjacent identical characters.
   * The loop runs from the first character (`i = 0`) up to the second-to-last character (`len - 1`).
   * This is because each character `s[i]` is compared with `s[i + 1]`.
   * Invariant: At the start of each iteration `i`, `count` holds the number of
   *            adjacent identical pairs found in the substring `s[0...i-1]`.
   * Pre-condition: `s` is a valid string.
   */
  for (int i = 0; i < len - 1; i++) {
    /**
     * Block Logic: Compares the current character `s[i]` with the next character `s[i + 1]`.
     * If they are the same, it indicates an adjacent identical pair.
     * Pre-condition: `s[i]` and `s[i+1]` are valid characters within the string.
     */
    if (s[i] == s[i + 1]) {
      count++; // Functional Utility: Increments the counter for each identical adjacent pair found.
    }
  }
  // Functional Utility: Prints the final calculated count of adjacent identical character pairs.
  cout << count;
  return 0; // Functional Utility: Indicates successful program termination.
}