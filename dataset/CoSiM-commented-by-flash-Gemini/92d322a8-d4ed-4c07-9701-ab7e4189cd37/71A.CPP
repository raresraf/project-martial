/**
 * @file 71A.CPP
 * @brief Implements "Way Too Long Words" problem from Codeforces with simplified I/O.
 *
 * This program reads an integer 'n' representing the number of test cases.
 * For each test case, it reads a string 's'. If the length of 's' is strictly
 * greater than 10, the program prints an abbreviated version of the word:
 * the first character, followed by the number of characters between the first
 * and last (length - 2), followed by the last character. Otherwise, it prints
 * the word as is. This version uses standard C++ streams and string manipulation.
 *
 * Algorithm: String length check and conditional formatting.
 * Time Complexity: O(L) for each test case, where L is the length of the string,
 *                  dominated by string input/output operations. Total O(N * L_avg).
 * Space Complexity: O(L_max) for storing the longest string.
 */
#include <iostream> // For std::cin, std::cout.
#include <string>   // For std::string.

using namespace std; // Uses the standard namespace.

int main() {
  /**
   * @brief Main function: Processes multiple test cases, abbreviating long words.
   * @return 0 if the program executes successfully.
   */
  int n;     // Variable to store the number of test cases.
  string s;   // Variable to store the input string for each test case.
  cin >> n;  // Read the number of test cases.
  /**
   * @brief Loop to process each test case.
   * @pre 'n' is a non-negative integer.
   * @post For each test case, an abbreviated or original string is printed.
   * @invariant 'n' decrements in each iteration until all test cases are processed.
   */
  while (n--) {
    cin >> s;       // Read the input string for the current test case.
    /**
     * @brief Conditional block to check string length and print accordingly.
     * @pre 's.length()' holds the length of the string 's'.
     */
    if (s.length() > 10) { // If the word is too long (length greater than 10).
      // Functional Utility: Prints the abbreviated word format: first_char + (length-2) + last_char.
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << "\n";
    } else {           // If the word is not too long (length 10 or less).
      cout << s << "\n"; // Print the original word.
    }
  }
  return 0; // Return 0 for successful execution.
}