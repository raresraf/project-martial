/**
 * @file 58A.CPP
 * @brief Checks if the word "hello" can be formed as a subsequence from an input string.
 *
 * This program reads an input string and determines if the characters 'h', 'e', 'l', 'l', 'o'
 * appear in order within the string, allowing for zero or more characters to be skipped
 * between them. This is a classic subsequence matching problem.
 *
 * Algorithm: Two-pointer approach. One pointer iterates through the input string,
 *            and another pointer tracks the progress of matching the "hello" sequence.
 * Time Complexity: O(N), where N is the length of the input string, as it involves a single pass.
 * Space Complexity: O(1) for storing constant-size strings and counters.
 */
#include <iostream> // Required for standard input/output operations.
#include <string>   // Required for string manipulation.

int main() {
  /**
   * @brief Main function: Reads an input string and checks for "hello" as a subsequence.
   * @return 0 if the program executes successfully.
   */
  std::string a, b = "hello"; // 'a': input string, 'b': target subsequence "hello".
  int n = 0, m = 0;           // 'n': pointer for 'b' (hello), 'm': count of matched characters from 'b'.
  std::cin >> a;              // Read the input string.

  /**
   * @brief Loop to iterate through the input string 'a' to find the subsequence "hello".
   * @pre 'a' contains the input string; 'b' is the target "hello"; 'n' and 'm' are initialized to 0.
   * @post If "hello" is found, 'm' will be 5 and the loop will break.
   * @invariant 'i' iterates through 'a'; 'n' advances only when a character from 'b' is matched;
   *            'm' tracks the total matched characters.
   */
  for (int i = 0; i < a.length(); i++) {
    /**
     * @brief Checks if the current character in 'a' matches the next expected character in 'b'.
     */
    if (a[i] == b[n]) {
      n++; // Advance pointer for 'b'.
      m++; // Increment count of matched characters.
    }

    /**
     * @brief Optimization: If "hello" is fully matched, no need to continue.
     */
    if (m == 5) { // 'hello' has 5 characters.
      break;      // Exit the loop early.
    }
  }

  /**
   * @brief Final check and output based on whether "hello" was fully matched.
   * @pre 'm' holds the count of matched characters from "hello".
   */
  if (m == 5) {
    std::cout << "YES"; // Output "YES" if "hello" is found.
  } else {
    std::cout << "NO"; // Output "NO" otherwise.
  }

  return 0;
}