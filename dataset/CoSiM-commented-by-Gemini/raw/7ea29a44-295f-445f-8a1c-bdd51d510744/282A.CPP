/**
 * @file 282A.CPP
 * @brief Solution for Codeforces Problem 282A, "Bit++".
 * @details This program calculates the final value of a variable `x` after
 * performing a series of increment or decrement operations.
 *
 * Algorithm:
 * 1. Initialize a variable `x` to 0.
 * 2. Read an integer `n`, the number of statements to follow.
 * 3. Loop `n` times to process each statement.
 * 4. In each iteration, read the statement as a string.
 * 5. Check if the string contains the substring "++". If it does, increment `x`.
 * 6. Check if the string contains the substring "--". If it does, decrement `x`.
 * 7. After the loop completes, print the final value of `x`.
 *
 * Time Complexity: O(N), where N is the number of statements. Each string find
 * operation is on a very short string (length 3), so it can be considered
 * constant time.
 * Space Complexity: O(1), as only one statement string is stored at a time.
 */
#include <iostream>
#include <string>

int main() {
  int n;
  int x = 0;
  std::cin >> n;
  /**
   * @brief This loop processes each of the `n` statements.
   */
  for (int i = 0; i < n; i++) {
    std::string s;
    std::cin >> s;
    /**
     * @brief These conditions check whether the statement is an increment
     * or a decrement by searching for "++" or "--" substrings.
     */
    if (s.find("++") != std::string::npos) {
      x += 1;
    }
    if (s.find("--") != std::string::npos) {
      x -= 1;
    }
  }
  std::cout << x << std::endl;
}