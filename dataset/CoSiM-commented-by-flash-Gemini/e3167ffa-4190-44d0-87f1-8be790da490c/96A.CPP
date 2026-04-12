/**
 * @file 96A.CPP
 * @brief Solution to the "Football" problem.
 *
 * This program analyzes a string representing player positions ('0' for one team, '1' for another)
 * to determine if a "dangerous" situation exists. A dangerous situation is defined as
 * 7 or more consecutive players of the same team.
 *
 * Algorithm: Substring search for "0000000" or "1111111".
 * Time Complexity: O(L) where L is the length of the input string, as `string::find`
 *                  effectively performs a single pass.
 * Space Complexity: O(1) beyond the input string, as only a few fixed-size strings are used.
 */
#include <iostream>
#include <string>
using namespace std;

/**
 * @brief Main function to determine if a football situation is dangerous.
 *
 * Functional Utility: Reads the player position string, checks for specific
 * dangerous patterns ("0000000" or "1111111"), and prints "YES" or "NO".
 */
int main() {
  std::string str; // Declare a string to store the input player positions.
  cin >> str;      // Read the input string.

  // Initialize two strings representing the "dangerous" patterns: seven '0's and seven '1's.
  std::string str0(7, '0'); // "0000000"
  std::string str1(7, '1'); // "1111111"

  /**
   * Block Logic: Checks if the input string is long enough to contain a dangerous situation.
   * If the string length is less than 7, a dangerous situation is impossible.
   * Pre-condition: `str` contains the input.
   */
  if (str.length() < 7) {
    std::cout << "NO"; // Not long enough for a dangerous situation.
  } else {
    // Search for the dangerous pattern "0000000" within the input string.
    std::size_t found0 = str.find(str0);
    // Search for the dangerous pattern "1111111" within the input string.
    std::size_t found1 = str.find(str1);

    /**
     * Block Logic: Determines if any dangerous pattern was found.
     * `std::string::npos` indicates that the substring was not found.
     * If either pattern is found, the situation is dangerous.
     * Post-condition: Prints "YES" if dangerous, "NO" otherwise.
     */
    if (found0 == std::string::npos && found1 == std::string::npos) {
      std::cout << "NO"; // Neither dangerous pattern found.
    } else {
      std::cout << "YES"; // At least one dangerous pattern found.
    }
  }

  return 0;
}