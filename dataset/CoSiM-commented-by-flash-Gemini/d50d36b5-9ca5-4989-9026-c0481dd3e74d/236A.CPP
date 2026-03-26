/**
 * @file 236A.CPP
 * @brief This program processes an input string to determine a specific output message
 * based on the parity (even or odd) of the count of its unique characters.
 * Algorithm: String sorting, unique character extraction, and parity check.
 * Time Complexity: O(L log L) due to sorting, where L is the length of the input string.
 * Space Complexity: O(L) for storing the string.
 */

#include <algorithm> // Required for std::sort and std::unique.
#include <iostream>  // Required for input/output operations (cin, cout).
#include <string>    // Required for std::string manipulation.

using namespace std;

/**
 * @brief Main function to analyze an input string for unique character count parity.
 * Reads a string, processes it to find unique characters, and prints
 * "CHAT WITH HER!" if the unique character count is even,
 * or "IGNORE HIM!" if it's odd.
 * @return 0 on successful execution.
 */
int main() {
  // @var s: String variable to store the input sequence of characters.
  string s;
  // Read the input string from standard input.
  cin >> s;
  // Sort the string alphabetically. This step is crucial for std::unique to work correctly,
  // as std::unique only operates on contiguous identical elements.
  sort(s.begin(), s.end());
  // Remove duplicate characters from the sorted string. std::unique rearranges the elements
  // such that all unique elements come first and returns an iterator to the end of the unique range.
  // s.erase then removes the elements from that iterator to the actual end of the string.
  s.erase(unique(s.begin(), s.end()), s.end());
  /**
   * @brief Conditional logic to determine the output message based on the parity of unique characters.
   * If the number of unique characters in the string `s` (after removing duplicates) is even,
   * it suggests a "chat" scenario; otherwise, it's an "ignore" scenario.
   */
  if (s.length() % 2 == 0) {
    // If the length of the string (number of unique characters) is even.
    cout << "CHAT WITH HER!" << endl;
  } else {
    // If the length of the string (number of unique characters) is odd.
    cout << "IGNORE HIM!" << endl;
  }
  return 0;
}