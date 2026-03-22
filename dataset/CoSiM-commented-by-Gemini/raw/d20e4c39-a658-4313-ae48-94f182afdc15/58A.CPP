/**
 * @file 58A.CPP
 * @brief Checks if the string "hello" is a subsequence of an input string.
 *
 * @details The program reads a string and determines if the characters 'h', 'e', 'l', 'l', 'o'
 * appear in the input in the correct order, but not necessarily consecutively.
 *
 * Algorithm: Single-pass linear scan using a state-machine-like approach.
 * The code maintains a position index and scans forward, looking for each character
 * of "hello" in sequence.
 * Time Complexity: O(N), where N is the length of the input string.
 * Space Complexity: O(N) to store the input string.
 */
#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main() {
  string input;
  cin >> input;

  size_t pos = 0; // Represents the current position in the input string.

  // Block Logic: Scan forward to find the 'h'.
  while (pos < input.length() && input[pos] != 'h') {
    ++pos;
  }
  ++pos; // Move past the found character to start the next search.

  // Block Logic: From the new position, scan forward to find the 'e'.
  while (pos < input.length() && input[pos] != 'e') {
    ++pos;
  }
  ++pos;

  // Block Logic: Scan forward to find the first 'l'.
  while (pos < input.length() && input[pos] != 'l') {
    ++pos;
  }
  ++pos;

  // Block Logic: Scan forward to find the second 'l'.
  while (pos < input.length() && input[pos] != 'l') {
    ++pos;
  }
  ++pos;

  // Block Logic: Scan forward to find the 'o'.
  while (pos < input.length() && input[pos] != 'o') {
    ++pos;
  }

  /**
   * Block Logic: Determine if the entire subsequence was found.
   * If the position `pos` is still within the string's bounds, it means the
   * final 'o' was successfully found in sequence.
   */
  if (pos < input.length()) {
    cout << "YES";
  } else {
    // If `pos` is out of bounds, one of the characters was not found.
    cout << "NO";
  }
  return 0;
}
