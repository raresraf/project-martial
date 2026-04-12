/**
 * @file 58A.CPP
 * @brief Solution to the "Chat room" problem.
 *
 * This program checks if the word "hello" can be formed as a subsequence from a given input string.
 * This implies that the characters 'h', 'e', 'l', 'l', 'o' must appear in the input string
 * in that specific order, not necessarily consecutively.
 *
 * Algorithm: Subsequence Matching using `std::string::find`.
 * Time Complexity: O(L) where L is the length of the input string, as `string::find`
 *                  operations effectively traverse the string a constant number of times.
 * Space Complexity: O(1) beyond the input string.
 */
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Main function to initiate the "hello" subsequence check.
 *
 * Functional Utility: Reads the input string, performs a sequential search for
 * each character of "hello", and prints "YES" or "NO" based on the result.
 */
int main() {
  string s;   // Declare a string to store the input.
  cin >> s;   // Read the input string.

  // Functional Utility: Finds the first occurrence of each character of "hello" sequentially.
  // The 'h' is searched from the beginning. Subsequent characters are searched from the
  // position immediately after the previously found character.
  // `string::find` returns `string::npos` (or -1 in some contexts, but specifically -1 for int conversion) if not found.
  int h = s.find('h'),              // Find 'h'
      e = s.find('e', h + 1),        // Find 'e' after 'h'
      l1 = s.find('l', e + 1),       // Find first 'l' after 'e'
      l2 = s.find('l', l1 + 1),      // Find second 'l' after first 'l'
      o = s.find('o', l2 + 1);       // Find 'o' after second 'l'

  /**
   * Block Logic: Checks if all characters of "hello" were found in the correct sequential order.
   * `string::find` returns -1 if a character is not found.
   * The conditions `h < e`, `e < l1`, etc., ensure the correct order.
   * Also, checks `!= -1` for each to ensure all characters were actually found.
   * Post-condition: Prints "YES" if "hello" is a subsequence, "NO" otherwise.
   */
  if (h < e && h < l1 && h < l2 && h < o && e < l1 && e < l2 && e < o &&
      l1 < l2 && l1 < o && l2 < o && h != -1 && e != -1 && l1 != -1 &&
      l2 != -1 && o != -1)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
