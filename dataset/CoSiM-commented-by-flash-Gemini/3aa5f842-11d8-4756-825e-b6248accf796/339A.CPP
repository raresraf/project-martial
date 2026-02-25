/**
 * @file 339A.CPP
 * @brief This program reconstructs a given string of digits into a formatted sum string.
 * It processes the input string to group identical digits ('1', '2', '3') and
 * arranges them in ascending order of digit value, separated by '+' signs.
 * For example, an input like "3121" would be transformed into "1+1+2+3".
 * Algorithm: Iterative digit-grouping and concatenation.
 * Time Complexity: O(N), where N is the length of the input string, due to multiple passes.
 * Space Complexity: O(N) for storing the result string.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  string s, str;
  cin >> s;
  /**
   * Block Logic: Iterates through the input string to collect all occurrences of the digit '1'.
   * Each '1' found is appended to the result string, followed by a '+' separator.
   * Invariant: After this loop, 'str' contains all '1's from 's' in their original relative order,
   * each followed by a '+', unless it's the last character.
   */
  for (auto c : s) {
    if (c == '1') {
      str += c;
      str += '+';
    }
  }
  /**
   * Block Logic: Continues iterating through the input string to collect all occurrences of the digit '2'.
   * Each '2' found is appended to the result string, followed by a '+' separator.
   * Invariant: After this loop, 'str' contains all '1's followed by all '2's from 's'
   * in their original relative order within their groups, each followed by a '+'.
   */
  for (auto c : s) {
    if (c == '2') {
      str += c;
      str += '+';
    }
  }
  /**
   * Block Logic: Final iteration through the input string to collect all occurrences of the digit '3'.
   * Each '3' found is appended to the result string, followed by a '+' separator.
   * Invariant: After this loop, 'str' contains all '1's, then all '2's, then all '3's from 's'
   * in their original relative order within their groups, each followed by a '+'.
   */
  for (auto c : s) {
    if (c == '3') {
      str += c;
      str += '+';
    }
  }
  /**
   * Block Logic: Removes the trailing '+' character from the constructed sum string.
   * Pre-condition: 'str' is not empty and ends with a '+'.
   * Post-condition: 'str' contains the fully formatted sum string without a trailing '+'.
   */
  str.erase(str.length() - 1, 1);
  cout << str << endl;
  return 0;
}
