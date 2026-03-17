/**
 * @file 58A.CPP
 * @brief Solution to the "Chat room" problem.
 *
 * @details This program determines if the word "hello" can be formed by deleting zero or more characters
 * from a given input string. In other words, it checks if "hello" is a subsequence of the input string.
 *
 * @section
 * Algorithm:
 * The program uses a greedy approach. It scans the input string from left to right, looking for the
 * characters 'h', 'e', 'l', 'l', 'o' in that specific order. It keeps a pointer to the current
 * position in the string and for each character of "hello", it finds its first occurrence after
 * the previously found character.
 *
 * Time Complexity: O(L), where L is the length of the input string, because it performs a single linear scan.
 * Space Complexity: O(L), to store the input string.
 */
#include <iostream>
#include <string>
using namespace std;
int main() {
  string s;
  cin >> s;
  size_t pos = 0;

  /**
   * @brief This sequence of loops implements the greedy search for the subsequence "hello".
   * Logic: For each character in "hello", it scans the input string 's' starting from the
   * position 'pos' where the previous character was found. If a character is found, 'pos'
   * is advanced past it for the next search.
   */
  while (pos < s.length() && s[pos] != 'h')
    ++pos;
  ++pos;
  while (pos < s.length() && s[pos] != 'e')
    ++pos;
  ++pos;
  while (pos < s.length() && s[pos] != 'l')
    ++pos;
  ++pos;
  while (pos < s.length() && s[pos] != 'l')
    ++pos;
  ++pos;
  while (pos < s.length() && s[pos] != 'o')
    ++pos;

  /**
   * @brief This block determines the result based on the final position of the scanner.
   * Pre-condition: The loops have completed their search for the "hello" subsequence.
   * Logic: If the 'pos' pointer successfully found the final 'o' and is still within the
   * bounds of the string, it means the entire subsequence was found in order.
   */
  if (pos < s.length()) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}