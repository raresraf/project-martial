/**
 * @file 58A.CPP
 * @brief Checks if the string "hello" is a subsequence of an input string.
 *
 * This program reads a string and determines if the word "hello" can be obtained
 * by deleting zero or more characters from the input. This implementation uses a
 * series of linear scans to find each character of "hello" in order.
 *
 * Algorithm:
 * 1. Read the input string 's'.
 * 2. Initialize an index 'i' to 0.
 * 3. Scan forward from 'i' until the character 'h' is found. Advance 'i'.
 * 4. Scan forward from the new 'i' until 'e' is found. Advance 'i'.
 * 5. Scan forward for the first 'l'. Advance 'i'.
 * 6. Scan forward for the second 'l'. Advance 'i'.
 * 7. Scan forward for 'o'.
 * 8. If the index 'i' is still within the bounds of the string after finding 'o',
 *    it means all characters were found in the correct order. Print "YES".
 * 9. Otherwise, if any scan reached the end of the string without finding its
 *    target character, print "NO".
 *
 * Time Complexity: O(L), where L is the length of the input string, as it
 * involves a single pass through the string.
 * Space Complexity: O(L) to store the input string.
 */
#include <bits/stdc++.h>

#include <string.h>
using namespace std;

int main() {
  string s;
  int i = 0;

  cin >> s;

  // Find the first 'h'.
  while (i < s.size() && s[i] != 'h')
    ++i;
  ++i;

  // Find the first 'e' after 'h'.
  while (i < s.size() && s[i] != 'e')
    ++i;
  ++i;
  
  // Find the first 'l' after 'e'.
  while (i < s.size() && s[i] != 'l')
    ++i;
  ++i;

  // Find the second 'l' after the first 'l'.
  while (i < s.size() && s[i] != 'l')
    ++i;
  ++i;

  // Find the 'o' after the second 'l'.
  while (i < s.size() && s[i] != 'o')
    ++i;

  /**
   * Block Logic: Final check to determine if the subsequence was found.
   * If the index 'i' is still within the string's bounds, it means the final
   * character 'o' was successfully found.
   */
  if (i < s.size()) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
