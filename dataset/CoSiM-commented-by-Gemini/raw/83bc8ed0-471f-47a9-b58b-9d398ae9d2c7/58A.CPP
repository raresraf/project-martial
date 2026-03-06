/**
 * @file 58A.CPP
 * @brief Attempts to determine if "hello" is a subsequence of a given string.
 *
 * This program reads a string and tries to determine if the characters 'h', 'e',
 * 'l', 'l', 'o' can be found within it.
 *
 * @algorithm Flawed Character Presence Check
 * @details The program uses a set of boolean flags to check if the characters
 * 'h', 'e', 'l', and 'o' appear in the input. It also counts the total number
 * of 'l's. The logic requires that an 'e' appears somewhere after an 'h' has
 * appeared, an 'l' appears after an 'e', and so on.
 *
 * @warning This algorithm is **incorrect** for solving the subsequence problem.
 * It does not enforce the correct order of characters. For example, for the
 * input "helol", it will incorrectly output "YES" because it finds 'h', 'e',
 * 'l', 'o' and sees that the total count of 'l's is >= 2, but it fails to
 * notice that the second 'l' appears after the 'o'. The correct algorithm
 * requires a two-pointer or state machine approach.
 *
 * @complexity Time: O(N), where N is the length of the string. Space: O(N) for the string.
 */
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Main function containing the flawed subsequence checking logic.
 */
int main() {
  string s;
  cin >> s;
  
  // Flags to track if a character type has been seen at all.
  bool hh = false, ee = false, ll = false, oo = false;
  // Counters for the total occurrences of each character.
  int h = 0, e = 0, l = 0, o = 0;

  // Block Logic: Iterate through the string, setting flags and incrementing counters.
  // This does not correctly track the state required for a subsequence.
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == 'h') {
      h++;
      hh = true;
    }
    // This condition is true if an 'e' is found anywhere, as long as an 'h' was seen previously.
    if (s[i] == 'e' && hh) {
      ee = true;
      e++;
    }
    // This is true if an 'l' is found, as long as 'h' and 'e' were seen previously.
    if (s[i] == 'l' && hh && ee) {
      ll = true;
      l++;
    }
    // This is true if an 'o' is found, as long as 'h', 'e', and 'l' were seen previously.
    if (s[i] == 'o' && hh && ee && ll) {
      oo = true;
      o++;
    }
  }
  
  // Final check based on the flawed logic.
  if (hh && ee && ll && oo) {
    // It correctly checks for at least two 'l's, but the sequential order is not guaranteed.
    if (l >= 2)
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  } else
    cout << "NO" << endl;
    
  return 0;
}
