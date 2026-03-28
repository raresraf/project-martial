/**
 * @file 58A.CPP
 * @brief Checks if "hello" is a subsequence of a given string.
 *
 * This program reads a string and determines if the word "hello" can be
 * formed by deleting zero or more characters from the input string.
 *
 * @b Algorithm: The program uses a greedy approach. It scans the string
 *    sequentially for each character of "hello". A single index `i` is used
 *    to maintain the position in the input string. After finding a character,
 *    the search for the next character begins from the subsequent position.
 *
 * @b Time_Complexity: O(L), where L is the length of the input string, as it
 *    requires only a single pass through the string.
 * @b Space_Complexity: O(L) to store the input string.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  string str;
  cin >> str;
  int i = 0, ctr = 0;

  /**
   * @brief This series of loops sequentially searches for the characters of "hello".
   * The index 'i' is not reset between loops, so the search for the next
   * character continues from where the last one was found.
   */

  // Find 'h'
  for (; i < str.size(); i++)
    if (str[i] == 'h') {
      i++, ctr++;
      break;
    }
  // Find 'e'
  for (; i < str.size(); i++)
    if (str[i] == 'e') {
      i++, ctr++;
      break;
    }
  // Find first 'l'
  for (; i < str.size(); i++)
    if (str[i] == 'l') {
      i++, ctr++;
      break;
    }
  // Find second 'l'
  for (; i < str.size(); i++)
    if (str[i] == 'l') {
      i++, ctr++;
      break;
    }
  // Find 'o'
  for (; i < str.size(); i++)
    if (str[i] == 'o') {
      i++, ctr++;
      break;
    }
  
  // If the counter has reached 5, all characters were found in order.
  cout << (ctr >= 5 ? "YES" : "NO");
  return 0;
}