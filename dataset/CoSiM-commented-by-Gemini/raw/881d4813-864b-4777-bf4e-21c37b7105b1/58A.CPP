/**
 * @file 58A.CPP
 * @brief An incorrect attempt to solve the "Chat room" problem (Codeforces 58A).
 *
 * @details
 * The program is intended to check if the word "hello" is a subsequence of a
 * given input string. However, the implemented logic is fundamentally flawed
 * and does not correctly perform a subsequence check.
 *
 * Flaw in the Algorithm:
 * The core function `FindHelloFromALine` attempts to build the word "hello"
 * by collecting characters from the input string. It adds the characters 'h',
 * 'e', 'l', 'o' to a temporary string if they are not already present (with a
 * special case to allow 'l' twice). This approach completely ignores the
 * required *order* of the characters for a subsequence. For example, for an
 * input "olleh", it would find all the necessary characters but in the wrong
 * order, and the logic would fail to construct "hello". The code does not
 * solve the problem correctly.
 *
 * A correct approach would use a greedy two-pointer algorithm to verify the
 * characters 'h', 'e', 'l', 'l', 'o' appear in the correct sequence.
 *
 * Time Complexity: O(N*M) where N is the length of the input string and M is the
 *                  length of the temporary string, due to the inefficient
 *                  `IfCharInString` call inside the main loop.
 * Space Complexity: O(N) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;
/**
 * @brief Helper function to check if a character exists within a string.
 * @param s The character to find.
 * @param str The string to search in.
 * @return True if the character is found, false otherwise.
 */
bool IfCharInString(char s, string str) {
  bool appear = false;

  for (int i = 0; i < str.size(); i++) {
    if (str[i] == s) {
      appear = true;
      break;
    }
  }
  return appear;
}

/**
 * @brief A flawed function intended to find "hello" in a string.
 * This function does not respect the order of characters and thus fails to
 * correctly identify a subsequence. It merely collects a set of characters.
 */
string FindHelloFromALine(string str) {
  char previous = ' ';
  short int count_l = 0;
  string temp;

  for (int i = 0; i < str.size(); i++) {
    if (str[i] == 'h' || str[i] == 'e' || str[i] == 'l' || str[i] == 'o') {
      if (str[i] == 'l')
        count_l++;

      // This condition incorrectly adds characters based on uniqueness, not order.
      if ((!IfCharInString(str[i], temp)) || (str[i] == 'l' && count_l == 2)) {
        temp += str[i];
      }
    }
    previous = str[i];
  }

  return temp;
}

int main() {
  string s;

  cin >> s;

  string output = FindHelloFromALine(s);

  // The final check is on the size of the incorrectly constructed string.
  if (output.size() == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}

/**
ahhellllloou
output:
YES

hlelo
output:
NO

pnnepelqomhhheollvlo
output:
YES
**/
