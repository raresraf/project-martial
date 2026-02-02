#include <iostream>
#include <string>

// http://codeforces.com/problemset/problem/71/A
// A. Way Too Long Words

using namespace std;

/**
 * @brief This program shortens long words according to a specific rule.
 *
 * It reads an integer `n` followed by `n` words. For each word, if its
 * length is greater than 10, it is replaced by an abbreviation consisting
 * of its first character, the number of characters between the first and
 * last character, and its last character. Otherwise, the word is printed
 * as is.
 *
 * @return 0 on successful execution.
 */
int main() {
  int n; // int data type accept characters but when u cout it it will output 0;
  string s; // string data type accept characters and numbers as well;
  cin >> n;
  while (n--) {
    cin >> s;
    if (s.length() > 10) {
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    } else {
      cout << s << endl;
    }
  }
  return 0;
}
