#include <iostream>

using namespace std;

/**
 * @brief This program compares two strings lexicographically, ignoring case.
 *
 * It reads two strings from the standard input, converts them to lowercase,
 * and then compares them character by character.
 *
 * @return 0 on successful execution.
 *
 * The program prints:
 * - "1" if the first string is lexicographically greater than the second.
 * - "-1" if the first string is lexicographically smaller than the second.
 * - "0" if the strings are equal.
 */
int main() {
  int ln1, ln2, i, k = 0;
  string str1, str2;
  cin >> str1 >> str2;
  ln1 = str1.length();
  ln2 = str2.length();

  if (ln1 == ln2) {
    for (i = 0; i < ln1; i++) {
      if (tolower(str1[i]) > tolower(str2[i])) {
        k = 1;
        break;
      }
      if (tolower(str1[i]) < tolower(str2[i])) {
        k = -1;
        break;
      }
      if (tolower(str1[i]) < tolower(str2[i])) {
        k = 0;
      }
    }
    if (k == 1) {
      cout << "1";
    }
    if (k == -1) {
      cout << "-1";
    }
    if (k == 0) {
      cout << "0";
    }
  }

  return 0;
}
