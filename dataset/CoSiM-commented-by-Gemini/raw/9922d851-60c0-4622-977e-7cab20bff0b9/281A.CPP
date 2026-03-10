/**
 * @file 281A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Word Capitalization").
 * This program capitalizes the first letter of a given word.
 *
 * @algorithm Direct character manipulation using the `toupper` standard library function.
 * @time_complexity O(L), where L is the length of the string, dominated by I/O.
 * The capitalization logic itself is O(1).
 * @space_complexity O(L_max), where L_max is the maximum allowed length of the
 * string, due to the fixed-size character array.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  char str[1001]; // C-style character array to store the word.
  scanf("%s", str);

  /**
   * @brief Capitalizes the first letter of the string.
   * The `toupper` function from the `<cctype>` library is used to convert the
   * first character to uppercase. This is a robust method that handles
   * non-lowercase characters correctly (i.e., leaves them unchanged).
   */
  str[0] = toupper(str[0]);

  cout << str << endl;
}
