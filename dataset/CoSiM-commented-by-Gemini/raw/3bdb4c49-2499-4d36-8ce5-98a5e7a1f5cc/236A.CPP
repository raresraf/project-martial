/**
 * @file 236A.CPP
 * @brief Solution to the "Boy or Girl" problem (Codeforces 236A).
 *
 * @details This program determines a user's gender for a chat application based on
 * a simple rule: if the number of distinct characters in their username is even,
 * the user is female ("CHAT WITH HER!"); if it's odd, the user is male ("IGNORE HIM!").
 *
 * @algorithm The core of the program is to count the number of unique characters
 * in the input string.
 * 1. The input string is read into a character array.
 * 2. The array is sorted. This places all identical characters adjacent to each other.
 * 3. The program then iterates through the sorted array, comparing each character
 *    `s[i]` with the next one `s[i+1]`.
 * 4. A counter is incremented every time `s[i]` is different from `s[i+1]`,
 *    which effectively counts the number of unique characters.
 * 5. The final output is decided based on the parity (even or odd) of this counter.
 *
 * @complexity
 * Time Complexity: O(L log L), where L is the length of the string. The sort
 *                  operation is the performance bottleneck.
 * Space Complexity: O(L), to store the input string.
 */
// C++ program to sort a string of characters
#include <bits/stdc++.h>
using namespace std;

int main() {
  char s[101];
  int i, n, counter = 0;
  cin >> s;
  n = strlen(s);
  
  // Sort the string to group identical characters together.
  sort(s, s + n);

  /**
   * @brief Block Logic: Counts unique characters by checking for changes in the sorted string.
   * @invariant `counter` holds the number of unique characters found in the prefix `s[0...i]`.
   */
  for (i = 0; i < n; i++) {
    if (s[i] != s[i + 1])
      counter++;
  }

  // The output depends on whether the count of distinct characters is odd or even.
  if (counter % 2 == 1)
    cout << "IGNORE HIM!" << endl;
  else
    cout << "CHAT WITH HER!" << endl;

  return 0;
}
