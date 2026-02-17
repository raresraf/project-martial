/**
 * @file 236A.CPP
 * @brief Solution for Codeforces Problem 236A, "Boy or Girl".
 * @details This program determines a user's gender for a chat application based on whether the number of distinct characters in their username is even or odd.
 *
 * @algorithm Sorting and Linear Scan
 * @description The program reads a username, sorts the characters in it, and then iterates through the sorted string to count the number of unique characters. After sorting, identical characters are grouped together, making it easy to count uniques by comparing adjacent characters.
 * @complexity
 * - Time: O(N log N), dominated by the `std::sort` operation, where N is the length of the string.
 * - Space: O(N), for storing the character array `str`.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  char str[100];
  int n, sum = 0;
  cin >> str;
  n = strlen(str);

  // Sort the character array to group identical characters together.
  sort(str, str + n);
  
  /**
   * @brief Counts the number of unique characters by iterating through the sorted array.
   * @invariant `sum` holds the count of unique characters found in the prefix of the string.
   * @note This loop has a potential out-of-bounds read at `str[i+1]` when `i` is `n-1`.
   *       It works in this context due to the null terminator of the C-style string, but it is not robust.
   */
  for (int i = 0; i < n; i++) {
    if (str[i] != str[i + 1]) {
      sum++;
    }
  }

  /**
   * @brief Determines the output based on the parity of the distinct character count.
   * If the count (`sum`) is even, it prints "CHAT WITH HER!"; otherwise, it prints "IGNORE HIM!".
   */
  if (sum % 2 == 0) {
    // cout<<sum;
    cout << "CHAT WITH HER!";
  } else {
    // cout<<sum;
    cout << "IGNORE HIM!";
  }
  return 0;
}
