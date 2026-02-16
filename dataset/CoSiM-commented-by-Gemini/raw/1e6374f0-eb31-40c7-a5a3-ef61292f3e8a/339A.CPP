/**
 * @file 339A.CPP
 * @brief Solution for Codeforces problem 339A, "Helpful Maths".
 *
 * @description
 * This program rearranges a given mathematical sum of single-digit numbers
 * (1, 2, and 3) into non-decreasing order. For example, an input of "3+2+1"
 * would be output as "1+2+3".
 *
 * This specific implementation uses a highly efficient Counting Sort algorithm,
 * which is ideal for data with a small, known range of values.
 *
 * @algorithm
 * 1. **Counting Phase**:
 *    a. Create an integer array `count` of size 3 to store the frequencies of
 *       the numbers '1', '2', and '3'.
 *    b. Iterate through the input string, stepping by 2 to read only the digits.
 *    c. For each digit, increment the corresponding counter in the `count` array.
 * 2. **Rebuilding Phase**:
 *    a. Iterate through the `count` array (from '1's to '3's).
 *    b. For each number, overwrite the original input string at the digit
 *       positions (0, 2, 4, ...) with the character of that number, as many
 *       times as it was counted.
 * 3. Print the modified, now sorted, string.
 *
 * @time_complexity O(L), where L is the length of the string. This is more
 *   efficient than a general-purpose O(L log L) sort.
 * @space_complexity O(1), as the extra space used (the count array) is constant
 *   and does not depend on the input size. The string is modified in-place.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string str;
  cin >> str;
  // `count` array stores the frequency of '1's, '2's, and '3's.
  // count[0] for '1', count[1] for '2', count[2] for '3'.
  int count[3] = {};

  // Block Logic: Counting Phase of Counting Sort.
  // Iterate through the string, only processing digit characters.
  for (int i = 0; i < str.size(); i += 2) {
    switch (str[i]) {
    case '1':
      count[0]++;
      break;
    case '2':
      count[1]++;
      break;
    case '3':
      count[2]++;
      break;
    }
  }

  // Block Logic: Rebuilding Phase. Overwrite the string in sorted order.
  int index = 0, i = 0; // `index` for count array, `i` for string position.
  while (i < str.size()) {
    // If we have finished placing all numbers of the current type (e.g., '1's),
    // move to the next type (e.g., '2's).
    if (count[index] == 0) {
      index++;
    } else {
      // Place the character corresponding to the current `index` into the string.
      switch (index) {
      case 0:
        str[i] = '1';
        break;
      case 1:
        str[i] = '2';
        break;
      case 2:
        str[i] = '3';
        break;
      }
      // Move to the next digit position in the string.
      i += 2;
      // Decrement the count for the number we just placed.
      count[index]--;
    }
  }
  cout << str;
}
