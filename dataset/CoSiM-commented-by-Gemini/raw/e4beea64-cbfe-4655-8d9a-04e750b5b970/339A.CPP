/**
 * @file 339A.CPP
 * @brief Solution to Codeforces problem 339A - Helpful Maths.
 * @description This program rearranges a string representing a sum of single-digit numbers
 *              (e.g., "3+1+2") into non-decreasing order (e.g., "1+2+3").
 *
 * Algorithm: This solution uses a clever two-step sorting and filtering process.
 * 1. Read the entire input string, including the '+' signs.
 * 2. Sort the entire string lexicographically. Because the ASCII value of '+' is less than
 *    the ASCII values of '1', '2', and '3', this effectively moves all '+' signs to the
 *    beginning of the string and sorts the digits after them (e.g., "3+1+2" becomes "++123").
 * 3. Use the standard C++ erase-remove idiom to eliminate all '+' characters from the string,
 *    leaving only the sorted digits (e.g., "++123" becomes "123").
 * 4. Iterate through the string of sorted digits and print them, inserting a '+' separator
 *    between each digit.
 *
 * Time Complexity: O(N log N), dominated by the initial sorting of the string, where N
 *                  is the length of the input string.
 * Space Complexity: O(N) to store the string.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  string str1;
  int i;
  cin >> str1;

  // Step 1: Sort the entire string. This groups '+' signs at the beginning
  // and sorts the numbers '1', '2', '3' correctly based on ASCII values.
  sort(str1.begin(), str1.end());

  // Step 2: Use the erase-remove idiom to eliminate all '+' characters.
  str1.erase(remove(str1.begin(), str1.end(), '+'), str1.end());
  
  auto size = str1.length();
  /**
   * Block Logic: Reconstruct the sum string by iterating through the sorted digits.
   */
  for (i = 0; i < size; i++) {
    cout << str1[i];
    // Add a '+' separator after each number except the last one.
    if (i != size - 1) {
      cout << '+';
    }
  }
  return 0;
}
