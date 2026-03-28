/**
 * @file 339A.CPP
 * @brief Rearranges a sum of numbers into non-decreasing order.
 *
 * This program solves the "Helpful Maths" problem. It takes a string representing
 * a sum of single-digit numbers (e.g., "3+1+2") and reorders the numbers to
 * print the sum in non-decreasing order (e.g., "1+2+3").
 *
 * @b Algorithm: The core logic is to first extract all the numbers from the
 *    input string, ignoring the '+' signs. Then, these numbers (as characters)
 *    are sorted. Finally, the sorted numbers are printed out, interleaved with
 *    '+' signs to form the new, ordered sum.
 * @b Time_Complexity: O(L log L), where L is the length of the string of numbers.
 *    The `sort` operation is the dominant factor.
 * @b Space_Complexity: O(L) to store the extracted numbers in a separate string.
 */
#include <algorithm>
#include <bits/stdc++.h>
#include <iostream>
#include <string>
using namespace std;
int main() {
  string str, str2;
  cin >> str;

  // Block Logic: Extract all the number characters from the input string.
  for (int i = 0; i < str.size(); i++) {
    if (str[i] != '+') {
      str2.push_back(str[i]);
    }
  }
  
  // Sort the string of numbers. Since they are single digits, character sorting
  // is equivalent to numerical sorting.
  sort(str2.begin(), str2.end());
  
  // Block Logic: Reconstruct and print the sorted sum.
  // Print the first number.
  cout << str2[0];
  // Print the rest of the numbers, each preceded by a '+'.
  for (int i = 1; i < str2.size(); i++) {
    cout << '+' << str2[i];
  }
  cout << endl;

  return 0;
}