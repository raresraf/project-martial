/**
 * @file 339A.CPP
 * @brief Solution for the "Helpful Maths" problem (Codeforces 339A).
 *
 * @algorithm
 * The problem asks to rearrange a given sum of single-digit numbers (1, 2, 3)
 * into non-decreasing order. For example, "3+2+1" should become "1+2+3".
 *
 * The algorithm implements a simple "extract, sort, reformat" strategy:
 * 1. Read the entire input sum as a string.
 * 2. Create a vector of integers to store the numbers.
 * 3. Iterate through the input string. If a character is a digit, convert it
 *    to an integer and add it to the vector. Ignore the '+' characters.
 * 4. Sort the vector of integers in non-decreasing order.
 * 5. Iterate through the sorted vector and print the numbers, separated by '+'.
 *
 * @complexity
 * - Time Complexity: O(N log N), where N is the number of digits in the sum.
 *   The dominant operation is sorting the vector of numbers.
 * - Space Complexity: O(N), to store the numbers in a vector.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  string s; // To store the input sum string, e.g., "3+2+1".
  cin >> s;
  vector<int> v; // To store the integer operands from the sum.
  /**
   * @brief First pass: Parse the string to extract the numbers.
   */
  for (int i = 0; i < s.size(); i++) {
    // Ignore the '+' characters.
    if (s[i] == '+')
      continue;
    else
      // Convert character digit to integer and add to the vector.
      // e.g., '1' - '0' = 1
      v.push_back(s[i] - '0');
  }
  // Sort the vector of numbers in non-decreasing order.
  sort(v.begin(), v.end());
  /**
   * @brief Second pass: Print the sorted numbers in the "sum" format.
   */
  for (int i = 0; i < v.size(); i++) {
    // If it's not the last number, print a '+' after it.
    if (i != v.size() - 1)
      cout << v[i] << "+";
    else
      // If it is the last number, print a newline.
      cout << v[i] << endl;
  }
  return 0;
}
