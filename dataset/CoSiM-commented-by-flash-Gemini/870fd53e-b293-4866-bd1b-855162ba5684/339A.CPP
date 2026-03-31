/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces (Problem 339A).
 * This program reorders a sum of numbers (e.g., "1+3+2") such that the numbers
 * are in non-decreasing order.
 *
 * Algorithm: Parse the numbers from the input string, sort them, and then
 * reconstruct the string with the sorted numbers.
 * Time Complexity: O(L log N), where L is the length of the input string and N is the count of numbers.
 *                  The dominant factor is sorting N numbers.
 * Space Complexity: O(N) to store the parsed numbers.
 */

/*
Codeforces Round #197 (Div. 2) - A. Helpful Maths
https://codeforces.com/problemset/problem/339/A
*/

#include <bits/stdc++.h> // @brief Includes many standard libraries (e.g., iostream, string, algorithm, vector, cctype).
using namespace std;
// @brief Macro to speed up C++ I/O operations (disables sync with C stdio and unties cin/cout).
#define FAST_INP                                                               \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL)

int main() {
  /**
   * @brief Main function to reorder a sum of numbers.
   * Reads an input string containing numbers separated by '+', extracts these numbers,
   * sorts them in non-decreasing order, and then prints the reordered sum string.
   */
  FAST_INP; // @brief Apply I/O optimization.
  string s; // @brief The input string containing numbers and '+' signs.
  cin >> s;
  vector<int> v; // @brief Vector to store the parsed numbers.
  
  // Block Logic: Extract numbers from the input string.
  // Iterate through the string, checking if each character is a digit.
  // If it is, convert it from char to int and add to the vector.
  // Functional Utility: `isdigit(s[i])` checks if `s[i]` is a digit.
  // `s[i] - '0'` converts a character digit to its integer value.
  for (int i = 0; i < s.size(); i++) {
    if (isdigit(s[i]))
      v.push_back(s[i] - '0'); // Inline: Convert char digit to int and add to vector.
  }
  
  // Block Logic: Sort the extracted numbers in non-decreasing order.
  // This is a crucial step for reordering the sum according to the problem statement.
  sort(v.begin(), v.end());
  
  string ans = ""; // @brief The resulting reordered sum string.
  
  // Block Logic: Construct the output string from the sorted numbers.
  // Iterate through the sorted numbers, appending each as a character to 'ans'.
  // Add a '+' separator between numbers, but not after the last one.
  for (int i = 0; i < v.size(); i++) {
    ans += (v[i] + '0'); // Inline: Convert int to char digit and append.
    if (i != v.size() - 1) // Inline: Check if it's not the last number.
      ans += "+"; // Inline: Append '+' as a separator.
  }
  cout << ans << "\n"; // @brief Print the final reordered sum string followed by a newline.
  return 0; // @brief Indicates successful program termination.
}
