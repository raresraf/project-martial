/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces.
 * @details The problem is to reorder the terms in a given mathematical sum
 * consisting of single-digit numbers (1, 2, 3) into non-decreasing order.
 * For example, "3+2+1" should become "1+2+3".
 *
 * @algorithm
 * 1. Read the input sum as a string.
 * 2. Create a vector of characters to store only the digits from the sum.
 * 3. Iterate through the input string and extract all characters that are digits,
 *    adding them to the vector.
 * 4. Sort the vector of digits. Since the digits are characters '1', '2', '3',
 *    a standard lexicographical sort works correctly.
 * 5. Reconstruct the output string by iterating through the sorted vector,
 *    printing each digit followed by a '+' sign, taking care not to add a
 *    '+' after the last digit.
 *
 * @complexity
 * Time Complexity: O(L log L), where L is the length of the input string.
 * This is dominated by the sort operation on the extracted digits. The number
 * of digits is proportional to L.
 * Space Complexity: O(L) to store the input string and the vector of digits.
 */
#include <algorithm>
#include <bits/stdc++.h>
using namespace std;

// This custom comparison function is defined but not used in the code.
// std::sort on chars uses the default operator<, which is sufficient here.
bool inc(int i, int j) { return (i < j); }

int main() {
  string s; // The input string, e.g., "3+2+1".
  cin >> s;
  
  vector<char> v; // A vector to hold the numerical characters from the string.
  int n = s.length();

  /**
   * @block
   * @brief Extract all digit characters from the input string.
   */
  for (int i = 0; i < n; i++) {
    if (isdigit(s[i])) {
      v.push_back(s[i]);
    }
  }

  // Sort the digits in non-decreasing order.
  sort(v.begin(), v.end());

  int size_v = v.size();
  /**
   * @block
   * @brief Reconstruct and print the sorted sum.
   */
  for (int i = 0; i < size_v; i++) {
    // Special handling for the last element to avoid a trailing '+'.
    if (i == size_v - 1)
      cout << v[i];
    else
      cout << v[i] << "+";
  }

  return 0;
}
