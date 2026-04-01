/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces (339A).
 * @details The problem requires rearranging a given sum of numbers (containing only 1, 2, and 3)
 * into non-decreasing order. For example, "3+2+1" should become "1+2+3".
 *
 * Algorithm: Extraction and Sorting.
 * 1. Read the input string representing the sum.
 * 2. Extract the numeric characters (at even indices) into a character vector.
 * 3. Sort the vector of characters using the standard library's sort function.
 * 4. Iterate through the sorted vector and print the numbers, interspersing them with '+'.
 *
 * Time Complexity: O(N log N) where N is the number of digits, dominated by the sort operation.
 * Space Complexity: O(N) to store the vector of digits.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  vector<char> v;

  /**
   * Block Logic: Extract the numbers from the sum.
   * The numbers are located at even indices (0, 2, 4, ...).
   */
  for (int i = 0; i < int(s.size()); i += 2) {
    v.push_back(s[i]);
  }

  // Sort the extracted numbers. Since they are characters ('1', '2', '3'),
  // a standard character sort achieves the desired numerical order.
  sort(v.begin(), v.end());

  /**
   * Block Logic: Print the sorted numbers with '+' separators.
   * Invariant: A '+' is printed after each number except for the last one.
   */
  for (int i = 0; i < int(v.size()); i++) {
    if (i != v.size() - 1) {
      cout << v[i] << "+";
    } else {
      cout << v[i];
    }
  }

  return 0;
}
