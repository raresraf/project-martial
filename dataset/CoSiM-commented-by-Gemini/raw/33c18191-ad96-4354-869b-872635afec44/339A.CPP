/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem by rearranging a sum.
 * @details This program reformats a string representing a sum of single-digit numbers
 * (e.g., "3+1+2") into a new string where the summands appear in non-decreasing
 * order (e.g., "1+2+3").
 *
 * @algorithm The algorithm consists of three main stages:
 * 1. Extraction: The input string is parsed to extract all numeric characters. These
 *    are appended to a separate string, effectively creating a string of digits.
 * 2. Sorting: This string of digits is sorted lexicographically using `std::sort`.
 *    For single digits, this is equivalent to sorting them numerically.
 * 3. Reconstruction: The sorted sum is printed by first printing the first digit, then
 *    looping through the remaining digits and printing each one prefixed by a '+' sign.
 *
 * @time_complexity O(N log N), where N is the number of digits in the sum. The performance
 * is dominated by the sorting operation on the string of digits.
 * @space_complexity O(N) to store the extracted digits in a separate string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s, a;
  int i, l;
  cin >> s;
  l = s.size();
  /**
   * Block Logic: Extract all numeric characters from the input sum string
   * and append them to string `a`.
   */
  for (i = 0; i < l; i++) {
    if (s[i] != '+') {
      a += s[i];
    }
  }
  // Sort the string of digits. For single digits, lexicographical and
  // numerical order are the same.
  sort(a.begin(), a.end());
  l = a.size();
  // Print the first digit of the new sum.
  cout << a[0];
  /**
   * Block Logic: Iterate through the rest of the sorted digits and print them,
   * prefixed with a '+' to reconstruct the sum string.
   */
  for (i = 1; i < l; i++) {
    cout << "+" << a[i];
  }
}
