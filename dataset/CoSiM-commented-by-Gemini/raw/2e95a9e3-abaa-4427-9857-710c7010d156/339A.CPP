/**
 * @file 339A.CPP
 * @brief Solution to the "Helpful Maths" problem on Codeforces.
 * @details This program sorts a given sum of single-digit numbers (1, 2, or 3) into non-decreasing order.
 * For example, "3+2+1" becomes "1+2+3". This implementation cleverly extracts the digits into a
 * separate string, sorts it, and then reconstructs the sum.
 *
 * Algorithm:
 * 1. Read the entire sum as a string `a`.
 * 2. Create a new string `b` to hold only the digits.
 * 3. Iterate through the input string `a`. If a character is not a '+', append it to `b`.
 * 4. After the loop, `b` contains all the digits from the sum (e.g., "321").
 * 5. Sort the string `b` lexicographically using `std::sort`. Since the characters are '1', '2', '3',
 *    a standard character sort correctly orders them ("123").
 * 6. Reconstruct the output string by iterating through the sorted digit string `b`.
 *    - Print each digit followed by a '+', except for the last digit.
 *    - Print the last digit followed by a newline.
 *
 * Time Complexity: O(L log L), where L is the number of digits in the string. This is dominated by the sorting operation on the digit string.
 * Space Complexity: O(L), to store the string of digits `b`.
 */
#include <algorithm>
#include <iostream>

// Macro to optimize C++ I/O operations by decoupling from C stdio and un-tying cin/cout.
#define optimizar_io                                                           \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);

using namespace std;
int main() {
  optimizar_io;
  string a, b = "";
  cin >> a;

  /**
   * Block Logic: Extract all digit characters from the input sum string.
   * Pre-condition: 'a' is a string like "d1+d2+...".
   * Invariant: 'b' contains all characters from 'a' that are not '+'.
   */
  for (int i = 0; i < (int)a.size(); ++i)
    if (a[i] != '+')
      b += a[i];

  // Sort the string of digits. Standard lexicographical sort works correctly for single digits.
  sort(b.begin(), b.end());

  /**
   * Block Logic: Print the sorted digits, interspersing them with '+'.
   * Pre-condition: 'b' is a string of sorted digit characters.
   * Invariant: Builds the final output string in the correct "d1+d2+..." format.
   */
  for (int i = 0; i < (int)b.size() - 1; ++i)
    cout << b[i] << "+";
  
  // Print the last digit without a trailing '+'.
  cout << b[b.size() - 1] << "\n";
  
  return 0;
}
