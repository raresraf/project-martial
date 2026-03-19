/**
 * @file 339A.CPP
 * @brief Solution for the "Helpful Maths" problem (Codeforces 339A).
 * @details The program takes a string representing a sum of single-digit numbers
 * (1, 2, and 3), such as "3+2+1", and rearranges it into non-decreasing
 * order, like "1+2+3".
 *
 * Algorithm: Extract, Sort, and Reconstruct
 * 1. The digits are extracted from the input string by iterating through it with
 *    a step of 2, effectively skipping the '+' characters.
 * 2. These extracted digit characters are stored in a `std::vector`.
 * 3. The vector of characters is sorted. Since the ASCII values of '1', '2', '3'
 *    are ordered, this correctly sorts the numbers.
 * 4. The sorted sum is reconstructed by iterating through the sorted vector,
 *    printing each digit followed by a '+' (except for the last digit).
 *
 * Time Complexity: O(L log L), where L is the number of digits in the sum. This is
 * dominated by the sorting operation.
 * Space Complexity: O(L) to store the digits in a vector.
 */

#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;
int main() {
  // x: the input string representing the sum.
  string x;
  cin >> x;
  
  int siz = x.size();
  // vec: a vector to store the character digits from the sum.
  vector<char> vec;

  /**
   * Block Logic: Extract digits from the input string.
   * The loop increments by 2 to skip the '+' characters.
   */
  for (int j = 0; j < siz;) {
    vec.push_back(x[j]);
    j += 2;
  }
  
  // Sort the vector of digit characters. '1' < '2' < '3'.
  sort(vec.begin(), vec.end());
  
  siz = vec.size();
  
  /**
   * Block Logic: Reconstruct and print the sorted sum string.
   * Pre-condition: 'vec' contains the sorted digits.
   */
  for (int j = 0; j < siz; j++) {
    // Print the current digit.
    cout << vec[j];
    // If this is not the last digit, print a '+' separator.
    if (j + 1 < siz) {
      cout << "+";
    }
  }
  return 0;
}