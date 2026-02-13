/**
 * @file 339A.CPP
 * @brief Solution for the "Helpful Maths" problem (Codeforces 339A).
 * @author competitive-programmer
 *
 * @details This program reorders a sum of numbers (1, 2, and 3), given as a
 * string, into non-decreasing order. For example, an input of "3+2+1"
 * is transformed into "1+2+3".
 *
 * Algorithm: Extract, Convert, Sort, and Rebuild
 * 1. Read the input sum into a C-style character array `ch`.
 * 2. Create an integer array `a` to store the numerical values of the digits.
 * 3. Iterate through the input string with a step of 2 to skip the '+' signs.
 *    For each digit character, convert it to an integer and store it in `a`.
 * 4. Sort the integer array `a` in non-decreasing order.
 * 5. Rebuild the output string by iterating through the sorted integer array,
 *    printing each number followed by a '+' (except for the last number).
 *
 * Time Complexity: O(N log N), where N is the number of digits in the sum,
 * dominated by the sorting operation.
 * Space Complexity: O(L) to store the input string and the integer array, where
 * L is the length of the input string.
 */
#include <algorithm>
#include <iostream>
#include <string.h>
using namespace std;
int main() {

  // `ch` stores the input string, `a` stores the extracted integer digits.
  char ch[102];
  int a[53], len;
  // freopen("339A.txt","r",stdin);
  cin >> ch;
  
  /**
   * Block Logic: Extract the numbers from the string.
   * The loop steps by 2 to skip the '+' characters. It converts the character
   * digit to an integer by subtracting the ASCII value of '0' (48).
   */
  for (int i = 0, j = 0; ch[i] != '\0'; i += 2) {
    a[j++] = ch[i] - 48;
  }
  
  len = strlen(ch);
  // Sort the array of integers. The number of integers is (length / 2) + 1.
  sort(a, a + (len / 2) + 1);
  
  /**
   * Block Logic: Reconstruct and print the sorted sum.
   */
  for (int i = 0, j = 0; i < (len / 2) + 1; i++) {
    // Print a '+' after each number except the last one.
    if (i != (len / 2))
      cout << a[i] << "+";
    else
      cout << a[i];
  }

  return 0;
}
