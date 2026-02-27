/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces.
 * @details The problem requires rearranging a given mathematical sum of single
 * digits (1, 2, 3) into non-decreasing order. For example, the input "3+2+1"
 * should produce the output "1+2+3".
 *
 * @algorithm
 * 1. Read the input sum as a C-style string.
 * 2. Create an integer array to store the numbers from the sum.
 * 3. Iterate through the input string.
 *    a. If a character is a digit, convert it from its ASCII character value to
 *       its integer value (e.g., '1' to 1) by subtracting '0'.
 *    b. Store the integer value in the integer array.
 * 4. Sort the integer array in non-decreasing order.
 * 5. Reconstruct the sum by iterating through the sorted integer array, printing
 *    each number followed by a '+', ensuring no trailing '+' is printed.
 *
 * @complexity
 * Time Complexity: O(L log L), where L is the length of the input string. This
 * complexity is dominated by the sorting step. The number of digits to sort
 * is proportional to L.
 * Space Complexity: O(L) to store the input string and the integer array.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // C-style string to hold the input sum, e.g., "3+2+1".
  char str[105];
  // An integer array to hold the extracted numerical values.
  int a[105];
  int i, j;

  // This loop structure can handle multiple test cases, though this problem has only one.
  while (scanf("%s", str) != EOF) {
    /**
     * @block
     * @brief Extract digits from the string and convert them to integers.
     */
    for (i = 0, j = 0; i < strlen(str); i++) {
      if (str[i] >= '0' && str[i] <= '9') {
        // Convert char digit to int value (e.g., '1' -> 1) and store it.
        a[j++] = str[i] - '0';
      }
    }
    
    // Sort the array of integers. `j` holds the count of numbers.
    sort(a, a + j);

    /**
     * @block
     * @brief Print the sorted numbers, formatted as a sum.
     */
    for (i = 0; i < j; i++) {
      cout << a[i];
      // Print a '+' after each number except the last one.
      if (i < j - 1)
        cout << "+";
    }
    cout << endl;
  }
  return 0;
}
