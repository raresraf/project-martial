/**
 * @file 339A.CPP
 * @brief Solution for a string sorting problem, "Helpful Maths".
 * @details The program reads a string representing a sum of single-digit numbers
 * (1s, 2s, and 3s), sorts the numbers, and prints the rearranged sum. This
 * implementation uses a counting sort algorithm for efficiency.
 *
 * Algorithm:
 * 1. Read the input string (e.g., "3+1+2+1").
 * 2. Use a frequency array (of size 4) to count the occurrences of each digit ('1', '2', '3').
 *    This is the core of the counting sort algorithm.
 * 3. Iterate to construct the output string. The output string will have the same
 *    length as the input.
 * 4. For even indices of the output string (where digits go), find the smallest
 *    digit (`k`) that still has a non-zero count in the frequency array.
 * 5. Print that digit `k` and decrement its count.
 * 6. For odd indices (where plus signs go), print a '+'.
 *
 * Time Complexity: O(L), where L is the length of the input string. This is more
 * efficient than a general-purpose sort.
 * Space Complexity: O(1), as the frequency array has a constant size.
 */
#include <bits/stdc++.h>
#include <ctype.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  int len = s.size();
  int a[4] = {0}; // Frequency array for digits 1, 2, 3. a[0] is unused.
  
  /**
   * @block
   * @description First pass: Count the occurrences of each digit (1, 2, or 3).
   * This populates the frequency array `a`.
   */
  for (int i = 0; i < len; i++) {
    if (s[i] != '+') {
      // Convert char '1'..'3' to int 1..3 and increment its count.
      a[(s[i] - '0')]++;
    }
  }

  int k = 1; // `k` tracks the current digit to be printed (1, 2, or 3).
  
  /**
   * @block
   * @description Second pass: Construct and print the sorted sum.
   * The loop builds the output string piece by piece.
   */
  for (int i = 0; i < len; i++) {
    // Even indices are for digits.
    if (i % 2 == 0) {
      // Find the smallest available digit to print next.
      while (a[k] == 0)
        k++;

      cout << k;
      a[k]--; // Decrement the count of the printed digit.
    } 
    // Odd indices are for the '+' separator.
    else {
      cout << "+";
    }
  }

  return 0;
}