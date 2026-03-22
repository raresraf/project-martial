/**
 * @file 339A.CPP
 * @brief Rearranges a mathematical sum into non-decreasing order.
 *
 * This program solves the "Helpful Maths" problem. It takes a string representing
 * a sum of single-digit numbers (1, 2, or 3), such as "3+2+1", and rearranges
 * the numbers in non-decreasing order to produce a new sum string, like "1+2+3".
 *
 * @b Algorithm: Extraction, Sorting, and Reconstruction
 *   1. The numbers are extracted from the input string and stored in an integer array.
 *   2. The integer array is sorted. This implementation uses Bubble Sort.
 *   3. The sorted numbers are used to construct the final output string, with '+'
 *      symbols separating them.
 *
 * @b Time_Complexity: O(N^2), where N is the number of terms in the sum. The
 *   bottleneck is the Bubble Sort algorithm. Using `std::sort` would be O(N log N).
 * @b Space_Complexity: O(N) to store the extracted numbers.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // `arr` stores the input string like "3+2+1".
  char arr[101];
  scanf("%s", &arr);
  int len = strlen(arr);
  
  int x, temp, i, j;
  // `x` is the number of integer terms in the sum.
  x = (len + 1) / 2;
  // `a` will store the integer values of the terms.
  int a[x];
  
  /**
   * @brief Extract numbers from the input string.
   * Numbers are at even indices (0, 2, 4, ...).
   */
  for (i = 0, j = 0; i < len; i += 2, j++) {
    if (arr[i] == '1') {
      a[j] = 1;
    } else if (arr[i] == '2') {
      a[j] = 2;
    } else if (arr[i] == '3') {
      a[j] = 3;
    }
  }
  
  /**
   * @brief Sort the array of numbers using Bubble Sort.
   */
  for (i = 0; i < x - 1; i++) {
    for (j = 0; j < x - 1 - i; j++) {
      if (a[j] > a[j + 1]) {
        temp = a[j];
        a[j] = a[j + 1];
        a[j + 1] = temp;
      }
    }
  }
  
  /**
   * @brief Reconstruct and print the sorted sum string.
   */
  for (i = 0, j = 0; i < len; i += 2, j++) {
    printf("%d", a[j]);
    // Print a '+' after each number except the last one.
    if (i == len - 1) {
      break;
    }
    printf("%c", '+');
  }
  return 0;
}
