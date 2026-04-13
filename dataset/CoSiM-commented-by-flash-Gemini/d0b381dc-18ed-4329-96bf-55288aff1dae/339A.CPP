/**
 * @file 339A.CPP
 * @brief This program solves the "Helpful Maths" problem. It takes an input string
 * representing a sum of '1', '2', and '3' digits separated by '+' signs (e.g., "3+2+1")
 * and outputs them in non-decreasing order (e.g., "1+2+3").
 * Algorithm: Parses the input string from a character array, extracts digits into an
 * integer array, sorts the integer array using bubble sort, and then reconstructs
 * the string by printing the sorted numbers with '+' signs in between.
 * Time Complexity: O(L^2) where L is the length of the input string, due to bubble sort.
 * Space Complexity: O(L) for storing the character and integer arrays.
 */
#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience, includes many standard libraries.
using namespace std;     // Brings all names from the std namespace into the current scope.

int main() {
  char arr[101]; // Character array to store the input sum string (max 100 chars + null terminator).
  // Reads the input sum string into the character array.
  scanf("%s", &arr);
  int len = strlen(arr); // Calculates the length of the input string.
  int x;                 // Stores the number of digits in the input string.
  int temp, i, j;        // Temporary variable for swapping, and loop counters.

  // Calculates the number of digits. Since digits are separated by '+', the number of digits
  // is (length + 1) / 2.
  x = (len + 1) / 2;
  int a[x]; // Declares a Variable Length Array (VLA) to store the integer values of the digits.
            // Note: VLAs are a C99 feature and a GNU extension in C++, not standard C++.

  /**
   * Block Logic: Parses the character array `arr` to extract digits and convert them to integers.
   * It iterates through `arr` at even indices (where digits are located), converts the character
   * digit ('1', '2', '3') to its integer equivalent (1, 2, 3), and stores it in array `a`.
   * Invariant: Array `a` contains all numerical digits extracted from `arr`.
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
   * Block Logic: Sorts the integer array `a` (containing the digits) in ascending order using Bubble Sort.
   * Invariant: After each pass of the outer loop, the largest unsorted element is moved to its correct position.
   */
  for (i = 0; i < x - 1; i++) {
    for (j = 0; j < x - 1 - i; j++) {
      if (a[j] > a[j + 1]) { // Compares adjacent elements.
        temp = a[j];         // Swaps elements if they are in the wrong order.
        a[j] = a[j + 1];
        a[j + 1] = temp;
      }
    }
  }

  /**
   * Block Logic: Reconstructs and prints the sorted sum string.
   * It iterates through the sorted integer array `a` and prints each digit.
   * A '+' character is printed after each digit, except for the last one.
   * Invariant: The output string progressively forms the sorted sum.
   */
  for (i = 0, j = 0; i < len; i += 2, j++) {
    printf("%d", a[j]); // Prints the current digit.
    if (i == len - 1) { // Checks if it's the last digit in the original string (i.e., no '+' after it).
      break;            // Exits loop to avoid printing a trailing '+'.
    }
    printf("%c", '+'); // Prints a '+' separator.
  }
  // Standard program exit.
  return 0;
}
