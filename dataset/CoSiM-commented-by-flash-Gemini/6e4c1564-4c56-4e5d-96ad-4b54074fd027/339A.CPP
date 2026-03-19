/**
 * @file 339A.CPP
 * @brief Solution to Codeforces Problem 339A: "Helpful Maths".
 * Rearranges a sum of numbers (1, 2, or 3) given as a string into non-decreasing order.
 * Algorithm: Parse numbers, sort them, then reconstruct the sum string.
 * Time Complexity: O(L log N) where L is the length of the input string and N is the count of numbers.
 * Space Complexity: O(L) to store the input string and extracted numbers.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to solve the "Helpful Maths" problem.
 * Reads a sum string, extracts and sorts the numbers, then prints the sorted sum.
 */
int main() {
  string s; // Stores the input sum as a string (e.g., "1+1+3+2").
  cin >> s;
  
  int array[100]; // Array to store the extracted numbers. Max 100 numbers (since string length is max 100, and numbers are separated by '+').
  int c = 0; // Counter for the actual number of elements stored in 'array'.

  /**
   * Block Logic: Parses the input string 's' to extract individual numbers.
   * Pre-condition: 's' contains digits and '+' characters. 'c' is initialized to 0.
   * Invariant: 'array' contains numbers extracted so far, and 'c' is their count.
   */
  for (int i = 0; i < s.size(); i++) {
    /**
     * Block Logic: Checks if the current character is a digit (not '+').
     * If it's a digit, it's converted to an integer and stored in 'array'.
     */
    if (s[i] != '+') {
      array[c++] = s[i] - '0'; // Converts char digit to int and stores it. '0' (ASCII 48) is subtracted.
    }
  }

  // Sorts the array containing the extracted numbers in non-decreasing order.
  sort(array, array + c);

  /**
   * Block Logic: Prints the sorted numbers, reconstructing the sum string with '+' separators.
   * Pre-condition: 'array' contains sorted numbers, 'c' is their count.
   * Invariant: Numbers are printed in non-decreasing order, with '+' between them.
   */
  for (int i = 0; i < c; i++) {
    cout << array[i]; // Prints the current number.
    /**
     * Block Logic: Adds a '+' separator between numbers, but not after the last one.
     */
    if (i == c - 1) {
      break; // Last number, no '+' needed.
    } else {
      cout << "+"; // Prints '+' separator.
    }
  }

  return 0;
}
