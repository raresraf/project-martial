/**
 * @6a2cfa51-5200-4bb8-a8c0-38f2d35875a6/339A.CPP
 * @brief Problem: Helpful Maths.
 *        Given a string representing a sum of numbers (e.g., "3+1+2"),
 *        reorder the numbers in non-decreasing order and print the resulting sum.
 *        This implementation uses char arrays and ASCII value comparison.
 * Algorithm: Parse, Sort, and Print using char array.
 *            1. Read the entire input string into a character array.
 *            2. Sort the characters in the array. This effectively sorts the numbers
 *               and the '+' signs, but the '+' signs will end up at the beginning
 *               due to their ASCII value.
 *            3. Iterate through the sorted array. Print only characters that are
 *               numbers (ASCII value > 43, which is the ASCII for '+').
 *            4. Re-insert '+' signs between the printed numbers.
 * Time Complexity: O(N log N) where N is the length of the input string.
 *                  `strlen` is O(N). `sort` for a char array of length N is O(N log N).
 *                  Printing is O(N).
 * Space Complexity: O(N) for storing the input character array.
 */
#include <algorithm> // Functional Utility: Provides the std::sort function.
#include <cstring>   // Functional Utility: Provides the strlen function.
#include <iostream>  // Functional Utility: Provides cin and cout for input/output.
using namespace std;

int main() {
  char a[110]; // Functional Utility: Declares a character array to store the input string.
               // Invariant: Assumes input string length does not exceed 109 characters.

  // Block Logic: Reads the input sum string into the character array 'a'.
  cin >> a;
  int n = strlen(a); // Functional Utility: Calculates the length of the string 'a'.
  // Functional Utility: Sorts the characters in the array 'a' in ascending order.
  // Algorithm: Typically IntroSort (hybrid of QuickSort, HeapSort, InsertionSort).
  sort(a, a + n);
  
  // Block Logic: Iterates through the sorted character array to print the reordered sum.
  // Pre-condition: 'a' is sorted.
  // Invariant: Only numeric characters are printed, separated by '+'.
  for (int i = 0; i < n; i++) {
    // Block Logic: Checks if the current character is a digit (by comparing its ASCII value to that of '+').
    // Pre-condition: 'a[i]' is the character to be evaluated.
    // Post-condition: If 'a[i]' is a digit, it is printed, followed by '+' if it's not the last digit.
    if (a[i] > 43) { // Inline: ASCII value of '+' is 43. Digits '0'-'9' have ASCII values 48-57.
      cout << a[i]; // Functional Utility: Prints the digit.
      // Block Logic: Adds a '+' sign after the digit, unless it's the last digit to be printed.
      // Pre-condition: 'a[i]' is a digit.
      // Post-condition: A '+' is printed if there are more digits to follow.
      if (i + 1 < n)
        cout << "+"; // Functional Utility: Prints the '+' separator.
    }
  }
  // Functional Utility: No explicit newline at the end in some competitive programming outputs,
  // but it's good practice. The problem statement implies this is the full output.
  return 0; // Functional Utility: Indicates successful program execution.
}
