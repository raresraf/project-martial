/**
 * @file 339A.CPP
 * @brief Solution to the "Helpful Maths" problem (Codeforces 339A).
 *
 * This program takes a string representing an addition problem with numbers
 * separated by '+' signs (e.g., "1+1+3+1"). It extracts the numbers, sorts
 * them in non-decreasing order, and then prints them back in the same
 * addition format, but with the numbers sorted.
 *
 * Algorithm: Parsing and Sorting. The numbers are extracted by iterating
 * through the string with a step of 2 (to skip '+' signs), stored in an
 * array, and then sorted.
 * Time Complexity: O(N/2 log (N/2)) which simplifies to O(N log N) due to sorting,
 * where N is the length of the input string.
 * Space Complexity: O(N) to store the extracted numbers.
 */
#include <bits/stdc++.h>
#include <iostream>

using namespace std;

/**
 * @brief Main function to sort and print numbers from an addition string.
 *
 * Reads an addition string (e.g., "1+1+3+1"), extracts the numbers,
 * sorts them in non-decreasing order, and then prints the numbers
 * in the same addition format.
 *
 * @return 0 on successful execution.
 */
int main() {
  // Declare a string 's' to store the input addition problem.
  string s;
  // Read the input string.
  cin >> s;
  // Declare a character array 'array' to store the extracted numbers.
  // Assuming a maximum string length of around 200, so a 200-element array is sufficient for numbers.
  char array[200];
  // Initialize 'count' to track the number of extracted numbers.
  int count = 0;
  /**
   * @brief Block Logic: Extracts numbers from the input string.
   *
   * Iterates through the input string, picking characters at even indices
   * (which correspond to numbers, skipping '+' signs).
   * Pre-condition: Input string 's' contains numbers separated by '+'.
   * Invariant: 'array' stores extracted numbers, and 'count' reflects their quantity.
   */
  for (int i = 0; i <= s.size() / 2; i++) {
    array[i] = s[2 * i];
    count++;
  }

  // Sort the 'array' containing the extracted numbers in non-decreasing order.
  sort(array, array + count);
  /**
   * @brief Block Logic: Prints the sorted numbers in addition format.
   *
   * Iterates through the sorted 'array', printing each number. It appends
   * a '+' sign after each number except the last one, which is followed by a newline.
   * Pre-condition: 'array' contains sorted numbers, and 'count' is their total.
   * Invariant: Numbers are printed in sorted order, separated by '+'.
   */
  for (int i = 0; i < count; i++) {
    // If it's the last number, print it followed by a newline.
    if (i == count - 1)
      cout << array[i] << endl;
    // Otherwise, print the number followed by a '+'.
    else
      cout << array[i] << "+";
  }
  return 0;
}
