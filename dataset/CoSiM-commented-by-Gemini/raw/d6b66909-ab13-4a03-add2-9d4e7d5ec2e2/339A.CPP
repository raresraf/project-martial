/**
 * @file 339A.CPP
 * @brief Solution for Codeforces Problem 339A, "Helpful Maths".
 * @details The problem asks to rearrange a given sum of single-digit numbers (1, 2, and 3)
 * into non-decreasing order. For example, "3+2+1" should become "1+2+3".
 * This implementation extracts the numbers from the string, sorts them, and then
 * reconstructs the sorted sum string.
 *
 * @algorithm Parsing and Sorting
 * @complexity
 * Time: O(L log L), where L is the length of the input string. The complexity is
 *       dominated by the sorting step. The number of elements to sort is roughly L/2.
 * Space: O(L) to store the extracted numbers.
 */
#include <bits/stdc++.h> // Includes all standard C++ libraries
using namespace std;

int main() {
  string s1;
  int nmbr[105]; // Array to store the numbers, size 105 is safe for string length up to 100.
  int j = 0;     // Index for the 'nmbr' array.
  
  cin >> s1;
  int len = s1.length();

  /**
   * @brief Parse the input string to extract the numbers.
   * It iterates through the string and adds any character that is not a '+'
   * to the 'nmbr' array after converting it from a char to an int.
   */
  for (int i = 0; i < len; i++) {
    if (s1[i] != '+') {
      nmbr[j] = s1[i] - '0'; // Convert char digit to integer
      j++;
    }
  }

  // Sort the array of numbers in non-decreasing order.
  sort(nmbr, nmbr + j);

  /**
   * @brief Reconstruct and print the sorted sum.
   * It iterates through the sorted numbers and prints them, separated by '+'.
   */
  for (int i = 0; i < j; i++) {
    cout << nmbr[i];
    // Add a '+' after each number except the last one.
    if (i != j - 1) {
      cout << "+";
    }
  }
  
  cout << endl;
  return 0; // Indicate successful execution.
}