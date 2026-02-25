/**
 * @file 339A.CPP
 * @brief Solution for a string sorting problem, likely from a programming contest.
 * @details The program reads a string representing a sum of single-digit numbers
 * (e.g., "3+1+2"), sorts the numbers, and prints the rearranged sum (e.g., "1+2+3").
 *
 * Algorithm:
 * 1. Read the input string which contains digits separated by '+' signs.
 * 2. Extract the digit characters from the string. The digits are located at
 *    even indices (0, 2, 4, ...).
 * 3. Store these digit characters in an array.
 * 4. Sort the array of digit characters. Since they are single digits '1'-'9',
 *    standard character sorting is equivalent to numerical sorting.
 * 5. Print the sorted digits, interleaved with '+' signs to reconstruct the sum format.
 *
 * Time Complexity: O(L log L) where L is the number of digits in the input string, dominated by the sort operation.
 * Space Complexity: O(L) to store the digit characters.
 */
#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  string s;
  cin >> s;
  char array[200];
  int count = 0;

  /**
   * @block
   * @description Extracts the digit characters from the input string.
   * The input is a sum, so digits appear at even indices (0, 2, 4, ...).
   */
  for (int i = 0; i <= s.size() / 2; i++) {
    array[i] = s[2 * i];
    count++;
  }

  /**
   * @block
   * @description Sorts the extracted digits. The std::sort function is used
   * on the character array. For single digits, character sorting yields the
   * correct numerical order.
   */
  sort(array, array + count);
  
  /**
   * @block
   * @description Reconstructs and prints the sorted sum. It iterates through
   * the sorted digits and prints them separated by '+' signs.
   */
  for (int i = 0; i < count; i++) {
    if (i == count - 1)
      cout << array[i] << endl;
    else
      cout << array[i] << "+";
  }
  return 0;
}