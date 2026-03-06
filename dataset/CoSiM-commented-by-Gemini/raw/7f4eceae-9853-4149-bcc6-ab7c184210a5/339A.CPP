/**
 * @file 339A.CPP
 * @brief Sorts a given mathematical sum of numbers (e.g., "1+3+2") in place.
 *
 * This program takes a string representing a sum and rearranges the number
 * characters in non-decreasing order while keeping the '+' separators in their
 * original positions.
 *
 * @algorithm In-place Bubble Sort (or Selection Sort variant) on string characters.
 * @complexity
 * Time: O(N^2), where N is the length of the string, due to the nested loops
 *       for sorting. This is less efficient than a linear-time counting sort.
 * Space: O(1) auxiliary space, as the sorting is done in-place on the input
 *        string.
 */
#include <iostream>
#include <string>
#include <algorithm> // For std::swap, though not used in original code

// Using the standard namespace for competitive programming convenience.
using namespace std;

/**
 * @brief Main function to read the input, sort it in-place, and print the result.
 */
int main() {
  int i, j;
  char temp;
  string s;
  
  // Read the string representing the sum.
  cin >> s;
  
  // Pre-condition: 's' contains the input string (e.g., "3+1+2").
  // Block Logic: Use a nested loop structure, characteristic of O(N^2) sorting
  // algorithms like Bubble Sort or Selection Sort, to compare and swap characters.
  for (i = 0; i < s.size(); i++) {
    for (j = i + 1; j < s.size(); j++) {
      
      // Invariant: Only compare characters that are digits, not '+' symbols.
      // Block Logic: If the character at the outer loop index 'i' is greater
      // than the character at the inner loop index 'j', swap them.
      if (s[i] != '+' && s[j] != '+' && s[j] < s[i]) {
        // Manual swap of characters.
        temp = s[j];
        s[j] = s[i];
        s[i] = temp;
      }
    }
  }
  
  // After sorting, print the modified string.
  cout << s;

  return 0;
}
