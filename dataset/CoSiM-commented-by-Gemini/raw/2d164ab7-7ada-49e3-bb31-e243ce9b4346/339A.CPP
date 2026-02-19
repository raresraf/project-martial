/**
 * @file 339A.CPP
 * @brief Solution to a programming problem requiring the sorting of a mathematical sum.
 * @details This program takes a string representing a sum of single-digit numbers (1, 2, or 3),
 * such as "3+2+1", and rearranges the numbers in non-decreasing order, outputting "1+2+3".
 *
 * Algorithm:
 * 1. Read the input string, which is a sum of numbers separated by '+'.
 * 2. Extract only the numeric digits from the string. The digits are located at even indices (0, 2, 4, ...).
 *    - Convert each digit character to an integer and store it in a `std::vector<int>`.
 * 3. Sort the vector of numbers in non-decreasing (ascending) order using `std::sort`.
 * 4. Reconstruct the final sorted string.
 *    - Iterate through the length of the original string.
 *    - For even indices, append the next number from the sorted vector (converted back to a character).
 *    - For odd indices, append the '+' character.
 * 5. Print the newly constructed, sorted sum string.
 *
 * Time Complexity: O(N log N), where N is the number of digits in the sum. The dominant operation is the sorting of the numbers vector.
 * Space Complexity: O(N), to store the extracted numbers in the vector and to build the final string.
 */
#include <bits/stdc++.h>
#include <iostream>

int main() {
  using namespace std;
  string addition;
  cin >> addition;
  vector<int> numbers;

  /**
   * Block Logic: Extract the numbers from the input string.
   * Pre-condition: 'addition' is a string like "d1+d2+...".
   * Invariant: The 'numbers' vector holds all the integer digits from the string.
   */
  for (auto i = 0; i < (int)addition.length(); i += 2)
    // Characters are at even positions. Convert char to int and add to vector.
    numbers.push_back(addition.at(i) - '0');
  
  // Sort the extracted numbers in non-decreasing order.
  sort(numbers.begin(), numbers.end());
  
  string final_str = "";
  /**
   * Block Logic: Reconstruct the sorted sum string.
   * Pre-condition: 'numbers' vector is sorted.
   * Invariant: 'final_str' is being built by alternating between sorted numbers and '+' signs.
   */
  for (auto i = 0; i < (int)addition.length(); i++) {
    if (i % 2 == 0)
      // Even indices get a number from the sorted vector.
      final_str += (numbers.at(i / 2) + '0');
    else
      // Odd indices get a '+' sign.
      final_str += '+';
  }
  
  // Print the final, rearranged sum.
  cout << final_str << endl;
  return 0;
}
