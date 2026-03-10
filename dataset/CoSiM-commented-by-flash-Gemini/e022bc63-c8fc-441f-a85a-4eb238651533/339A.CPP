/**
 * @file 339A.CPP
 * @brief This program rearranges a given sum of numbers (1, 2, or 3) into non-decreasing order.
 *   The input is a string representing the sum (e.g., "1+3+2+1"), and the output should be the rearranged sum (e.g., "1+1+2+3").
 *   Problem description: https://codeforces.com/problemset/problem/339/A
 * * Algorithm: Parsing, Sorting, and Formatting.
 *   1. Numbers are extracted from the input string by skipping the '+' characters.
 *   2. The extracted numbers are then sorted in non-decreasing order.
 *   3. The sorted numbers are printed, interleaved with '+' signs.
 * Time Complexity: O(L + N log N), where L is the length of the input string and N is the number of summands (approx. L/2).
 *   Effectively O(L log L) given the relationship between L and N. For L <= 100, this is very efficient.
 * Space Complexity: O(N) to store the extracted numbers in a vector.
 */

/* Xenia the beginner mathematician is a third year student at elementary
school. She is now learning the addition operation.

The teacher has written down the sum of multiple numbers. Pupils should
calculate the sum. To make the calculation easier, the sum only contains numbers
1, 2 and 3. Still, that isn't enough for Xenia. She is only beginning to count,
so she can calculate a sum only if the summands follow in non-decreasing order.
For example, she can't calculate sum 1+3+2+1 but she can calculate sums 1+1+2
and 3+3.

You've got the sum that was written on the board. Rearrange the summans and
print the sum in such a way that Xenia can calculate the sum.

Input
The first line contains a non-empty string s   the sum Xenia needs to count.
String s contains no spaces. It only contains digits and characters "+".
Besides, string s is at most
100 characters long.

Output
Print the new sum that Xenia can count. */

#include <algorithm> // Required for std::sort.
#include <iostream>  // Required for std::cin, std::cout, std::endl.
#include <string>    // Required for std::string.
#include <vector>    // Required for std::vector.

using namespace std;

/**
 * @brief Main function to rearrange and print the sum in non-decreasing order.
 * Functional Utility: Reads the input sum string, extracts the numbers, sorts them,
 *                     and then prints the numbers with '+' signs in between.
 */
int main() {
  // Functional Utility: Optimizes C++ standard I/O streams for faster execution.
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);

  string s;
  vector<int> v; // Vector to store the numbers extracted from the sum string.
  // Block Logic: Reads the input sum string.
  cin >> s;
  /**
   * Block Logic: Parses the input string to extract the individual summands.
   * Pre-condition: The input string 's' contains digits and '+' signs, with digits at even indices.
   * Invariant: In each iteration, a digit (number) is extracted from an even index of 's' and added to vector 'v'.
   * Inline: `s[i] - 48` converts the character digit (e.g., '1') to its integer value (e.g., 1) by subtracting the ASCII value of '0' (which is 48).
   */
  for (int i = 0; i < s.length(); i += 2)
    v.push_back(s[i] - 48); // Extracts the digit (number) and converts to int.
  // Block Logic: Sorts the extracted numbers in non-decreasing (ascending) order.
  sort(v.begin(), v.end());
  /**
   * Block Logic: Formats and prints the sorted numbers, re-inserting '+' signs.
   * Pre-condition: Vector 'v' contains the sorted numbers.
   * Invariant: Each number from 'v' is printed, with '+' prepended if it's not the first number.
   */
  for (vector<int>::iterator it = v.begin(); it != v.end(); it++) {
    if (it == v.begin()) // Checks if it's the first element.
      cout << *it;       // Prints the first number without a leading '+'.
    else
      cout << "+" << *it; // Prints subsequent numbers with a leading '+'.
  }
  // Prints a newline character at the end of the output.
  cout << endl;
  return 0;
}