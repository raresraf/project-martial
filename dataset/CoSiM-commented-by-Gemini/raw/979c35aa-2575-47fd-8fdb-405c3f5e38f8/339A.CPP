/**
 * @file 339A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Helpful Maths").
 * The task is to reorder a given sum of numbers (1, 2, and 3) into non-decreasing order.
 * For example, an input of "3+2+1" should produce the output "1+2+3".
 *
 * @algorithm Counting Sort. The algorithm first counts the occurrences of each number
 * (1, 2, 3) and then reconstructs the string in sorted order.
 *
 * @time_complexity O(L), where L is the length of the input string. The process
 * involves a single pass for counting and another pass for printing.
 * @space_complexity O(L) to store the input string. The counting variables
 * use constant extra space.
 */
#include <iostream>
#include <string.h>
using namespace std;

int main() {
  string s;
  int one = 0, two = 0, three = 0;
  cin >> s;

  /**
   * @block
   * @brief Counts the occurrences of each digit (1, 2, 3).
   * This loop iterates through the input string, ignoring the '+' characters,
   * and increments the counters `one`, `two`, and `three`.
   */
  for (int i = 0; i < s.length(); i++) {
    if (s.at(i) == '1')
      one++;
    if (s.at(i) == '2')
      two++;
    if (s.at(i) == '3')
      three++;
  }

  /**
   * @block
   * @brief Prints the sorted sum based on the counts.
   * This series of loops and conditional statements reconstructs the sorted
   * sum. The logic is designed to correctly place the '+' separators between
   * the numbers.
   */
  // cout << one << two << three << endl;
  if (one != 0) {
    // Print all but the last '1', each followed by a '+'
    for (int i = 1; i < one; i++)
      cout << "1+";
    // Handle the last '1'. Print a '+' if other numbers will follow.
    if (two == 0 && three == 0)
      cout << 1;
    else
      cout << "1+";
  }
  if (two != 0) {
    // Print all but the last '2', each followed by a '+'
    for (int i = 1; i < two; i++)
      cout << "2+";
    // Handle the last '2'. Print a '+' if '3's will follow.
    if (three == 0)
      cout << 2;
    else
      cout << "2+";
  }
  if (three != 0) {
    // Print all but the last '3', each followed by a '+'
    for (int i = 1; i < three; i++)
      cout << "3+";
    // Print the final '3'.
    cout << 3;
  }

  return 0;
}
