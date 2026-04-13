/**
 * @file 339A.CPP
 * @brief This program solves the "Helpful Maths" problem by sorting a sum string containing '1', '2', and '3' digits.
 * The input string is a sum (e.g., "3+2+1"), and the output is the sorted sum (e.g., "1+2+3").
 * Algorithm: Counting sort. It counts the occurrences of each digit ('1', '2', '3') and then reconstructs
 * the string by placing them in non-decreasing order, maintaining the '+' separators.
 * Time Complexity: O(L) where L is the length of the input string, due to two passes over the string.
 * Space Complexity: O(1) for digit counters.
 */
#include <cstring>  // Required for string functions like strlen (though not directly used with std::string here).
#include <iostream> // Required for input/output operations (cin, cout).
using namespace std;

int main() {
  string s; // Stores the input sum string (e.g., "1+3+2").
  int c1 = 0, c2 = 0, c3 = 0; // Counters for digits '1', '2', and '3' respectively.
  int i; // Loop counter.
  // Reads the sum string from standard input.
  cin >> s;
  /**
   * Block Logic: Iterates through the input string to count the occurrences of each digit ('1', '2', '3').
   * Plus signs '+' are ignored as they are not digits.
   * Invariant: 'c1', 'c2', 'c3' accurately reflect the counts of '1's, '2's, and '3's in the string.
   */
  for (i = 0; i < s.length(); i++) {
    if (s[i] == '1') {
      c1++;
    }
    if (s[i] == '2') {
      c2++;
    }
    if (s[i] == '3') {
      c3++;
    }
  }
  /**
   * Block Logic: Reconstructs the string 's' by placing the counted digits in non-decreasing order.
   * It iterates through the string, placing '1's first, then '2's, then '3's at positions where digits should be (even indices).
   * The '+' separators are naturally preserved since only digit positions are modified.
   * Invariant: 's' is progressively filled with sorted digits.
   */
  for (i = 0; i < s.length(); i += 2) {
    if (c1 > 0) {
      s[i] = '1'; // Place a '1'.
      c1--;       // Decrement '1' counter.
    } else if (c2 > 0) {
      s[i] = '2'; // Place a '2'.
      c2--;       // Decrement '2' counter.
    } else if (c3 > 0) {
      s[i] = '3'; // Place a '3'.
      c3--;       // Decrement '3' counter.
    }
  }
  // Prints the sorted sum string to standard output.
  cout << s << endl;
  // Standard program exit.
  return 0;
}
