/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two strings.
 * @details This program solves the "Petya and Strings" problem from Codeforces.
 * It compares two strings lexicographically, treating uppercase and lowercase
 * letters as equivalent. It outputs -1 if the first string is smaller, 1 if the
 * second is smaller, and 0 if they are equal.
 *
 * @see https://codeforces.com/problemset/problem/112/A
 *
 * Algorithm:
 * The program uses an unconventional but functionally correct method.
 * 1. Two strings are read from input.
 * 2. It iterates through the strings, character by character.
 * 3. In each iteration, it converts the characters at the current position to lowercase.
 * 4. It maintains two running sums of the ASCII values of the characters processed so far
 *    for each string.
 * 5. If the running sums differ at any point, the comparison is decided, the result
 *    is printed, and the program terminates. This works because the first character
 *    difference will create a difference in the sums, and the direction of that
 *    difference determines the lexicographical order.
 * 6. If the loop completes with the sums always being equal, the strings are identical.
 *
 * Time Complexity: O(L), where L is the length of the shorter string, as the loop
 * terminates at the first differing character or the end of the strings.
 * Space Complexity: O(L) to store the two input strings.
 */
#include <iostream>
#include <string>  // For std::string
#include <cctype>  // For tolower

// Macro to optimize C++ I/O streams for competitive programming.
#define optimizar_io                                                           \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);

using namespace std;
int main() {
  optimizar_io;
  
  // 'cont' and 'conti' are running sums of ASCII values for strings 'a' and 'b'.
  long long cont = 0, conti = 0;
  string a, b;
  cin >> a >> b;

  /**
   * @brief Main comparison loop.
   *
   * Pre-condition: Strings 'a' and 'b' have the same length.
   * Logic: The loop compares the running sums of character values. It exits as soon
   * as a difference is found, which is equivalent to finding the first differing
   * character in a standard lexicographical comparison.
   */
  for (int i = 0; i < (int)a.size(); ++i) {
    // Convert characters to lowercase for case-insensitive comparison.
    char char_a = tolower(a[i]);
    char char_b = tolower(b[i]);
    
    // Check for difference directly (a more standard approach).
    if (char_a < char_b) {
      cout << "-1\n";
      return 0;
    }
    if (char_a > char_b) {
      cout << "1\n";
      return 0;
    }
  }

  // If the loop completes without finding any differences, the strings are equal.
  cout << "0\n";
  return 0;
}
