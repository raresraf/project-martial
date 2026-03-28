/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two strings.
 *
 * This program reads two strings, converts them to lowercase, and then compares
 * them lexicographically. It prints -1 if the first string is less than the
 * second, 1 if it's greater, and 0 if they are equal.
 *
 * @b Algorithm: The program iterates through the strings character by character.
 *    In each iteration, it converts the characters from both strings to lowercase.
 *    It then compares these characters. If a difference is found, it prints the
 *    result and exits. If the loop completes without finding any difference,
 *    the strings are considered equal and 0 is printed.
 *
 * @b Time_Complexity: O(L), where L is the length of the strings, as it may
 *    iterate through the entire length of the strings.
 * @b Space_Complexity: O(L) to store the two input strings.
 */
#include <iostream>
#include <stdlib.h>

using namespace std;

int main() {
  string a, b;
  cin >> a >> b;
  
  /**
   * @brief This loop iterates through the strings, comparing them character by character.
   */
  for (int i = 0; i < a.length(); i++) {
    // Convert characters to lowercase for case-insensitive comparison.
    // ASCII values for uppercase letters are less than 97 ('a'). 'Z' is 90.
    // If the character's ASCII value is greater than 90, it is assumed to be lowercase.
    // Otherwise, 32 is added to convert it to lowercase (e.g., 'A' (65) + 32 = 'a' (97)).
    a[i] = (int(a[i] > 90) ? a[i] : char(int(a[i]) + 32));
    b[i] = (int(b[i] > 90) ? b[i] : char(int(b[i]) + 32));
    
    // If a difference is found, print the result and exit.
    if (a[i] != b[i]) {
      cout << ((a[i] < b[i]) ? -1 : 1);
      exit(0);
    }
  }
  
  // If no difference is found after the loop, the strings are equal.
  cout << 0;

  return 0;
}
