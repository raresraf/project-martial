/**
 * @file 112A.CPP
 * @brief Solution to the "Petya and Strings" problem.
 *
 * This program performs a case-insensitive lexicographical comparison of two input strings.
 * It determines which string is lexicographically smaller, larger, or if they are equal.
 *
 * Algorithm: Case-Insensitive Lexicographical Comparison.
 * Time Complexity: O(L) where L is the length of the shorter string, as it compares character by character.
 * Space Complexity: O(1) beyond storing the input strings.
 */
#include <iostream>
#include <stdlib.h>

using namespace std;

/**
 * @brief Main function to compare two strings lexicographically (case-insensitive).
 *
 * Functional Utility: Reads two strings, normalizes them to lowercase, compares them
 * character by character, and prints -1, 1, or 0 based on the comparison result.
 */
int main() {
  string a, b;
  cin >> a >> b; // Read the two input strings.

  /**
   * Block Logic: Iterates through both strings simultaneously, comparing characters.
   * Characters are first converted to lowercase for case-insensitive comparison.
   * If a difference is found, the lexicographical order is determined, and the program exits.
   * Pre-condition: Strings 'a' and 'b' contain the input.
   * Invariant: At each iteration 'i', characters a[i] and b[i] are compared in lowercase.
   */
  for (int i = 0; i < a.length(); i++) {
    // Convert character a[i] to lowercase if it's uppercase.
    // ASCII values: 'A'-'Z' are 65-90, 'a'-'z' are 97-122. Adding 32 converts uppercase to lowercase.
    a[i] = (int(a[i] > 90) ? a[i] : char(int(a[i]) + 32));
    // Convert character b[i] to lowercase if it's uppercase.
    b[i] = (int(b[i] > 90) ? b[i] : char(int(b[i]) + 32));

    // Compare the lowercase characters.
    if (a[i] != b[i]) {
      // If characters differ, determine lexicographical order and print result.
      cout << ((a[i] < b[i]) ? -1 : 1);
      exit(0); // Terminate the program as the result is found.
    }
  }
  // If the loop completes, it means all characters were identical, so strings are equal.
  cout << 0;

  /*
   * The following commented out block is unused and irrelevant to the solution.
   * It's retained as-is to adhere to the "Zero Code Mutation" constraint.
  string a,b;
  cin>>a>>b;
  for(int i=0; i<a.length();i++)
  {

  }*/
}
