/**
 * @8d79026b-5809-4a3c-b7ee-99df0f6f2c2e/112A.CPP
 * @brief This program compares two input strings lexicographically, ignoring case.
 * It determines which string is lexicographically greater, or if they are equal,
 * and prints 1, -1, or 0 respectively. This solves the "Petya and Strings" problem.
 * Problem Link: https://codeforces.com/problemset/problem/112/A
 * Algorithm: Case-insensitive lexicographical comparison.
 * Time Complexity: O(L), where L is the length of the shorter string, as it compares character by character.
 * Space Complexity: O(L) to store the input strings, though in-place transformation uses original string memory.
 */

#include <algorithm> // Required for std::transform
#include <cstdio>    // Required for printf for C-style output
#include <iostream>  // Required for std::cin
#include <string>    // Required for std::string

using namespace std;

int main(int argc, char *argv[]) {
  // @var a: Stores the first input string.
  // @var b: Stores the second input string.
  string a, b;
  cin >> a >> b; // Read the two input strings.
  
  // Block Logic: Convert both strings to lowercase for case-insensitive comparison.
  // Invariant: Both strings 'a' and 'b' are converted to their lowercase equivalents.
  // Pre-condition: Strings 'a' and 'b' contain the original input.
  transform(a.begin(), a.end(), a.begin(), ::tolower); // Transform 'a' to lowercase.
  transform(b.begin(), b.end(), b.begin(), ::tolower); // Transform 'b' to lowercase.
  
  // Block Logic: Compare the strings character by character lexicographically.
  // Invariant: Comparison proceeds until a difference is found or the end of a string is reached.
  // Pre-condition: Both strings are of equal length (implicitly assumed by the problem, or shorter length is boundary).
  for (int i = 0; i < a.length(); i++) {
    // @var ac: Current character from string 'a'.
    // @var bc: Current character from string 'b'.
    char ac = a.at(i);
    char bc = b.at(i);
    
    // Block Logic: If a difference is found between characters.
    if (ac != bc) {
      // Block Logic: Determine which character is lexicographically greater.
      if (ac > bc) {
        printf("1
"); // Print 1 if string 'a' is greater.
      } else {
        printf("-1
"); // Print -1 if string 'b' is greater.
      }
      return 0; // Terminate the program after printing the result.
    }
  }
  
  // Post-condition: If the loop completes, both strings are identical.
  printf("0
"); // Print 0 if the strings are equal.
  return 0; // Explicitly return 0 for successful execution.
}
