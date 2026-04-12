/**
 * @file 281A.CPP
 * @brief This program reads a single line of text from standard input,
 * converts the first character of the line to uppercase, and then
 * prints the modified line to standard output. This is a common
 * string manipulation task often found in introductory programming
 * problems or competitive programming.
 *
 * The overall design focuses on:
 * 1. Reading an entire line using `getline`.
 * 2. Accessing and modifying the first character.
 * 3. Printing the result.
 *
 * Time Complexity: O(L), where L is the length of the input line, primarily
 * due to reading and printing the string.
 * Space Complexity: O(L) for storing the input string.
 */
#include <bits/stdc++.h>

// Macros for loops, common in competitive programming to reduce verbosity.
#define FOR(i, a, n) for (int i = (int)(a); i < (int)(n); ++i)
#define REP(i, n) FOR(i, 0, n)
#define all(a) a.begin(), a.end()
#define pb push_back

// Type definitions for common data types, often used in competitive programming
// for convenience and to handle large numbers.
typedef unsigned long long llu;
typedef long long ll;
typedef long double ld;

using namespace std;

/**
 * @brief Main function where the program execution begins.
 *
 * This function performs the core logic of reading a line,
 * capitalizing its first character, and printing it.
 *
 * @return 0 if the program executes successfully.
 */
int main() {
  string line;
  getline(cin, line); // Read the entire line of input into the 'line' string.
  line[0] = toupper(line[0]); // Convert the first character of the string to uppercase.
                              // If the string is empty, this operation might
                              // lead to undefined behavior (accessing line[0]).
                              // For competitive programming, inputs are usually
                              // guaranteed to be non-empty.
  cout << line << endl; // Print the modified line followed by a newline.
  return 0;
}

