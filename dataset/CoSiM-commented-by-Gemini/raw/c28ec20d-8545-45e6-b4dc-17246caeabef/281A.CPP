/**
 * @file 281A.CPP
 * @brief Solution for the "Word Capitalization" problem (Codeforces 281A).
 * @details This program takes a single word as input and capitalizes its first letter.
 * If the first letter is already capitalized, it remains unchanged.
 *
 * Algorithm: Direct Modification
 * The program reads the input word into a string, directly accesses the first
 * character (at index 0), converts it to uppercase using the `toupper` function,
 * and then prints the modified string.
 *
 * Time Complexity: O(L) where L is the length of the input word, dominated by the I/O operations.
 * The capitalization itself is O(1).
 * Space Complexity: O(L) to store the input string.
 */

#include <bits/stdc++.h>

// Unused macro for a standard for-loop.
#define FOR(i, a, n) for (int i = (int)(a); i < (int)(n); ++i)
// Unused macro for a for-loop from 0 to n-1.
#define REP(i, n) FOR(i, 0, n)
// Unused macro to get iterators to the beginning and end of a container.
#define all(a) a.begin(), a.end()
// Unused macro for vector push_back.
#define pb push_back

typedef unsigned long long llu;
typedef long long ll;
typedef long double ld;

using namespace std;

int main() {
  // 'line' will store the input word.
  string line;
  // Read the entire line of input. For this problem, it will be a single word.
  getline(cin, line);
  
  // Core Logic: Capitalize the first character of the string.
  // The toupper() function handles both lowercase letters and characters that
  // are already uppercase.
  line[0] = toupper(line[0]);
  
  // Print the modified string to standard output.
  cout << line << endl;
  return 0;
}