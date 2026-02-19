/**
 * @file 281A.CPP
 * @brief Solution to the "Word Capitalization" problem.
 * @details This program capitalizes the first letter of a given input word.
 * It uses the standard library function `toupper` for the conversion.
 *
 * Algorithm:
 * 1. Read the input word into a string `s`.
 * 2. Unconditionally apply the `toupper()` function to the first character of the string (`s[0]`).
 *    - `toupper()` converts a lowercase letter to uppercase.
 *    - If the character is already uppercase or not an alphabet character, it remains unchanged.
 * 3. Print the modified string.
 *
 * @note The code contains a redundant `if (islower(s[0]));` statement. This `if` has an empty body
 * and thus has no effect on the program's execution, as the `toupper` call is performed regardless.
 *
 * Time Complexity: O(L) due to reading and printing the string of length L. The capitalization itself is O(1).
 * Space Complexity: O(L) to store the input string.
 */
#include <bits/stdc++.h>

// Common competitive programming macros.
#define ll long long int
#define mod 1000000007
#define pb push_back
#define mp make_pair
#define ff first
#define ss second
#define all(v) v.begin(), v.end()

using namespace std;

int main()

{
  // Optimize C++ standard streams for faster I/O.
  ios::sync_with_stdio(false);
  cin.tie(0);

  int st = 0; // Unused variable.
  string s;
  cin >> s;

  // This `if` statement has no effect as it is followed by an empty statement ';'.
  if (islower(s[0]))
    ;

  // The core logic: capitalize the first character.
  // toupper() handles both cases where the character is already uppercase or is lowercase.
  s[0] = toupper(s[0]);

  cout << s;

  return 0;
}
