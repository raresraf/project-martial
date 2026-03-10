/**
 * @file 110A.CPP
 * @brief This program solves the "Nearly Lucky Number" problem (Codeforces 110A).
 *   A "lucky number" is one containing only the digits 4 and 7.
 *   A "nearly lucky number" is a number where the *count* of lucky digits
 *   in its decimal representation is itself a lucky number.
 *   Problem description: https://codeforces.com/problemset/problem/110/A
 * * Algorithm: Digit counting and conditional check.
 *   1. Reads the input number as a string.
 *   2. Iterates through the digits of the string, counting occurrences of '4' or '7'.
 *   3. Checks if this count is exactly 4 or 7.
 * Time Complexity: O(L) where L is the number of digits in the input number (length of string).
 * Space Complexity: O(L) for storing the input string.
 */
/***

            Bismillahir Rahmanir Rahim
            Read in the name of Allah, who created you!!!
            Author : Shah Newaj Rabbi Shishir,
            Department of CSE, City University, Bangladesh.

***/

#include <bits/stdc++.h> // Includes most standard C++ libraries, including <cstdio> and <string>.
using namespace std;

// Functional Utility: Macro to define the main function entry point.
#define ebar_khela_hoppe int main(void)
// Functional Utility: Macro for returning 0 from main, indicating successful execution.
#define bair_ho return 0
// Functional Utility: Macro for `scanf`, used for formatted input.
#define sf scanf
// Functional Utility: Macro for `printf`, used for formatted output.
#define pf printf
// Functional Utility: Macro for `sscanf`, used for scanning formatted input from a string.
#define ssf sscanf
// Functional Utility: Macro for `sprintf`, used for printing formatted output to a string.
#define spf sprintf
// Functional Utility: Macro for `fscanf`, used for scanning formatted input from a file.
#define fsf fscanf
// Functional Utility: Macro for `fprintf`, used for printing formatted output to a file.
#define fpf fprintf
// Functional Utility: Macro to optimize C++ standard I/O streams for faster execution.
#define fast ios_base::sync_with_stdio(0), cin.tie(0), cout.tie(0)
// Functional Utility: Macro for reading an integer into `tc` (test cases).
#define scase sf("%d", &tc)
// Functional Utility: Macro for reading an integer into `n`.
#define sn sf("%d", &n)
// Functional Utility: Macro for a `while` loop that iterates `tc` times.
#define whilecase while (tc--)
// Functional Utility: Macro for a `while` loop that reads until EOF.
#define eof while (cin >> n)
// Functional Utility: Macro for a `for` loop with `pos` counter.
#define forloop for (pos = 1; pos <= tc; pos++)
// Functional Utility: Macro for a `for` loop with `i` counter for arrays (incomplete/syntactically incorrect here).
#define arrayloop (i = 0; i < n; i++)
// Functional Utility: Macro for reading a string using `cin`.
#define cinstr cin >> str
// Functional Utility: Macro for reading a line into `str` using `getline`.
#define getstr getline(cin, str)
// Functional Utility: Macro for printing case numbers in competitive programming.
#define pcase pf("Case %d: ", pos)
// Functional Utility: Alias for `std::vector<int>`.
#define vi vector<int>
// Functional Utility: Alias for `std::pair<int, int>`.
#define pii pair<int, int>
// Functional Utility: Alias for `std::map<int, int>`.
#define mii map<int, int>
// Functional Utility: Alias for `push_back` method of vectors.
#define pb push_back
// Functional Utility: Alias for `insert` method of containers.
#define in insert
// Functional Utility: Alias for `unsigned long long`.
#define llu unsigned long long
// Functional Utility: Alias for `long long`.
#define lld long long
// Functional Utility: Alias for `unsigned int`.
#define U unsigned int
// Functional Utility: Macro for newline character, often used for faster output than `std::endl`.
#define endl "\n"

// Constant for modulo operation, common in number theory problems.
const int MOD = 1000000007;
// Constant for maximum size, e.g., for array bounds.
const int MAX = 1000005;

/**
 * @brief Functional Utility: Sets the `x`-th bit of an integer `n`.
 * @param n The integer whose bit is to be set.
 * @param x The 0-indexed position of the bit to set.
 * @return The integer `n` with its `x`-th bit set.
 */
int SetBit(int n, int x) { return n | (1 << x); }
/**
 * @brief Functional Utility: Clears the `x`-th bit of an integer `n`.
 * @param n The integer whose bit is to be cleared.
 * @param x The 0-indexed position of the bit to clear.
 * @return The integer `n` with its `x`-th bit cleared.
 */
int ClearBit(int n, int x) { return n & ~(1 << x); }
/**
 * @brief Functional Utility: Toggles (flips) the `x`-th bit of an integer `n`.
 * @param n The integer whose bit is to be toggled.
 * @param x The 0-indexed position of the bit to toggle.
 * @return The integer `n` with its `x`-th bit toggled.
 */
int ToggleBit(int n, int x) { return n ^ (1 << x); }
/**
 * @brief Functional Utility: Checks if the `x`-th bit of an integer `n` is set.
 * @param n The integer to check.
 * @param x The 0-indexed position of the bit to check.
 * @return True if the `x`-th bit is set, False otherwise.
 */
bool CheckBit(int n, int x) { return (bool)(n & (1 << x)); }

// C-style string to store the input number. Max length 24 characters + null terminator.
char str[25];

/**
 * @brief Main function (aliased as `ebar_khela_hoppe`) to determine if a number is "nearly lucky".
 * Functional Utility: Reads a number as a string, counts its lucky digits, and prints
 *                     "YES" if the count is a lucky number (4 or 7), "NO" otherwise.
 */
ebar_khela_hoppe {
  /*
      Block Logic: Commented-out lines for redirecting input/output to files.
      This is common in competitive programming for local testing without modifying the submission code.
      freopen ("input.txt","r",stdin);
      freopen ("output.txt","w",stdout);
  */

  int i, k = 0; // `i`: loop counter, `k`: counter for lucky digits ('4' or '7').

  // Block Logic: Reads the input number as a C-style string into `str` using `scanf`.
  sf("%s", str);

  /**
   * Block Logic: Iterates through each character (digit) of the input string.
   * Pre-condition: `str` is populated with the input number.
   * Invariant: `k` accumulates the count of lucky digits ('4' or '7').
   */
  for (i = 0; str[i]; i++) // Loop continues as long as `str[i]` is not the null terminator.
    // Conditional Logic: Checks if the current digit is '4' or '7'.
    if (str[i] == '4' || str[i] == '7')
      ++k; // Increments the lucky digit counter.

  /**
   * Conditional Logic: Determines if the count of lucky digits (`k`) is itself a lucky number.
   * If `k` is 4 or 7, the input number is "nearly lucky" and "YES" is printed.
   * Otherwise, "NO" is printed.
   */
  if (k == 4 || k == 7)
    pf("YES");
  else
    pf("NO");

  pf("\n"); // Prints a newline character.

  bair_ho; // Functional Utility: Macro to return 0, indicating successful program termination.
}
