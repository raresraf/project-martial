/**
 * @file 58A.CPP
 * @brief This program solves the "Chat room" problem (Codeforces 58A).
 *   It determines if the word "hello" can be formed as a subsequence of a given input string.
 *   This means the characters 'h', 'e', 'l', 'l', 'o' must appear in the input string
 *   in that exact order, but not necessarily contiguously.
 *   Problem description: https://codeforces.com/problemset/problem/58/A
 * * Algorithm: Two-pointer linear scan.
 *   One pointer (`j`) tracks the current character needed from the target word "hello",
 *   and another pointer (`i`) scans the input string. When a match is found, the "hello"
 *   pointer advances. If the "hello" pointer successfully traverses all characters
 *   of "hello", then it is a subsequence.
 * Time Complexity: O(L) where L is the length of the input string, as both pointers
 *   iterate at most once through their respective strings.
 * Space Complexity: O(L) for storing the input string.
 */
/***

            Bismillahir Rahmanir Rahim
            Read in the name of Allah, who created you!!!
            Author : Shah Newaj Rabbi Shishir,
            Department of CSE, City University, Bangladesh.

***/

#include <bits/stdc++.h> // Includes most standard C++ libraries, including <cstdio>, <iostream>, <string>.
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
// Functional Utility: Macro for a `for` loop with `i` counter for arrays.
#define arrayloop for (i = 0; i < n; i++)
// Functional Utility: Macro for reading a string using `cin`.
#define cinstr cin >> str
// Functional Utility: Macro for reading a line into `str` using `getline`.
#define getstr getline(cin, str)
// Functional Utility: Macro for printing case numbers in competitive programming.
#define pcase pf("Case %d: ", pos)
// Functional Utility: Alias for `std::vector<int>`.
#define vi vector<int>
// Functional Utility: Alias for `std::vector<std::string>`.
#define vs vector<string>
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

char str[105]; // C-style string to store the input string. Max length 104 characters + null terminator.

/**
 * @brief Main function (aliased as `ebar_khela_hoppe`) to check if "hello" is a subsequence.
 * Functional Utility: Reads an input string, performs a two-pointer scan to find "hello" as a subsequence,
 *                     and prints "YES" or "NO" based on the result.
 */
ebar_khela_hoppe {
  /*
      Block Logic: Commented-out lines for redirecting input/output to files.
      This is common in competitive programming for local testing without modifying the submission code.
      freopen ("input.txt","r",stdin);
      freopen ("output.txt","w",stdout);
  */

  char x[6] = "hello"; // C-style string representing the target word "hello".
  int i, j = 0;        // `i` scans the input string, `j` tracks characters found from "hello".

  sf("%s", str); // Block Logic: Reads the input string into `str` using `scanf`.

  /**
   * Block Logic: Implements a two-pointer linear scan to find `x` ("hello") as a subsequence in `str`.
   * Pre-condition: `str` is populated with the input string.
   * Invariant: `i` iterates through `str`. `j` advances only when a character from `x` is found in `str` in order.
   */
  for (i = 0; str[i]; i++) // Loop continues as long as `str[i]` is not the null terminator.
    if (x[j] == str[i])    // Conditional Logic: Checks if the current character in `str` matches the `j`-th character of "hello".
      ++j;                 // If a match is found, advance the "hello" pointer.

  // Conditional Logic: Checks if all 5 characters of "hello" were found in order (i.e., `j` reached 5).
  if (j == 5)
    pf("YES"); // Prints "YES" if "hello" is a subsequence.
  else
    pf("NO");  // Prints "NO" otherwise.

  pf("\n"); // Prints a newline character.

  bair_ho; // Functional Utility: Macro to return 0, indicating successful program termination.
}
