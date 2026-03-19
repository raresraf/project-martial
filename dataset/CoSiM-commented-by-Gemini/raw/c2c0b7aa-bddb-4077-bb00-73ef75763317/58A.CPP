/*** 

            Bismillahir Rahmanir Rahim
            Read in the name of Allah, who created you!!!
            Author : Shah Newaj Rabbi Shishir,
            Department of CSE, City University, Bangladesh.

***/
/**
 * @file 58A.CPP
 * @brief Solution for the "Chat room" problem (Codeforces 58A).
 * @details This program checks if a given string contains the word "hello" as a subsequence.
 * The characters 'h', 'e', 'l', 'l', 'o' must appear in that order, but not
 * necessarily contiguously.
 *
 * Algorithm: Greedy Two-Pointer Subsequence Search
 * The code uses a two-pointer approach. One pointer ('i') iterates through the input
 * string, and another pointer ('j') tracks the current character in the target
 * subsequence "hello". When a character in the input string matches the character
 * pointed to by 'j', 'j' is advanced. If 'j' successfully traverses the entire
 * "hello" string, the subsequence is present.
 *
 * NOTE: The code makes heavy use of C-style macros, including defining `main` as
 * `ebar_khela_hoppe` and `return 0` as `bair_ho`, which can harm readability.
 *
 * Time Complexity: O(N), where N is the length of the input string, due to a single pass.
 * Space Complexity: O(N) for the input string buffer.
 */

#include <bits/stdc++.h>
using namespace std;

// Custom macros for common operations and keywords, a C++ competitive programming practice.
#define ebar_khela_hoppe int main(void) // Defines the main function.
#define bair_ho return 0             // Defines the return statement for main.
#define sf scanf
#define pf printf
#define ssf sscanf
#define spf sprintf
#define fsf fscanf
#define fpf fprintf
#define fast ios_base::sync_with_stdio(0), cin.tie(0), cout.tie(0)
#define scase sf("%d", &tc)
#define sn sf("%d", &n)
#define whilecase while (tc--)
#define eof while (cin >> n)
#define forloop for (pos = 1; pos <= tc; pos++)
#define arrayloop for (i = 0; i < n; i++)
#define cinstr cin >> str
#define getstr getline(cin, str)
#define pcase pf("Case %d: ", pos)
#define vi vector<int>
#define vs vector<string>
#define pii pair<int, int>
#define mii map<int, int>
#define pb push_back
#define in insert
#define llu unsigned long long
#define lld long long
#define U unsigned int
#define endl "\n"

const int MOD = 1000000007;
const int MAX = 1000005;

// Unused helper functions for bit manipulation.
int SetBit(int n, int x) { return n | (1 << x); }
int ClearBit(int n, int x) { return n & ~(1 << x); }
int ToggleBit(int n, int x) { return n ^ (1 << x); }
bool CheckBit(int n, int x) { return (bool)(n & (1 << x)); }

// Input string buffer.
char str[105];

// The main function, defined by a macro.
ebar_khela_hoppe {
  /*
      freopen ("input.txt","r",stdin);
      freopen ("output.txt","w",stdout);
  */

  // x: The target subsequence "hello".
  // i: Pointer for iterating through the input string `str`.
  // j: Pointer for tracking progress through the target string `x`.
  char x[6] = "hello";
  int i, j = 0;

  // Read the input string using C-style scanf.
  sf("%s", str);

  /**
   * Block Logic: Greedily search for the "hello" subsequence.
   * Pre-condition: 'str' contains the input, 'j' is 0.
   * Invariant: 'j' holds the length of the prefix of "hello" found so far.
   */
  for (i = 0; str[i]; i++)
    // If the current character in the input matches the current target character...
    if (x[j] == str[i])
      // ...advance the target pointer.
      ++j;

  // After checking the whole string, if 'j' is 5, all characters of "hello" were found in order.
  if (j == 5)
    pf("YES");
  else
    pf("NO");

  pf("\n");

  // Return 0, defined by a macro.
  bair_ho;
}