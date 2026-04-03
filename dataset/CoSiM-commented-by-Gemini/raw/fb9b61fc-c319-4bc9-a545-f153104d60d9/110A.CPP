/**
 * @file 110A.CPP
 * @brief Solves the "Nearly Lucky Number" problem from Codeforces (Problem 110A).
 * @see https://codeforces.com/problemset/problem/110/A
 *
 * @description
 * A "lucky number" is a positive integer whose decimal representation contains only the digits 4 and 7.
 * A "nearly lucky number" is a number where the count of lucky digits ('4' and '7') within it is a lucky number.
 * This program determines if a given number is a "nearly lucky number".
 *
 * @algorithm
 * 1. Read the input number as a string.
 * 2. Initialize a counter to zero.
 * 3. Iterate through each character of the string.
 * 4. If a character is '4' or '7', increment the counter.
 * 5. After the loop, the counter holds the total number of lucky digits in the input.
 * 6. Check if the counter's value is itself a lucky number (i.e., 4 or 7).
 * 7. If the count is 4 or 7, the original number is "nearly lucky", so print "YES".
 * 8. Otherwise, print "NO".
 *
 * @time_complexity O(L), where L is the number of digits in the input number,
 *                  due to the single pass required to count the lucky digits.
 * @space_complexity O(L) to store the input number as a string.
 */

/***

            Bismillahir Rahmanir Rahim
            Read in the name of Allah, who created you!!!
            Author : Shah Newaj Rabbi Shishir,
            Department of CSE, City University, Bangladesh.

***/

#include <bits/stdc++.h>
using namespace std;

//==================================================================================
// Section: User-defined macros for competitive programming
//----------------------------------------------------------------------------------
// These are personalized shorthands used to speed up coding in competitive contests.
//==================================================================================
#define ebar_khela_hoppe int main(void)
#define bair_ho return 0
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
#define arrayloop (i = 0; i < n; i++)
#define cinstr cin >> str
#define getstr getline(cin, str)
#define pcase pf("Case %d: ", pos)
#define vi vector<int>
#define pii pair<int, int>
#define mii map<int, int>
#define pb push_back
#define in insert
#define llu unsigned long long
#define lld long long
#define U unsigned int
#define endl "
"

const int MOD = 1000000007;
const int MAX = 1000005;

// Bit manipulation helper functions
int SetBit(int n, int x) { return n | (1 << x); }
int ClearBit(int n, int x) { return n & ~(1 << x); }
int ToggleBit(int n, int x) { return n ^ (1 << x); }
bool CheckBit(int n, int x) { return (bool)(n & (1 << x)); }

// Character array to store the input number as a string.
char str[25];

// Main function, aliased with a custom macro.
ebar_khela_hoppe {
  /*
      // These lines are typically used for redirecting I/O to files for local testing.
      freopen ("input.txt","r",stdin);
      freopen ("output.txt","w",stdout);
  */

  int i;
  // k: counts the number of lucky digits ('4' or '7').
  int k = 0;

  // Read the number as a string.
  sf("%s", str);

  /**
   * @brief Block Logic: Count the occurrences of lucky digits.
   * This loop iterates through the string representation of the number.
   */
  for (i = 0; str[i]; i++)
    if (str[i] == '4' || str[i] == '7')
      ++k;

  /**
   * @brief Block Logic: Check if the count of lucky digits is itself a lucky number.
   * According to the problem, the number of lucky digits must be 4 or 7.
   */
  if (k == 4 || k == 7)
    pf("YES");
  else
    pf("NO");

  pf("
");

  bair_ho; // Macro for "return 0;"
}
