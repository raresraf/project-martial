/*
Hey there stalker....











Like what you see?
*/

/**
 * @file 236A.CPP
 * @brief Solution for Codeforces Problem 236A, "Boy or Girl".
 * @details This program determines if the number of distinct characters in a user's
 * name is even or odd. If the count is even, it prints "CHAT WITH HER!"; otherwise,
 * it prints "IGNORE HIM!". This implementation finds the number of distinct characters
 * by counting the number of non-unique characters and subtracting this from the
 * total length of the string.
 *
 * @algorithm Brute-force counting of duplicate characters using nested loops.
 * @time_complexity O(L^2), where L is the length of the input string, due to the
 * nested loops used to find duplicate characters.
 * @space_complexity O(L) to store the input string.
 */
#include <algorithm>
#include <bits/stdc++.h>
#include <set>
#include <string>
#include <vector>
using namespace std;
// Macro for optimizing C++ I/O streams.
#define IOS                                                                    \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);
// Convenience macros for common operations.
#define pb push_back
#define pii pair<ll, ll>
#define vi vector<ll>
#define all(a) (a).begin(), (a).end()
#define F first
#define S second
#define sz(x) (ll) x.size()
#define rep(i, a, b) for (ll i = a; i < b; i++)
#define repr(i, a, b) for (ll i = a; i > b; i--)
#define lbnd lower_bound
#define ubnd upper_bound
#define bs binary_search
#define mp make_pair
// Define 'int' as 'long long' to avoid overflow issues in other problems.
#define ll long long int
#define lld long double
#define endl "\n"
#define MOD 1000000007

int main() {
  IOS

  // A common template for multiple test cases, hardcoded to 1 for this problem.
  int TESTS = 1;
  // cin >> TESTS;
  while (TESTS--) {
    string s;
    cin >> s;
    ll count = 0; // Counter for non-first occurrences of characters.

    /**
     * Block Logic: A brute-force O(L^2) approach to count duplicate characters.
     * For each character, it checks if it has appeared before in the string.
     */
    rep(i, 0, sz(s)) {
      rep(j, 0, i) {
        if (s[i] == s[j]) {
          count++; // Increment if a character is a duplicate of a previous one.
          break;   // Break to avoid counting multiple duplicates of the same char.
        }
      }
    }
    
    // Calculate the number of distinct characters.
    ll d = sz((s)) - count;
    
    /**
     * Block Logic: Determine the output based on the parity of the distinct
     * character count.
     */
    if (d % 2 == 0) {
      cout << "CHAT WITH HER!" << endl;
    } else {
      cout << "IGNORE HIM!" << endl;
    }
  }
  return 0;
}