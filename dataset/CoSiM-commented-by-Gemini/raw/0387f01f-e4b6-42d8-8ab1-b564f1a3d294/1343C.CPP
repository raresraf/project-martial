/**
 * @file 1343C.CPP
 * @author Your name (you should change this)
 * @brief Solves a competitive programming problem related to alternating subsequences.
 *
 * This program attempts to find the longest subsegment of an array that consists of
 * alternating positive and negative numbers and, among all such subsegments of
 * maximum length, finds the one with the maximum possible sum.
 *
 * Algorithm: The code iterates through the array, identifying contiguous subsegments
 * where the signs of adjacent elements alternate. It keeps track of the longest
 * such subsegment found so far and the maximum sum associated with that length.
 *
 * Time Complexity: O(N) for each test case, where N is the size of the input array,
 * as it involves a single pass through the array.
 * Space Complexity: O(N) to store the input array for each test case.
 */
//============================================================================
// Name        : template.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <bits/stdc++.h>
using namespace std;
#define _CRT_SECURE_NO_DEPRECATE // suppress some compilation warning messages
                                 // (for VC++ users)
// Shortcuts for "common" data types in contests
typedef long long ll;
typedef vector<int> vi;
typedef pair<int, int> ii;
typedef vector<ii> vii;
typedef set<int> si;
typedef map<string, int> msi;
// To simplify repetitions/loops, Note: define your loop style and stick with
// it!
#define x first
#define y second
#define pb push_back
#define mp make_pair
#define REP(i, a, b)                                                           \
  for (int i = int(a); i <= int(b); i++) // a to b, and variable i is local!
#define forn(i, n) for (int i = 0; i < (n); i++)
#define TRvi(c, it) for (vi::iterator it = (c).begin(); it != (c).end(); it++)
#define TRvii(c, it) for (vii::iterator it = (c).begin(); it != (c).end(); it++)
#define TRmsi(c, it) for (msi::iterator it = (c).begin(); it != (c).end(); it++)
#define INF 2000000000 // 2 billion
// If you need to recall how to use memset:
#define MEMSET_INF 127 // about 2B
#define MEMSET_HALF_INF 63 // about 1B
// memset(dist, MEMSET_INF, sizeof dist); // useful to initialize shortest path
// distances memset(dp_memo, -1, sizeof dp_memo); // useful to initialize DP
// memoization table memset(arr, 0, sizeof arr); // useful to clear array of
// integers

int main() {
  // Fast I/O for competitive programming.
  ios::sync_with_stdio(false);
  cin.tie(0);
  int tc;
  cin >> tc;

  // Loop to handle multiple test cases.
  while (tc--) {
    int n;
    cin >> n;
    vi a(n);
    forn(i, n) cin >> a[i];

    int len = 0;      // Current length of the alternating subsegment.
    int maxlen = 0;   // Maximum length found so far.
    ll sm = -1e10 - 7; // Maximum sum for the longest alternating subsegment.
    ll curr = 0;     // Sum of the current alternating subsegment.
    int i = 0;        // Loop counter for iterating through the array.
    bool valid = true; // Flag to check if the current subsegment is alternating.


    // Main loop to iterate through the array and find alternating subsegments.
    while (i < n) {

      len = 0;
      valid = true;
      curr = 0;

      // Special handling for the last element if it's reached as a starting point.
      if (i == n - 1) {
        len++;
        curr += a[i];
        i++;
        if (len == maxlen)
          sm = max(sm, curr);
        else if (len > maxlen)
          sm = curr;
        continue;
      }

      // Inner loop to extend the current alternating subsegment.
      // Pre-condition: The subsegment from the start of the current scan is alternating.
      // Invariant: `valid` is true if a[j] and a[j+1] have opposite signs for all processed j.
      while (valid && i + 1 < n) {
        len++;
        curr += a[i];
        valid = (a[i] > 0 && a[i + 1] < 0) || (a[i] < 0 && a[i + 1] > 0);
        i++;
      }

      // If the inner loop terminated because of `valid` becoming false,
      // this block includes the last element of the valid sequence.
      if (valid) {
        if ((a[i - 1] > 0 && a[i] < 0) || (a[i - 1] < 0 && a[i] > 0)) {
          len += 1;
          curr += a[n - 1];
          i++;
        }
      }

      // After a subsegment is processed, update the maximum length and corresponding sum.
      if (len == maxlen) {
        sm = max(sm, curr);
      } else if (len > maxlen) {
        maxlen = len;
        sm = curr;
      }
      cout << sm << " " << len << "\n";
    }
    // Final output of the maximum sum found across all longest alternating subsegments.
    cout << sm << "\n";
  }
  return 0;
}
