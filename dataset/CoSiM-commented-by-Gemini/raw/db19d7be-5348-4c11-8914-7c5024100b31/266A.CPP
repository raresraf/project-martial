/**
 * @file 266A.CPP
 * @brief Solution for Codeforces Problem 266A - "Stones on the Table".
 * @author @tridib_2003
 *
 * @details
 * This program calculates the minimum number of stones to remove from a row
 * so that no two adjacent stones have the same color.
 *
 * Algorithm:
 * A greedy approach is used. The program iterates through the row of stones
 * and counts every time a stone is immediately followed by another of the
- * same color. Each such pair requires one stone to be removed. This simple
 * count yields the minimum number of removals required.
 *
 * Time Complexity: O(N), where N is the number of stones, because it involves a single pass.
 * Space Complexity: O(N), to store the string representing the stones.
 */

/*
 ~ Author    : @tridib_2003
 ~ Title     : 266A. Stones on the Table
 ~ Link      : https://codeforces.com/contest/266/problem/A
*/

#include <bits/stdc++.h>
using namespace std;

// Macro for long long integer type.
#define ll long long
// Macro for the value of Pi.
#define PI 3.1415926535897932384626
// Macro for a common modulo value in competitive programming.
#define MOD 1000000007
// Macro for creating a vector of integers.
#define vi vector<int>
// Macro for dynamic array allocation.
#define mk(arr, n, type) type *arr = new type[n];
// Macro for a standard for-loop.
#define FOR(a, c) for (int(a) = 0; (a) < (c); (a)++)
// Macro for a test-case loop structure.
#define w(x)                                                                   
  int x;                                                                       
  cin >> x;                                                                    
  while (x--)
// Macro for fast I/O operations in C++.
#define FIO                                                                    
  ios::sync_with_stdio(false);                                                 
  cin.tie(0);                                                                  
  cout.tie(0);

int main() {

  // Functional Utility: Decouples C++ streams from C standard streams for faster I/O.
  FIO;

  int tc = 1;
  // The code is structured for multiple test cases, but currently runs only one.
  // cin >> tc;
  while (tc--) {

    int n;
    cin >> n;

    string s;
    cin >> s;

    // 'moves' will count the minimum number of stones to remove.
    int moves = 0;

    /**
     * Block Logic: Implements the greedy strategy by iterating through the stones.
     * It compares each stone with the next one to find adjacent pairs of the same color.
     */
    for (int i = 0; i < n - 1; ++i) {
      // Pre-condition: If a stone and its right neighbor are the same color,
      // one must be removed.
      if (s[i] == s[i + 1])
        ++moves;
    }

    // Output the total number of removals required.
    cout << moves << '
';
  }

  return 0;
}
