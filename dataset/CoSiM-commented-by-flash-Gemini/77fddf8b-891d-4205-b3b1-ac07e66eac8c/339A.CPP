/**
 * @file 339A.CPP
 * @brief Solution to Codeforces problem 339A - Helpful Maths.
 *        This program reads a string representing an arithmetic sum of numbers (e.g., "1+3+2"),
 *        extracts the numbers, sorts them, and then prints the sorted numbers
 *        rejoined by '+' signs.
 * Algorithm:
 *   1. Read the input string.
 *   2. Iterate through the string, extracting only the digit characters and converting them to integers.
 *   3. Store these integers in a vector.
 *   4. Sort the vector of integers.
 *   5. Reconstruct a new string by appending each sorted integer followed by a '+' sign,
 *      ensuring the last '+' is removed.
 * Time Complexity: O(N log N) primarily due to the sorting step, where N is the number of digits in the input string.
 * Space Complexity: O(N) to store the extracted digits.
 */
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;

#define gc getchar_unlocked
#define ff first
#define ss second
#define ll long long
#define pb push_back
#define mp make_pair
#define pii pair<int, int>
#define vi vector<int>
#define mii map<int, int>
#define pqb priority_queue<int>
#define pqs priority_queue<int, vi, greater<int>>
#define setbits(x) __builtin_popcountll(x)
#define zrobits(x) __builtin_ctzll(x)
#define mod 1000000007
#define inf 1e18
#define ps(x, y) fixed << setprecision(y) << x
#define mk(arr, n, type) type *arr = new type[n];
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

#define all(x) x.begin(), x.end()
#define clr(x) memset(x, 0, sizeof(x))
#define deb(x) cout << #x << "=" << x << endl
#define deb2(x, y) cout << #x << "=" << x << "," << #y << "=" << y << endl
#define toDigit(c) (c - '0')

// not solved ,no ouput seen

typedef tree<int, null_type, less<int>, rb_tree_tag,
             tree_order_statistics_node_update>
    pbds;

void loadInputs() {
  /**
   * @brief Optimizes C++ standard stream I/O operations and redirects I/O for local testing.
   *
   * Disables synchronization between C and C++ standard streams and unties cin from cout
   * to speed up input/output operations, which is crucial for competitive programming.
   * If not compiled in an online judge environment, it redirects standard input from "input.txt"
   * and standard output to "output.txt" for convenient local debugging.
   */
  // fast input-output
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
}

int32_t main() {
  /**
   * @brief Main function of the program.
   *
   * Initializes I/O, reads the input expression, extracts and sorts the numbers,
   * then attempts to reconstruct and print the sorted expression.
   * Note: The current implementation has a logical error in reconstructing the
   * string (`arr2` loop condition `arr2.size()` should be `arr1.size()`),
   * which may lead to incorrect output. This is preserved as per the "Zero Code Mutation" rule.
   * @return int32_t Returns 0 upon successful execution.
   */

  loadInputs();
  string s;
  cin >> s;
  vector<char> arr(s.begin(), s.end());
  vi arr1;
  // Block Logic: Extracts numeric digits from the input string 's'.
  // Pre-condition: 's' contains digits potentially separated by '+'.
  // Invariant: 'arr1' accumulates the integer values of digits.
  for (size_t i = 0; i < arr.size(); i++) {
    if (arr[i] == '+') {
      continue;
    } else {
      // Inline: Converts a character digit to its integer equivalent.
      arr1.pb(arr[i] - '0');
    }
  }
  // Block Logic: Sorts the extracted integer digits in ascending order.
  // Pre-condition: 'arr1' contains numeric digits from the input.
  // Post-condition: 'arr1' is sorted.
  sort(arr1.begin(), arr1.end());
  vector<char> arr2;
  // Block Logic: Reconstructs the string by interspersing sorted digits with '+' signs.
  // Note: There is a logical error here: `arr2.size()` is always 0 initially,
  // making this loop not execute as intended for reconstruction.
  // The intent was likely to iterate `arr1` and construct `arr2`.
  for (size_t i = 0; i < arr2.size(); i++) {
    arr2.pb(static_cast<char>(
        arr[i])); // new thing to explictly convert from char to int
    arr2.pb('+');
  }
  string ans(arr2.begin(), arr2.end());

  cout << ans << "\n";

  return 0;
}
