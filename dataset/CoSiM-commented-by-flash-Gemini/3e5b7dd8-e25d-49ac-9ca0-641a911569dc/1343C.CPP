/**
 * @file 1343C.CPP
 * @brief Solution for competitive programming problem: "Alternating Subsequence".
 *
 * @details This program finds the maximum sum of a subsequence of numbers
 *          such that adjacent elements in the subsequence have opposite signs.
 *          Within a contiguous block of same-signed numbers in the original array,
 *          only the largest absolute value (or largest number for positive,
 *          smallest for negative) is considered for the alternating subsequence.
 *          Algorithm: Greedy approach - iterate through the array, maintaining
 *                     the largest element of the current same-sign segment,
 *                     and adding it to the result when the sign changes.
 *          Time Complexity: O(N), where N is the number of elements in the input array.
 *          Domain: Competitive Programming.
 */
// Problem Link : https://codeforces.com/problemset/problem/1343/C

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
/**
 * @brief Checks if two long long integers have opposite signs.
 *
 * @details This function uses the XOR bitwise operator to efficiently determine
 *          if two numbers have different signs. If `x ^ y` results in a negative
 *          number, it means their most significant bits (sign bits) were different.
 *
 * @param x The first long long integer.
 * @param y The second long long integer.
 * @return `true` if `x` and `y` have opposite signs, `false` otherwise.
 */
bool oppositesigns(ll x, ll y) { return ((x ^ y) < 0); }
/**
 * @brief Checks if the first long long integer is greater than the second.
 *
 * @details This is a simple comparison function, returning 1 if `x` is greater
 *          than `y`, and 0 otherwise.
 *
 * @param x The first long long integer.
 * @param y The second long long integer.
 * @return 1 if `x > y`, 0 otherwise.
 */
bool large(ll x, ll y) {
  if (x > y)
    return 1;
  return 0;
}
int main() {
  // Functional Utility: Decouples C++ streams from standard C streams and flushes
  // to optimize high-throughput I/O required for competitive programming constraints.
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  
  // Declare variables:
  // 't': number of test cases.
  // 'n': number of elements in the current array.
  // 'p': stores the current maximum/minimum value of a contiguous same-sign block.
  // 'i': loop counter.
  ll t, n, p, i;
  cin >> t; // Read the number of test cases.
  
  // Block Logic: Process each test case.
  while (t--) {
    cin >> n; // Read the number of elements for the current test case.
    ll a[n];   // Declare a variable-length array to store the input numbers.
    vector<ll> b; // Vector to store the chosen numbers for the alternating subsequence.
    
    // Block Logic: Read all numbers for the current test case into array 'a'.
    for (i = 0; i < n; i++)
      cin >> a[i];
    
    // Initialize 'p' with the first element of the array.
    p = a[0];
    
    // Block Logic: Iterate through the array to construct the alternating subsequence.
    // Invariant: 'p' holds the largest (absolute) value of the current same-sign segment.
    for (i = 0; i < n; i++) {
      // Precondition: Check if the current number 'a[i]' has the same sign as 'p'.
      if (oppositesigns(p, a[i]) == 0) { // If same signs.
        // If the current number is larger than 'p' (considering absolute value for negative numbers implicitly by 'large' func).
        if (large(a[i], p) == 1) // Note: `large` simply checks `a[i] > p`, which might not be what's intended for negative numbers (e.g., -1 > -5 is true, but -5 has a larger absolute value).
          p = a[i]; // Update 'p' to the larger value within the same-sign block.
      } else { // If opposite signs.
        b.push_back(p); // Add the accumulated largest element of the previous same-sign block to 'b'.
        p = a[i];       // Start a new same-sign block with the current number.
      }
    }
    b.push_back(p); // Add the last accumulated largest element to 'b'.
    
    ll sum = 0; // Declare 'sum' to calculate the total sum of the alternating subsequence.
    // Block Logic: Calculate the sum of elements in the constructed alternating subsequence.
    for (i = 0; i < b.size(); i++) {
      sum = sum + b[i]; // Add each element to the sum.
    }
    cout << sum << endl; // Output the total sum.
  }
  return 0; // Indicate successful program termination.
}
