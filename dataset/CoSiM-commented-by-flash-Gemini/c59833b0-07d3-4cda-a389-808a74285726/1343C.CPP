/**
 * @file 1343C.CPP
 * @brief This program solves a problem that involves finding the maximum values from alternating sign segments within an array.
 *   For a given array of integers, it sums the largest values of contiguous subsegments where all numbers in a subsegment share the same sign.
 *   Problem description: "Alternating Subsequence", Codeforces Round #636 (Div. 3), Problem C.
 * * Algorithm: Iterative segment maximum summation. It iterates through the array, maintaining the maximum element within the current segment of same-signed numbers. When the sign of the current number differs from the previous segment's sign, the accumulated maximum of the previous segment is added to the total, and a new segment begins.
 * Time Complexity: O(N) per test case, where N is the size of the input array, as it involves a single pass through the array.
 * Space Complexity: O(N) for storing the input array 'a'.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.

using namespace std;

typedef long long ll;     // Functional Utility: Alias for long long, often used for larger integer values.
typedef vector<int> vi;   // Functional Utility: Alias for vector of integers.
typedef pair<int, int> pii; // Functional Utility: Alias for pair of integers.

/**
 * @brief Overloads the << operator for std::pair.
 * Functional Utility: Provides a convenient way to print std::pair objects for debugging or specific output formats.
 */
template <typename T1, typename T2>
ostream &operator<<(ostream &os, pair<T1, T2> p) {
  os << p.first << " " << p.second;
  return os;
}
/**
 * @brief Overloads the << operator for std::vector.
 * Functional Utility: Provides a convenient way to print std::vector objects for debugging or specific output formats.
 */
template <typename T> ostream &operator<<(ostream &os, vector<T> &v) {
  for (T i : v)
    os << i << ", ";
  return os;
}
/**
 * @brief Overloads the << operator for std::set.
 * Functional Utility: Provides a convenient way to print std::set objects for debugging or specific output formats.
 */
template <typename T> ostream &operator<<(ostream &os, set<T> s) {
  for (T i : s)
    os << i << ", ";
  return os;
}
/**
 * @brief Overloads the << operator for std::map.
 * Functional Utility: Provides a convenient way to print std::map objects for debugging or specific output formats.
 */
template <typename T1, typename T2>
ostream &operator<<(ostream &os, map<T1, T2> m) {
  for (pair<T1, T2> i : m)
    os << i << endl;
  return os;
}

/**
 * @brief Main function to process the array and calculate the sum of alternating segment maximums.
 * Functional Utility: Handles multiple test cases, reads array data, applies the segment maximum logic, and prints the total sum.
 */
int main() {
  int t;
  // Block Logic: Reads the number of test cases.
  cin >> t;
  /**
   * Block Logic: Iterates through each test case.
   * Pre-condition: 't' is a non-negative integer representing the number of test cases.
   * Invariant: Each iteration processes one full array according to the problem's logic.
   */
  while (t--) {
    int n;
    // Block Logic: Reads the size of the array for the current test case.
    cin >> n;
    // Declares a dynamic array (C99 feature, often supported as an extension) to store the input integers.
    int a[n];
    // Block Logic: Reads the 'n' integers into the array 'a'.
    for (int i = 0; i < n; i++) {
      cin >> a[i];
    }
    // 'm' stores the maximum value found in the current segment of same-signed numbers.
    int m = a[0];
    // 'sign' stores the sign of the current segment (true for negative, false for positive/zero based on signbit).
    // Inline: `signbit(a[0])` returns true if 'a[0]' is negative, false otherwise.
    bool sign = (bool)signbit(a[0]);
    // 'ans' accumulates the sum of maximums from each alternating segment.
    ll ans = 0;
    /**
     * Block Logic: Iterates through the array starting from the second element to identify segments and find their maximums.
     * Pre-condition: `a[0]` has been processed to initialize `m` and `sign`.
     * Invariant: `m` holds the maximum value of the current segment, and `ans` accumulates segment maximums when the sign changes.
     */
    for (int i = 1; i < n; i++) {
      // Block Logic: Checks if the sign of the current element `a[i]` is different from the `sign` of the current segment.
      // If the sign changes, it means the current segment has ended.
      // Inline: `(bool)signbit(a[i]) != sign` compares the sign of current element with the current segment's sign.
      if ((bool)signbit(a[i]) != sign) {
        ans += m;      // Adds the maximum of the completed segment to the total answer.
        m = INT_MIN;   // Resets 'm' to a very small value to find the maximum of the new segment.
        sign = (bool)signbit(a[i]); // Updates 'sign' to reflect the sign of the new segment.
      }
      // Functional Utility: Updates 'm' to be the maximum value seen so far within the current segment.
      m = max(m, a[i]);
    }
    // Block Logic: After the loop, the maximum of the very last segment needs to be added to the total.
    ans += m;
    // Block Logic: Prints the final calculated sum for the current test case.
    cout << ans << endl;
  }
  return 0;
}
