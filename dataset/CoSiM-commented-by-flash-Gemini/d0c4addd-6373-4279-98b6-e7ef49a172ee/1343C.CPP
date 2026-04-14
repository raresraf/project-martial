/**
 * @file 1343C.CPP
 * @brief This program solves a problem that involves processing an array of integers
 *        to find the sum of maximum absolute values within contiguous subsequences
 *        of numbers that share the same sign (all positive or all negative).
 *
 * Algorithm:
 * 1. Read the number of test cases (t) and for each test case:
 * 2. Read the array size (n) and the array elements.
 * 3. Perform a forward pass: Iterate through the array. If consecutive elements `a[i]` and `a[i-1]`
 *    have the same sign, and `a[i-1]` has a larger absolute value than `a[i]`, update `a[i]` to `a[i-1]`.
 *    This effectively propagates the maximum absolute value forward within a same-signed block.
 * 4. Perform a backward pass: Iterate through the array from right to left. If consecutive elements
 *    `a[i]` and `a[i+1]` have the same sign, and `a[i+1]` has a larger absolute value than `a[i]`,
 *    update `a[i]` to `a[i+1]`. This ensures the maximum absolute value is propagated backward as well,
 *    so each element in a same-signed block eventually becomes the maximum absolute value of that block.
 * 5. Calculate the final sum: Iterate through the modified array. Add `a[i]` to the sum if it's the
 *    first element, or if `a[i]` is different from `a[i-1]`. This sums up the representative (maximum absolute)
 *    value for each distinct same-signed contiguous block.
 *
 * Time Complexity: O(N) per test case, where N is the size of the array, due to three linear passes.
 * Space Complexity: O(N) for storing the input array.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, common in competitive programming.
using namespace std;     // Uses the standard namespace to avoid prefixing std::.

// Macros for competitive programming convenience.
#define F first
#define S second
#define PB push_back
#define MP make_pair
#define REP(i, a, b) for (int i = a; i < b; i++) // Custom loop macro.
typedef long long ll;                           // Type alias for long long.
#define int ll                                  // Redefines int as long long for convenience.
typedef vector<int> vi;                         // Type alias for vector of ints (long longs).
typedef vector<long long> vl;                   // Type alias for vector of long longs.
typedef pair<int, int> pi;                      // Type alias for pair of ints (long longs).
#define trace(x) cout << #x << "=" << x << "\n"; // Debug macro to print variable name and value.
#define sz(x) (int)(x.size())                  // Returns the size of a container.
#define mod 1000000007                         // Modulo constant, often used in problems involving large numbers.
#define endl "\n"                              // Faster newline output.

// Overloaded operator<< for convenient printing of pairs, vectors, sets, and maps.
template <typename F, typename S>
ostream &operator<<(ostream &os, const pair<F, S> &p) {
  return os << "(" << p.first << ", " << p.second << ")";
}

template <typename T> ostream &operator<<(ostream &os, const vector<T> &v) {
  os << "{";
  typename vector<T>::const_iterator it;
  for (it = v.begin(); it != v.end(); it++) {
    if (it != v.begin())
      os << ", ";
    os << *it;
  }
  return os << "}";
}

template <typename T> ostream &operator<<(ostream &os, const set<T> &v) {
  os << "[";
  typename set<T>::const_iterator it;
  for (it = v.begin(); it != v.end(); it++) {
    if (it != v.begin())
      os << ", ";
    os << *it;
  }
  return os << "]";
}

template <typename F, typename S>
ostream &operator<<(ostream &os, const map<F, S> &v) {
  os << "[";
  typename map<F, S>::const_iterator it;
  for (it = v.begin(); it != v.end(); it++) {
    if (it != v.begin())
      os << ", ";
    os << it->first << " = " << it->second;
  }
  return os << "]";
}

#define deb(x) cerr << #x << " = " << x << endl; // Debug macro for error stream output.

int32_t main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output, crucial in competitive programming.
  ios::sync_with_stdio(false);
  int t;
  // Reads the number of test cases.
  cin >> t;
  vi a;
  // set<int>S;
  // Block Logic: Iterates through each test case.
  // Invariant: At the beginning of each iteration, 't' represents the remaining test cases.
  while (t--) {
    // chck for n=1
    int n;
    // Reads the size of the array for the current test case.
    cin >> n;
    a.clear();
    a.resize(n);
    // Block Logic: Reads 'n' integers into the vector 'a'.
    REP(i, 0, n) { cin >> a[i]; }

    /**
     * Block Logic: Forward pass to propagate the maximum absolute value within contiguous sequences of same-signed numbers.
     * This loop ensures that for any segment of numbers with the same sign, if an element `a[i-1]` has a
     * larger absolute value than `a[i]`, `a[i]` is updated to `a[i-1]`.
     * Pre-condition: Array 'a' contains the input integers.
     * Invariant: After each iteration, `a[i]` is updated to hold the maximum absolute value
     * encountered so far in its same-signed contiguous segment, considering elements from the left.
     */
    for (int i = 1; i < n; i++) {
      if (a[i] < 0 && a[i - 1] < 0) { // Both are negative.
        if (a[i - 1] > a[i]) {        // If previous negative is "less negative" (larger absolute value).
          a[i] = a[i - 1];            // Propagate the larger absolute value.
        }
      }
      if (a[i] > 0 && a[i - 1] > 0) { // Both are positive.
        if (a[i - 1] > a[i]) {        // If previous positive is larger.
          a[i] = a[i - 1];            // Propagate the larger absolute value.
        }
      }
    }

    /**
     * Block Logic: Backward pass to propagate the maximum absolute value within contiguous sequences of same-signed numbers.
     * This loop is similar to the forward pass but processes the array from right to left,
     * ensuring that the maximum absolute value is correctly established throughout each same-signed segment.
     * Pre-condition: Array 'a' has been partially processed by the forward pass.
     * Invariant: After each iteration, `a[i]` is updated to hold the maximum absolute value
     * encountered so far in its same-signed contiguous segment, considering elements from the right.
     * After this loop, each element in a same-signed block will be the maximum absolute value of that block.
     */
    for (int i = n - 2; i >= 0; i--) {
      if (a[i] < 0 && a[i + 1] < 0) { // Both are negative.
        if (a[i + 1] > a[i]) {        // If next negative is "less negative" (larger absolute value).
          a[i] = a[i + 1];            // Propagate the larger absolute value.
        }
      }
      if (a[i] > 0 && a[i + 1] > 0) { // Both are positive.
        if (a[i + 1] > a[i]) {        // If next positive is larger.
          a[i] = a[i + 1];            // Propagate the larger absolute value.
        }
      }
    }
    // for(int i=0;i<n;i++){
    //     S.insert(a[i]);
    // }if(i==0)
    int sum = 0;
    /**
     * Block Logic: Calculates the final sum by adding the unique propagated values.
     * It iterates through the array, adding an element `a[i]` to `sum` only if it's the
     * first element or if `a[i]` is different from `a[i-1]`. This effectively sums
     * the single representative (maximum absolute) value for each same-signed contiguous block.
     * Pre-condition: Array 'a' has been fully processed by both forward and backward passes.
     * Invariant: `sum` accurately accumulates the total based on the problem's criteria,
     * considering only the distinct maximum values from contiguous same-signed segments.
     */
    for (int i = 0; i < n; i++) {
      if (i == 0) {
        sum += a[i];
      } else {
        if (a[i] != a[i - 1]) {
          sum += a[i];
        }
      }
    }
    // Prints the calculated sum, followed by a newline.
    cout << sum << "\n";
  }

  // Standard program exit.
  return 0;
}