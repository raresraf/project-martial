/**
 * @file 1343C.CPP
 * @brief Solution to a competitive programming problem (Codeforces 1343C - "Alternating Subsequence").
 *
 * This program finds the maximum possible sum of an alternating subsequence from a given array.
 * An alternating subsequence is formed by taking elements such that the sign of consecutive
 * elements alternates (e.g., positive, negative, positive, or negative, positive, negative).
 * The greedy strategy is applied: for each contiguous block of numbers with the same sign,
 * only the element with the maximum absolute value (i.e., the largest positive number,
 * or the negative number closest to zero) is selected for the subsequence.
 *
 * Algorithm: Greedy approach with a single pass.
 * It iterates through the input array, dynamically building an alternating subsequence.
 * If the current element has the same sign as the last element chosen for the subsequence,
 * the last element is updated to retain the maximum value within that sign block.
 * If the sign changes, the current element starts a new block in the subsequence.
 *
 * Time Complexity: O(N) for each test case, where N is the number of elements in the sequence.
 * Total time complexity is O(T * N), where T is the number of test cases.
 * Space Complexity: O(N) to store the input array `arr` and the alternating subsequence `v`.
 */
/*
                    Moontasir Mahmood
          Information and Communication Engineering
                 University of Rajshahi
*/
#include <bits/stdc++.h> // Includes most standard C++ libraries, common in competitive programming.

// Competitive programming macros for shorthand operations.
#define MEM(a, b) memset(a, (b), sizeof(a)) // Functional Utility: Sets memory to a value.
#define CLR(a) memset(a, 0, sizeof(a))      // Functional Utility: Sets memory to zero.
#define MAX(a, b) ((a) > (b) ? (a) : (b))   // Functional Utility: Returns the maximum of two values.
#define MIN(a, b) ((a) < (b) ? (a) : (b))   // Functional Utility: Returns the minimum of two values.
#define ABS(X) ((X) > 0 ? (X) : (-(X)))    // Functional Utility: Returns the absolute value.
#define S(X) ((X) * (X))                    // Functional Utility: Returns the square of a value.
#define SZ(V) (int)V.size()                 // Functional Utility: Returns the size of a container.
#define FORN(i, n) for (int i = 0; i < n; i++) // Functional Utility: Loop from 0 to n-1.
#define FORAB(i, a, b) for (int i = a; i <= b; i++) // Functional Utility: Loop from a to b inclusive.
#define ALL(V) V.begin(), V.end()             // Functional Utility: Returns iterators to the beginning and end of a container.
#define ALLR(V) V.rbegin(), V.rend()          // Functional Utility: Returns reverse iterators.
#define IN(A, B, C) ((B) <= (A) && (A) <= (C)) // Functional Utility: Checks if A is within range [B, C].
//#define AIN(A, B, C)        assert(IN(A, B, C)) // Assert A is in range.

#define watch2(x, y)                                                           \
  cout << (#x) << " " << (#y) << " is " << (x) << " " << (y) << endl // Debug macro to print two variables.
#define watch(x) cout << (#x) << " is " << (x) << endl // Debug macro to print one variable.
#define hitAi                                                                  \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(0);                                                                  \
  cout.tie(0);                                                                 \
  ios::sync_with_stdio(0) // Functional Utility: Optimizes C++ I/O streams for speed.

#define ll long long int // Type alias for long long integer.
#define xx first         // Functional Utility: Accesses the first element of a pair.
#define yy second        // Functional Utility: Accesses the second element of a pair.
#define pb(x) push_back(x) // Functional Utility: Appends an element to a vector.
#define PI acos(-1.0)      // Mathematical constant Pi.

#define PII pair<int, int>                         // Type alias for pair of integers.
#define PLL pair<long long int, long long int>     // Type alias for pair of long long integers.
#define VI vector<int>                             // Type alias for vector of integers.

// cout << fixed << setprecision(20) << p << endl; // Commented utility for formatted output.

/**
 * @brief Template function for calculating (p^e) % M.
 * @tparam T A numeric type.
 * @param p Base.
 * @param e Exponent.
 * @param M Modulo.
 * @return T Result of (p^e) % M.
 */
template <class T> inline T bigmod(T p, T e, T M) {
  ll ret = 1;
  for (; e > 0; e >>= 1) { // Block Logic: Binary exponentiation loop.
    if (e & 1)             // If current bit of exponent is 1.
      ret = (ret * p) % M; // Multiply result by base.
    p = (p * p) % M;       // Square the base.
  }
  return (T)ret;
}

/**
 * @brief Template function for calculating the greatest common divisor (GCD) of two numbers.
 * @tparam T A numeric type.
 * @param a First number.
 * @param b Second number.
 * @return T GCD of a and b.
 */
template <class T> inline T gcd(T a, T b) {
  if (b == 0)
    return a;
  return gcd(b, a % b); // Functional Utility: Euclidean algorithm for GCD.
}

/**
 * @brief Template function for calculating modular multiplicative inverse using Fermat's Little Theorem.
 *        Requires M to be a prime number.
 * @tparam T A numeric type.
 * @param a Number.
 * @param M Modulo (must be prime).
 * @return T Modular inverse of a modulo M.
 */
template <class T> inline T modinverse(T a, T M) { return bigmod(a, M - 2, M); }

/*----------------------Graph Moves----------------*/
// Commented out arrays for common graph traversal moves (e.g., 4-directional, 8-directional, Knight's move).
// const int fx[]={+1,-1,+0,+0};
// const int fy[]={+0,+0,+1,-1};
// const int fx[]={+0,+0,+1,-1,-1,+1,-1,+1};   // Kings Move
// const int fy[]={-1,+1,+0,+0,+1,+1,-1,-1};  // Kings Move
// const int fx[]={-2, -2, -1, -1,  1,  1,  2,  2};  // Knights Move
// const int fy[]={-1,  1, -2,  2, -2,  2, -1,  1}; // Knights Move
/*------------------------------------------------*/

/*-----------------------Bitmask------------------*/
// Commented out utility functions for bit manipulation (set, reset, check bit).
// int Set(int N,int pos){return N=N | (1<<pos);}
// int reset(int N,int pos){return N= N & ~(1<<pos);}
// bool check(int N,int pos){return (bool)(N & (1<<pos));}
/*------------------------------------------------*/

using namespace std;

// Empty function, possibly a placeholder or remnant.
void go(ll pos, ll prev, ll len) {}

/**
 * @brief Main function: Entry point of the program.
 *
 * Processes multiple test cases. For each test case, it reads a sequence of numbers
 * and computes the maximum alternating subsequence sum using a greedy strategy.
 */
int main() {
  // freopen("E:/Desktop/Programming/input.txt", "r" , stdin ); // Commented out for reading from file.

  hitAi; // Optimize I/O.
  ll n, x, testCase, m, y;

  // Pre-condition: Read the number of test cases.
  cin >> testCase;
  // Block Logic: Iterate through each test case.
  // Invariant: 't' represents the current test case index.
  FORN(t, testCase) {
    // Pre-condition: Read 'n', the length of the current sequence.
    cin >> n;
    ll arr[n + 1]; // Declare a dynamic array to store the input sequence.
    vector<ll> v;  // Vector to store the chosen elements of the alternating subsequence.
    
    // Pre-condition: Read the first element and initialize 'v' and 'sum'.
    cin >> arr[0];
    v.pb(arr[0]);
    ll sum = arr[0];
    
    // Block Logic: Iterate through the rest of the sequence from the second element.
    // Invariant: 'i' represents the current index in the input array.
    FORAB(i, 1, n - 1) {
      // Pre-condition: Read the current element.
      cin >> arr[i];
      // Block Logic: If current element `arr[i]` and last element in `v` are both negative.
      if (arr[i] < 0 && v[v.size() - 1] < 0) {
        // Remove the old (smaller absolute value) negative number from sum.
        sum -= v[v.size() - 1];
        // Update the last element in 'v' to be the negative number closest to zero (largest).
        v[v.size() - 1] = MAX(v[v.size() - 1], arr[i]);
        // Add the new (larger absolute value) negative number to sum.
        sum += v[v.size() - 1];
      }
      // Block Logic: Else if current element `arr[i]` and last element in `v` are both positive.
      else if (arr[i] > 0 && v[v.size() - 1] > 0) {
        // Remove the old (smaller) positive number from sum.
        sum -= v[v.size() - 1];
        // Update the last element in 'v' to be the largest positive number.
        v[v.size() - 1] = MAX(v[v.size() - 1], arr[i]);
        // Add the new (larger) positive number to sum.
        sum += v[v.size() - 1];
      }
      // Block Logic: If the signs are different, a new alternating block begins.
      else {
        // Add the current element to the alternating subsequence.
        v.pb(arr[i]);
        // Add the current element to the total sum.
        sum += arr[i];
      }
    }

    // Output the maximum alternating subsequence sum for the current test case.
    cout << sum << endl;
  }
}
