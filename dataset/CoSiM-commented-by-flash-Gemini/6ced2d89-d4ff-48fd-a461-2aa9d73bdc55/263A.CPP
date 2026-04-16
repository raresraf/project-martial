/**
 * @file 263A.CPP
 * @brief Solution to Codeforces Problem 263A: "Beautiful Matrix".
 *
 * Algorithm: The problem requires finding the minimum number of moves to bring a single '1'
 * in a 5x5 matrix to its center position (row 3, column 3, 1-indexed). A move consists of
 * swapping the '1' with an adjacent element (up, down, left, or right). The minimum number
 * of moves is equivalent to the Manhattan distance between the current position of '1'
 * and the center position (3,3). The solution iterates through the 5x5 matrix to find
 * the coordinates of '1' and then directly applies the Manhattan distance formula:
 * `|row_1 - row_center| + |col_1 - col_center|`.
 *
 * Time Complexity: O(1) as the matrix size is fixed at 5x5.
 * Space Complexity: O(1).
 * Domain: Competitive Programming, Ad-hoc, Implementation.
 *
 * Original Problem Details:
 * SubmissionId: 2936993
 * ContestId: 263
 * Index: A
 * ProblemName: Beautiful Matrix
 * ProblemTags: ['implementation']
 * ProgrammingLanguage: GNU C++
 * Verdict: OK
 */

#include <algorithm>
#include <bitset>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory.h>
#include <queue>
#include <set>
#include <sstream>
#include <vector>
// Functional Utility: Directs the linker to increase stack size, common in competitive programming
// for deep recursion or large local arrays.
#pragma comment(linker, "/STACK:16777216")
using namespace std;

typedef long long ll;
typedef long double ld;
typedef double dbl;
typedef float flt;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef pair<double, double> pdd;
/**
 * @brief Template function to convert between types, with optional precision for floating points.
 * Functional Utility: General-purpose type conversion, typically for debugging or specific I/O formatting.
 * @tparam F Source type.
 * @tparam T Target type.
 * @param a The value to convert.
 * @param p Precision for floating point numbers (-1 for default).
 * @return The converted value.
 */
template <class F, class T> T convert(F a, int p = -1) {
  stringstream ss;
  if (p >= 0)
    ss << fixed << setprecision(p);
  ss << a;
  T r;
  ss >> r;
  return r;
}
/**
 * @brief Template function to print a value with optional precision.
 * Functional Utility: General-purpose printing, typically for debugging or specific I/O formatting.
 * @tparam T Type of the value to print.
 * @param a The value to print.
 * @param p Precision for floating point numbers (-1 for default).
 */
template <class T> void print(T a, int p = -1) {
  if (p >= 0)
    cout << fixed << setprecision(p);
  cout << a;
}
/**
 * @brief Calculates the greatest common divisor (GCD) of two numbers using Euclidean algorithm.
 * Functional Utility: Basic number theory utility.
 * @tparam T Numeric type.
 * @param a First number.
 * @param b Second number.
 * @return The GCD of a and b.
 */
template <class T> T gcd(T a, T b) {
  T r;
  while (b != 0) {
    r = a % b;
    a = b;
    b = r;
  }
  return a;
}
/**
 * @brief Calculates the least common multiple (LCM) of two numbers.
 * Functional Utility: Basic number theory utility.
 * @tparam T Numeric type.
 * @param a First number.
 * @param b Second number.
 * @return The LCM of a and b.
 */
template <class T> T lcm(T a, T b) { return a / gcd(a, b) * b; }
// Constant array representing days in each month (non-leap year).
const int days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
// Constant arrays for 4-directional movement (up, right, down, left).
const int dx[] = {-1, 0, +1, 0};
const int dy[] = {0, +1, 0, -1};
// Constant for a large integer value, often used as infinity.
const int inf = (int)1e9 + 5;
// Constant for a large long long integer value, often used as infinity.
const ll linf = (ll)1e16 + 5;
// Constant for a small double value, used for floating-point comparisons.
const double eps = 1e-9;
// Constant for the mathematical constant Pi.
const double pi = 3.141592654;
// Functional Utility: Shorthand for std::vector.
#define vec vector
// Functional Utility: Shorthand for unsigned keyword.
#define un unsigned
// Functional Utility: Pushes an element to the back of a container.
#define pb push_back
// Functional Utility: Creates a pair.
#define mp make_pair
// Functional Utility: Shorthand for iterator.
#define IT iterator
// Functional Utility: Accesses the first element of a pair.
#define X first
// Functional Utility: Accesses the second element of a pair.
#define Y second
/**
 * @brief Debug macro to print variable name, value, and line number to cerr.
 * Functional Utility: Essential for debugging in competitive programming.
 * @param x The variable to debug.
 */
#define debug(x)                                                               \
  cerr << #x << " = " << x << " (line: " << __LINE__ << ")" << endl
// Functional Utility: Calculates the square of 'a'.
#define sqr(a) ((a) * (a))
// Functional Utility: Returns the size of a container 'x'.
#define sz(x) ((int)x.size())
// Functional Utility: Sets all bytes of a memory block 'x' to value 'a'.
#define ms(x, a) memset(x, a, sizeof x)
/**
 * @brief Debug macro to print a simple test message with line number to cerr.
 * Functional Utility: Marks specific points in code during debugging.
 */
#define test()                                                                 \
  cerr << "Test!"                                                              \
       << " (line: " << __LINE__ << ")" << endl
/**
 * @brief Macro for a for-loop iterating from 'a' to 'b' inclusive.
 * Functional Utility: Convenient shorthand for common loop patterns.
 * @param i Loop variable.
 * @param a Start value.
 * @param b End value.
 */
#define For(i, a, b) for (typeof(b) i = (a); i <= (b); i++)
/**
 * @brief Macro for a for-loop iterating downwards from 'a' to 'b' inclusive.
 * Functional Utility: Convenient shorthand for common loop patterns.
 * @param i Loop variable.
 * @param a Start value.
 * @param b End value.
 */
#define Ford(i, a, b) for (typeof(a) i = (a); i >= (b); i--)
/**
 * @brief Macro for iterating through a container from beginning to end.
 * Functional Utility: Simplifies range-based iteration for older C++ standards or specific iterator types.
 * @param i Iterator variable.
 * @param v Container.
 */
#define Fit(i, v) for (typeof((v).begin()) i = (v).begin(); i != (v).end(); i++)
/**
 * @brief Macro for iterating through a container from reverse beginning to reverse end.
 * Functional Utility: Simplifies reverse range-based iteration.
 * @param i Reverse iterator variable.
 * @param v Container.
 */
#define Fitd(i, v)                                                             \
  for (typeof((v).rbegin()) i = (v).rbegin(); i != (v).rend(); i--)
// Functional Utility: Returns iterators to the beginning and end of a container.
#define all(n) (n).begin(), (n).end()
// Functional Utility: Prints 'x' followed by a newline.
#define wln(x) cout << x << endl
// Functional Utility: Prints 'x'.
#define w(x) cout << x
/**
 * @brief Macro to declare a variable 'x' of 'type' and read its value from cin.
 * Functional Utility: Shorthand for input operations.
 * @param x Variable name.
 * @param type Variable type.
 */
#define r(x, type)                                                             \
  ;                                                                            \
  type x;                                                                      \
  cin >> x;
/**
 * @brief Macro to declare a string 'x', ignore the rest of the current line, and read a full line from cin.
 * Functional Utility: Shorthand for reading line-based string input.
 * @param x String variable name.
 */
#define rs(x)                                                                  \
  ;                                                                            \
  string x;                                                                    \
  cin.ignore();                                                                \
  getline(cin, x);
/**
 * @brief Macro to read elements into an array from index 'i' to 'n'.
 * Functional Utility: Shorthand for array input.
 * @param array The array.
 * @param i Loop variable.
 * @param n End index.
 */
#define rarr(array, i, n)                                                      \
  ;                                                                            \
  for (int i = 0; i <= n; i++)                                                 \
    cin >> array[i];
/**
 * @brief Macro to print elements of an array from index 'i' to 'n'.
 * Functional Utility: Shorthand for array output.
 * @param array The array.
 * @param i Loop variable.
 * @param n End index.
 */
#define warr(array, i, n)                                                      \
  ;                                                                            \
  for (int i = 0; i <= n; i++)                                                 \
    cout << array[i] << " ";
/**
 * @brief Macro to print elements of an array from index 'n' down to 'i'.
 * Functional Utility: Shorthand for reverse array output.
 * @param array The array.
 * @param i Loop variable.
 * @param n Start index (inclusive, descending).
 */
#define warrd(array, i, n)                                                     \
  ;                                                                            \
  for (int i = n; i >= 0; i--)                                                 \
    cout << array[i] << " ";
// Functional Utility: Defines a constant for maximum array size or similar limits.
#define maxn 100010

int main() {
  // Functional Utility: Disables synchronization between C-style I/O and C++ streams,
  // and unties cin from cout, leading to faster input/output operations.
  ios::sync_with_stdio(false);
  int x, y; // x for column, y for row (1-indexed).

  // Block Logic: Iterates through the 5x5 matrix (1-indexed rows and columns) to find the '1'.
  // Pre-condition: The input guarantees exactly one '1' in the 5x5 matrix.
  // Invariant: After the loops, 'x' and 'y' will hold the 1-indexed coordinates of the '1'.
  For(i, 1, 5) { // Loop for rows from 1 to 5
    For(j, 1, 5) { // Loop for columns from 1 to 5
      int a;
      cin >> a; // Read the current cell value
      // Block Logic: If the current cell contains '1', store its coordinates.
      if (a) {
        x = j; // Column of '1'
        y = i; // Row of '1'
      }
    }
  }
  // Functional Utility: Calculates and prints the Manhattan distance from the '1'
  // (at (x,y)) to the center of the 5x5 matrix (at (3,3)).
  // Manhattan Distance = |x1 - x2| + |y1 - y2|
  cout << abs(x - 3) + abs(y - 3) << endl;
  return 0;
}
