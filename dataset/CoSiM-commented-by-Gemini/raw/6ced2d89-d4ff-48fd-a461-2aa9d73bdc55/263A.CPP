/**
 * @file 263A.CPP
 * @brief Solution for the "Beautiful Matrix" problem.
 * @details This program solves a problem where a 5x5 matrix contains a single '1' and all other elements are '0'.
 * The goal is to find the minimum number of moves to place the '1' at the center of the matrix (row 3, column 3).
 * A move consists of swapping two adjacent rows or two adjacent columns.
 *
 * The problem can be modeled as finding the Manhattan distance between the current position of the '1'
 * and the target center position. The Manhattan distance |x1 - x2| + |y1 - y2| directly corresponds to the
 * minimum number of single row/column swaps required.
 *
 * Algorithm:
 * 1. Read the 5x5 matrix element by element.
 * 2. Identify the coordinates (row y, column x) of the '1'.
 * 3. Calculate the Manhattan distance to the center (3, 3).
 * 4. The result is |x - 3| + |y - 3|.
 *
 * Time Complexity: O(1), since the matrix size is fixed at 5x5 (25 elements).
 * Space Complexity: O(1), as it only stores a few variables for the coordinates.
 *
 * Problem Tags: ['implementation']
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
#pragma comment(linker, "/STACK:16777216")
using namespace std;

//- Architectural Intent: The following section contains a large number of pre-processor macros,
//- typedefs, and template functions. These are common in competitive programming to reduce
//- boilerplate code and accelerate development, but are not specific to the logic of this problem.
typedef long long ll;
typedef long double ld;
typedef double dbl;
typedef float flt;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef pair<double, double> pdd;
template <class F, class T> T convert(F a, int p = -1) {
  stringstream ss;
  if (p >= 0)
    ss << fixed << setprecision(p);
  ss << a;
  T r;
  ss >> r;
  return r;
}
template <class T> void print(T a, int p = -1) {
  if (p >= 0)
    cout << fixed << setprecision(p);
  cout << a;
}
template <class T> T gcd(T a, T b) {
  T r;
  while (b != 0) {
    r = a % b;
    a = b;
    b = r;
  }
  return a;
}
template <class T> T lcm(T a, T b) { return a / gcd(a, b) * b; }
const int days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
const int dx[] = {-1, 0, +1, 0};
const int dy[] = {0, +1, 0, -1};
const int inf = (int)1e9 + 5;
const ll linf = (ll)1e16 + 5;
const double eps = 1e-9;
const double pi = 3.141592654;
#define vec vector
#define un unsigned
#define pb push_back
#define mp make_pair
#define IT iterator
#define X first
#define Y second
#define debug(x)                                                               
  cerr << #x << " = " << x << " (line: " << __LINE__ << ")" << endl
#define sqr(a) ((a) * (a))
#define sz(x) ((int)x.size())
#define ms(x, a) memset(x, a, sizeof x)
#define test()                                                                 
  cerr << "Test!"                                                              
       << " (line: " << __LINE__ << ")" << endl
#define For(i, a, b) for (typeof(b) i = (a); i <= (b); i++)
#define Ford(i, a, b) for (typeof(a) i = (a); i >= (b); i--)
#define Fit(i, v) for (typeof((v).begin()) i = (v).begin(); i != (v).end(); i++)
#define Fitd(i, v)                                                             
  for (typeof((v).rbegin()) i = (v).rbegin(); i != (v).rend(); i--)
#define all(n) (n).begin(), (n).end()
#define wln(x) cout << x << endl
#define w(x) cout << x
#define r(x, type)                                                             
  ;                                                                            
  type x;                                                                      
  cin >> x;
#define rs(x)                                                                  
  ;                                                                            
  string x;                                                                    
  cin.ignore();                                                                
  getline(cin, x);
#define rarr(array, i, n)                                                      
  ;                                                                            
  for (int i = 0; i <= n; i++)                                                 
    cin >> array[i];
#define warr(array, i, n)                                                      
  ;                                                                            
  for (int i = 0; i <= n; i++)                                                 
    cout << array[i] << " ";
#define warrd(array, i, n)                                                     
  ;                                                                            
  for (int i = n; i >= 0; i--)                                                 
    cout << array[i] << " ";
#define maxn 100010

int main() {
  /**
   * Functional Utility: This is a common optimization in competitive programming.
   * It unties C++'s iostreams from C's stdio, which can significantly speed up
   * input/output operations.
   */
  ios::sync_with_stdio(false);
  int x, y;

  /**
   * Block Logic: Iterate through the 5x5 matrix to find the location of the '1'.
   * The problem guarantees exactly one '1' in the matrix.
   * Invariant: The loops scan the matrix row by row, column by column.
   */
  For(i, 1, 5) For(j, 1, 5) {
    int a;
    cin >> a;
    /**
     * Block Logic: When the element '1' is found, its 1-indexed coordinates (j, i)
     * are stored in variables x and y.
     */
    if (a) {
      x = j;
      y = i;
    }
  }

  /**
   * Functional Utility: Calculate the Manhattan distance from the found coordinates (x, y)
   * to the center of the matrix (3, 3). This distance represents the minimum number
   * of row and column swaps needed.
   */
  cout << abs(x - 3) + abs(y - 3) << endl;
  return 0;
}
