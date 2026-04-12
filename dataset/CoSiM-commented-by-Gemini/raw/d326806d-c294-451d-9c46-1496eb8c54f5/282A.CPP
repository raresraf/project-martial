/**
 * @file 282A.CPP
 * @brief Solution for Codeforces Problem 282A - "Bit++".
 * @author The solution is submitted by a competitive programmer.
 *
 * @section Algorithm
 * The problem asks to calculate the final value of a variable 'X' which starts at 0 and is
 * modified by a series of statements. The statements are either '++X', 'X++', '--X', or 'X--'.
 *
 * The algorithm simulates this process directly:
 * 1. Initialize a counter variable, 'x', to 0.
 * 2. Read the number of statements, 'n'.
 * 3. Loop 'n' times:
 *    a. Read the statement string.
 *    b. Check if the string contains a minus sign ('-'). This is a shortcut to identify
 *       decrement operations ('--X' or 'X--').
 *    c. If a minus sign is present, decrement 'x'.
 *    d. Otherwise, it must be an increment operation, so increment 'x'.
 * 4. Print the final value of 'x'.
 *
 * @section Complexity
 * - Time Complexity: O(N), where N is the number of statements, as we process each one once.
 * - Space Complexity: O(1), as we only store a few variables regardless of the input size.
 *
 * @note This file contains a large amount of boilerplate code and macros common in
 * competitive programming, most of which are not used for this specific problem.
 */

/*
        SubmissionId	:	3300161
        ContestId	:	282
        Index	:	A
        ProblemName	:	Bit++
        ProblemTags	:	['implementation']
        ProgrammingLanguage	:	GNU C++
        Verdict	:	OK
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
#include <list>
#include <map>
#include <memory.h>
#include <queue>
#include <set>
#include <sstream>
#include <vector>
#pragma comment(linker, "/STACK:16777216")
using namespace std;

// --- Boilerplate and Unused Code Section ---
typedef long long ll;
typedef unsigned long long ull;
typedef long double ld;
typedef double dbl;
typedef float flt;
typedef string str;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef pair<double, double> pdd;
template <class F, class T> T convert(F a, int p = -1) { /* ... */ }
template <class T> void print(T a, int p = -1) { /* ... */ }
template <class T> bool isPrime(T a) { /* ... */ }
template <class T> T gcd(T a, T b) { /* ... */ }
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
#define debug(x) cerr << #x << " = " << x << " #" << __LINE__ << '
'
#define sqr(a) ((a) * (a))
#define sz(x) ((int)(x).size())
#define first(x) (*((x).begin()))
#define last(x) (*((x).end() - 1))
#define ms(x, a) memset(x, a, sizeof x)
#define test() cerr << "Test!" << " #" << __LINE__ << '
'
#define For(i, a, b) for (typeof(b) i = (a); i <= (b); i++)
#define Ford(i, a, b) for (typeof(a) i = (a); i >= (b); i--)
#define Fit(i, v) for (typeof((v).begin()) i = (v).begin(); i != (v).end(); i++)
#define Fitd(i, v) for (typeof((v).rbegin()) i = (v).rbegin(); i != (v).rend(); i--)
#define all(n) (n).begin(), (n).end()
#define wln(x) cout << x << '
'
#define w(x) cout << x
#define r(x, type) ; type x; cin >> x;
#define rs(x) ; string x; getline(cin, x);
#define rarr(array, i, a, n) ; for (int i = a; i <= n; i++) cin >> array[i];
#define warr(array, i, a, n) ; for (int i = a; i <= n; i++) cout << array[i] << " ";
#define warrd(array, i, a, n) ; for (int i = n; i >= a; i--) cout << array[i] << " ";
#define maxn 100010

int main() {
  // Fast I/O setup
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  
  int n;
  cin >> n;
  
  // 'x' will store the final value, initialized to 0 as per the problem.
  int x = 0;
  
  // Block Logic: Loop through each of the 'n' statements.
  while (n--) {
    string s;
    cin >> s;
    
    /**
     * @brief Core operation logic.
     * This is a clever shortcut. Instead of checking for "++" or "--", it simply
     * checks for the existence of a '-' character anywhere in the 3-character string.
     * If a '-' is found, it's a decrement operation. Otherwise, it must be an increment.
     */
    if (s[1] == '-' || s[0] == '-' || s[2] == '-')
      x--;
    else
      x++;
  }
  
  // Output the final value of x.
  cout << x << endl;
  
  return 0;
}
