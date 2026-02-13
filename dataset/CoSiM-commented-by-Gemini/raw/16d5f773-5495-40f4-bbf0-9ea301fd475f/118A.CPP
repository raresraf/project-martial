/**
 * @file 118A.CPP
 * @brief A modern C++ solution for the "String Task" problem using regular expressions.
 * @details This program transforms an input string by removing all vowels and prepending a
 * '.' to the remaining consonants. This implementation uses the C++ <regex>
 * library for vowel removal, showcasing a high-level approach to string manipulation.
 * The code is embedded within a typical competitive programming template.
 *
 * @section algorithm Algorithm
 * 1. The `solve` function reads the input string.
 * 2. It converts the entire string to lowercase in-place using a range-based for loop.
 * 3. A regular expression `[aeiouy]` is defined to match all vowels (including 'y').
 * 4. `std::regex_replace` is used to create a new string where all characters matching the
 *    vowel regex are removed.
 * 5. The program then iterates through the new consonant-only string, printing each
 *    character prepended with a '.'.
 * 6. The `main` function is configured for fast I/O, a common optimization in competitive programming.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: The complexity is dominated by `regex_replace`. For this simple pattern,
 *   it is effectively linear, O(L), where L is the length of the string.
 * - Space Complexity: O(L) to store the intermediate and final strings.
 */
// template
#include <bits/stdc++.h>
using namespace std;
// NOTE: The following is a standard template for competitive programming.
#define gc getchar_unlocked
#define fo(i, n) for (i = 0; i < n; i++)
#define Fo(i, k, n) for (i = k; k < n ? i < n : i > n; k < n ? i += 1 : i -= 1)
#define ll long long
#define si(x) scanf("%d", &x)
#define sl(x) scanf("%lld", &x)
#define ss(s) scanf("%s", s)
#define pi(x) printf("%d\n", x)
#define pl(x) printf("%lld\n", x)
#define ps(s) printf("%s\n", s)
#define deb(x) cout << #x << "=" << x << endl
#define deb2(x, y) cout << #x << "=" << x << "," << #y << "=" << y << endl
#define pb push_back
#define mp make_pair
#define clr(x) memset(x, 0, sizeof(x))
#define PI 3.1415926535897932384626
typedef pair<int, int> pii;
typedef pair<ll, ll> pl;
typedef vector<int> vi;
typedef vector<long> vl;
typedef vector<char> vc;
typedef vector<pii> vpii;
typedef vector<pl> vpl;
typedef vector<vi> vvi;
typedef vector<vl> vvl;

const int mod = 1'000'000'007;
const int N = 3e5, M = N;
//=========end-of-template===========

/**
 * @brief Contains the core logic for solving the String Task problem.
 */
void solve() {
  string s, str;
  cin >> s;
  // First, convert the entire string to lowercase for case-insensitive matching.
  for (auto &c : s)
    c = tolower(c);

  // Define a regular expression to find all vowels (including 'y').
  regex r("[aeiouy]");
  // Use regex_replace to create a new string where all vowels are removed.
  str = regex_replace(s, r, "");

  // Iterate through the resulting string of consonants and print the output.
  for (auto &c : str) {
    cout << '.' << c;
  }
}

int main() {
  // A common competitive programming optimization to speed up C++ streams.
  ios_base::sync_with_stdio(0), cin.tie(0), cout.tie(0);

  solve();

  return 0;
}
