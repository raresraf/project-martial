/**
 * @file 71A.CPP
 * @brief Solution to a string abbreviation problem, commonly known as "Way Too Long Words".
 *
 * This program reads a number of words and, for each word, checks its length.
 * If the word is longer than 10 characters, it is abbreviated by printing its
 * first character, the number of characters between the first and last, and
 * its last character. Otherwise, the word is printed as is.
 *
 * Algorithm:
 * 1. Read the number of test cases, `t`.
 * 2. Loop `t` times:
 *    a. Read the input string.
 *    b. Get its length, `l`.
 *    c. If `l > 10`, print `s[0]`, `l-2`, and `s[l-1]`.
 *    d. Otherwise, print the original string.
 *
 * Time Complexity: O(T * L) where T is the number of test cases and L is the
 * maximum length of a string, dominated by string I/O. The logic for each
 * string is O(1).
 * Space Complexity: O(L) to store the largest string.
 */

//--------------**************---------------------
/*
 #        "    ""#    ""#                      #
 #   m  mmm      #      #     mmm    mmm    mmm#   mmm
 # m"     #      #      #    #"  "  #" "#  #" "#  #"  #
 #"#      #      #      #    #      #   #  #   #  #""""
 #  "m  mm#mm    "mm    "mm  "#mm"  "#m#"  "#m##  "#mm"  */

#include <bits/stdc++.h>
using namespace std;

// These are common macros used in competitive programming for brevity.
#define pb push_back
#define mp make_pair
typedef pair<int, int> pii;
typedef long long ll;
typedef double ld;
typedef vector<int> vi;
#define fi first
#define se second
#define fe first
#define SZ 666666
#define si(n) scanf("%d", &n);
#define sl(n) scanf("%ld", &n);
#define pi(n) printf("%d\n", n);
#define pl(n) printf("%ld\n", n);
#define pf(n) printf("%f\n", n);
#define FILL(a, b) memset(a, 0, sizeof(b));
#define rep(i, n) for (int i = 0; i < n; i++)
#define reps(i, a, b) for (int i = 1; i <= b; i++)
const int INF = 1e9 + 5;
const int MOD = 1000000007;

//--------------**************---------------------

int main() {
  int t;
  si(t); // Read the number of test cases.

  // Loop through each test case.
  while (t--) {
    string s;
    cin >> s;
    int l = s.length();

    // Pre-condition: Check if the word is "too long".
    if (l <= 10) {
      // If the word's length is 10 or less, print it unchanged.
      cout << s << endl;
    } else {
      // If the word is longer than 10 characters, create and print the abbreviation.
      // The abbreviation consists of the first character, the count of
      // characters between the first and the last (l-2), and the last character.
      cout << s[0] << l - 2 << s[l - 1] << endl;
    }
  }
}