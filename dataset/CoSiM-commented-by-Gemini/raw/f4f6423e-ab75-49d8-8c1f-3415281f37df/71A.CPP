/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem.
 * @details This program reads a number of words and, for each word, if it is longer
 * than 10 characters, it prints an abbreviation. The abbreviation consists of the
 * first character, the number of characters in between, and the last character.
 * Otherwise, it prints the word as is. This is a common problem in competitive
 * programming.
 *
 * Algorithm: String abbreviation
 * Time Complexity: O(N * L) where N is the number of words and L is the length of the longest word for input reading. The processing per word is O(1).
 * Space Complexity: O(L) for storing the longest word.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @def faster
 * @brief A macro to speed up C++ I/O operations.
 * @details It unties cin from cout and disables synchronization with C-style I/O.
 */
#define faster ios_base::sync_with_stdio(false), cin.tie(NULL), cout.tie(NULL)
#define mp make_pair
#define mod 1000000007
#define qmod 998244353
#define endl "\n"
#define pb push_back
#define ff first
#define ss second

typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;

const int MOD = 1e9 + 7;
const int INF = 1e9 + 5;
const ll LINF = LLONG_MAX;

int main() {
  faster;
  int n;
  cin >> n;
  /**
   * @brief Main loop to process each of the N words.
   * Pre-condition: n contains the number of words to process.
   * Invariant: The loop iterates n times to handle each word.
   */
  for (int i = 0; i < n; i++) {
    string a;
    cin >> a;
    /**
     * @brief Logic to abbreviate words longer than 10 characters.
     * If the word's length is greater than 10, it is abbreviated.
     * Otherwise, the word is printed as is.
     */
    if (a.size() > 10) {
      cout << a.front() << a.size() - 2 << a.back();
    } else
      cout << a;

    cout << "\n";
  }
}