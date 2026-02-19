/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem by abbreviating long strings.
 * @details This program reads a collection of words, and for any word longer than 10
 * characters, it prints a special abbreviation. Shorter words are printed as-is.
 * This version reads all words into memory before processing them.
 *
 * @algorithm The algorithm is implemented in two main phases:
 * 1. Data Ingestion: All `n` input words are read from standard input and stored
 *    in a `std::vector`.
 * 2. Processing: The program iterates through the vector. For each word, it checks
 *    its length. If the length is greater than 10, an abbreviation is formed using the
 *    first character, the number of middle characters (length - 2), and the last
 *    character. This abbreviation or the original word is then printed.
 *
 * @time_complexity O(L_total), where L_total is the total number of characters across all words,
 * as the time is dominated by I/O operations.
 * @space_complexity O(L_total), as all words are stored in a `std::vector` before processing.
 */
#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Common macros from competitive programming.
#define REP(i, n) for (int i = 0; i < (int)(n); ++i)
#define FOR(i, m, n) for (int i = (m); i < (int)(n); ++i)
#define ALL(x) (x).begin(), (x).end()
#define INF 2000000000

#ifdef LOCAL
#define dprintf(...) fprintf(stdout, __VA_ARGS__)
#else
#define dprintf(...) 0
#endif

using namespace std;

typedef long long LL;
typedef unsigned long long ULL;
typedef unsigned int uint;

const double PI = 3.1415926535897932384626433832795;

int main() {
  int n;
  cin >> n;
  vector<string> words(n);
  // Block Logic: Read all 'n' words from input and store them in a vector.
  REP(i, n) { cin >> words[i]; }
  /**
   * Block Logic: Iterate through the stored words and apply the abbreviation rule.
   */
  REP(i, n) {
    // Pre-condition: Check if the word's length exceeds the threshold of 10.
    if (words[i].size() > 10) {
      // Logic for abbreviation: first char, (length - 2), last char.
      cout << words[i].front() << words[i].size() - 2 << words[i].back()
           << endl;
    } else {
      // Logic for short words: print the word as is.
      cout << words[i] << endl;
    }
  }

  return 0;
}
