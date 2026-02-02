/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces.
 * @details The program takes a string representing a sum of single-digit numbers
 * and rearranges the numbers into non-decreasing order.
 *
 * Algorithm:
 * 1. Read the input string, which is a sum like "3+2+1".
 * 2. Use a helper function `split` to break the string into a vector of strings
 *    using '+' as the delimiter.
 * 3. Sort the vector of number strings lexicographically. Since the numbers are
 *    single digits, this correctly sorts them numerically.
 * 4. Use a helper function `join` to concatenate the sorted number strings
 *    back into a single string, separated by '+'.
 * 5. Print the resulting sorted sum.
 *
 * Time Complexity: O(N log N), where N is the number of terms in the sum,
 * dominated by the sort operation.
 * Space Complexity: O(L), where L is the length of the input string, to store
 * the vector of strings.
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

/**
 * @brief Splits a string into a vector of strings by a delimiter.
 * @param str The string to be split.
 * @param delim The character to split the string by.
 * @return A vector of strings.
 */
std::vector<std::string> split(const std::string &str, char delim) {
  std::istringstream stream(str);
  std::string element;
  std::vector<std::string> result;
  while (std::getline(stream, element, delim)) {
    result.push_back(element);
  }
  return result;
}

/**
 * @brief Joins a vector of strings into a single string with a delimiter.
 * @param v The vector of strings to be joined.
 * @param delim The delimiter to insert between elements.
 * @return The joined string.
 */
std::string join(const std::vector<std::string> &v,
                 const std::string delim = "") {
  std::string result;
  for (unsigned int i = 0; i < v.size(); ++i) {
    result += v[i];
    if (i != v.size() - 1) {
      result += delim;
    }
  }
  return result;
}

int main() {
  string in;
  cin >> in;
  // The input string is split into a vector of number strings.
  vector<string> v = split(in, '+');
  // The vector is sorted. For single digits, lexicographical sort is equivalent
  // to numerical sort.
  sort(v.begin(), v.end());
  // The sorted vector is joined back into a string with '+' as the delimiter.
  string answer = join(v, "+");
  cout << answer << endl;
  return 0;
}
