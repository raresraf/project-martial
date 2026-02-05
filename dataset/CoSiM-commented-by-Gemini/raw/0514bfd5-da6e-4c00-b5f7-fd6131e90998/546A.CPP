/**
 * @file 546A.CPP
 * @brief Solution for Codeforces Problem 546A, "Soldier and Bananas".
 * @details This program calculates the amount of money a soldier needs to borrow
 * to buy a certain number of bananas. The cost of the bananas increases
 * linearly: the first banana costs k, the second 2k, ..., and the w-th
 * banana costs w*k.
 *
 * @section Algorithm
 * The total cost of `w` bananas forms an arithmetic series:
 * Cost = k * (1 + 2 + 3 + ... + w).
 * The sum of the first `w` integers is given by the formula w * (w + 1) / 2.
 * Therefore, the total cost can be calculated as `k * w * (w + 1) / 2`.
 *
 * The program first calculates this total cost and then compares it with the
 * initial amount of money `n` the soldier has. If the cost is greater than
 * `n`, the difference (cost - n) is the amount that needs to be borrowed.
 * Otherwise, the soldier doesn't need to borrow any money.
 *
 * @section Complexity
 * - Time Complexity: O(W) for the implemented iterative solution. The problem can
 *   also be solved in O(1) using the arithmetic series formula directly.
 * - Space Complexity: O(1), as only a few variables are used.
 */
#ifndef _GLIBCXX_NO_ASSERT
#include <cassert>
#endif
#include <cctype>
#include <cerrno>
#include <cfloat>
#include <ciso646>
#include <climits>
#include <clocale>
#include <cmath>
#include <csetjmp>
#include <csignal>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#if __cplusplus >= 201103L
#include <ccomplex>
#include <cfenv>
#include <cinttypes>
//#include <cstdalign>
#include <cstdbool>
#include <cstdint>
#include <ctgmath>
#include <cwchar>
#include <cwctype>
#endif

// C++
#include <algorithm>
#include <bitset>
#include <complex>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iosfwd>
#include <iostream>
#include <istream>
#include <iterator>
#include <limits>
#include <list>
#include <locale>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ostream>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <typeinfo>
#include <utility>
#include <valarray>
#include <vector>

#if __cplusplus >= 201103L
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <forward_list>
#include <future>
#include <initializer_list>
#include <mutex>
#include <random>
#include <ratio>
#include <regex>
#include <scoped_allocator>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#endif
#define ll long long
#define MAX 100001

using namespace std;

/**
 * @brief Main function to calculate the banana purchasing deficit.
 *
 * Reads the initial banana cost (k), the soldier's money (n), and the
 * number of bananas to buy (w). It then calculates the total cost and
 * determines the amount of money the soldier needs to borrow.
 */
int main() {
  int i, n, t, k, w;
  long long sum = 0;

  cin >> k >> n >> w;
  /**
   * @brief This loop calculates the total cost of `w` bananas.
   * The cost of the i-th banana is `i * k`. The loop sums these costs.
   */
  for (i = 1; i <= w; i++) {
    sum = sum + (i * k);
  }
  /**
   * @brief Determines the amount to borrow.
   * If the total cost (`sum`) is greater than the money the soldier has (`n`),
   * the difference is the required loan. Otherwise, no loan is needed.
   */
  if (sum <= n) {
    cout << 0 << "\n";
  } else
    cout << sum - n << "\n";

  return 0;
}
