/**
 * @file 266A.CPP
 * @brief Solves a problem by counting adjacent identical characters in a sequence.
 *
 * This program calculates the minimum number of items to remove from a sequence
 * such that no two adjacent items are the same. This is achieved by counting
 * pairs of consecutive identical characters in a string.
 *
 * @algorithm Single-pass linear scan.
 * @time_complexity O(N), where N is the length of the input string.
 * @space_complexity O(N) for storing the input string.
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

int main(int argc, char const *argv[]) {
  ll number, count = 0;
  std::string word;

  cin >> number >> word;

  /**
   * @block_logic Iterates through the string to find consecutive identical characters.
   * The loop condition `i < number` combined with accessing `word[i + 1]` means the
   * final comparison is `word[number - 1] == word[number]`. Accessing `word[number]` on a
   * std::string of length `number` is safe and yields a null character.
   */
  for (size_t i = 0; i < number; i++) {
    /**
     * @invariant Checks if the character at the current position is the same as the next one.
     * If they match, it signifies a pair that needs one removal, so the counter is incremented.
     */
    if (word[i] == word[i + 1])
      count++;
  }

  // Outputs the total count of characters that need to be removed.
  cout << count << endl;

  return 0;
}
