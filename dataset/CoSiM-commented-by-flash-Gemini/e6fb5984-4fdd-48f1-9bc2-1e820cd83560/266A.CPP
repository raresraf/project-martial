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

// C++ Standard Library headers
#include <algorithm> // For algorithms like std::sort, std::min, std::max.
#include <bitset>    // For fixed-size sequence of bits.
#include <complex>   // For complex numbers.
#include <deque>     // For double-ended queue.
#include <exception> // For standard exception classes.
#include <fstream>   // For file stream operations.
#include <functional> // For function objects.
#include <iomanip>   // For output formatting.
#include <ios>       // For base I/O definitions.
#include <iosfwd>    // For forward declarations of I/O objects.
#include <iostream>  // For standard input/output streams (cin, cout).
#include <istream>   // For input stream operations.
#include <iterator>  // For iterator primitives.
#include <limits>    // For numeric limits.
#include <list>      // For doubly-linked list.
#include <locale>    // For localization utilities.
#include <map>       // For associative array (key-value pairs, sorted by key).
#include <memory>    // For memory management utilities.
#include <new>       // For dynamic memory allocation.
#include <numeric>   // For numeric operations (e.g., accumulate).
#include <ostream>   // For output stream operations.
#include <queue>     // For queue data structure.
#include <set>       // For set data structure (unique sorted elements).
#include <sstream>   // For string stream manipulation.
#include <stack>     // For stack data structure.
#include <stdexcept> // For standard exception classes.
#include <streambuf> // For stream buffer operations.
#include <string>    // For string class.
#include <typeinfo>  // For dynamic type identification.
#include <utility>   // For std::pair and other utility classes.
#include <valarray>  // For array of numeric values.
#include <vector>    // For dynamic array.

#if __cplusplus >= 201103L
// C++11 and later headers
#include <array>          // For fixed-size arrays.
#include <atomic>         // For atomic operations.
#include <chrono>         // For time utilities.
#include <condition_variable> // For condition variables.
#include <forward_list>   // For singly-linked list.
#include <future>         // For asynchronous operations.
#include <initializer_list> // For initializer list.
#include <mutex>          // For mutexes.
#include <random>         // For random number generation utilities.
#include <ratio>          // For compile-time rational arithmetic.
#include <regex>          // For regular expressions.
#include <scoped_allocator> // For scoped allocators.
#include <system_error>   // For system errors.
#include <thread>         // For threading support.
#include <tuple>          // For tuples.
#include <type_traits>    // For compile-time type introspection.
#include <typeindex>      // For type indexing.
#include <unordered_map>  // For hash map.
#include <unordered_set>  // For hash set.
#endif

#define ll long long // Alias for long long integer.
#define MAX 100001   // A constant for maximum size or limit.

/**
 * @brief Standard namespace usage.
 * In competitive programming, 'using namespace std;' is common practice to avoid
 * repeatedly prefixing standard library elements with 'std::'.
 */
using namespace std;

/**
 * @file 266A.CPP
 * @brief Solution to a competitive programming problem: "Stones on the Table".
 *
 * This program calculates the minimum number of stones to remove from a row
 * such that no two adjacent stones are of the same color. The problem is solved
 * by iterating through the stones and counting instances where an adjacent pair
 * has identical colors. Each such instance implies one stone must be removed.
 *
 * Algorithm: Linear scan (single pass) with adjacent character comparison.
 * Time Complexity: O(N), where N is the number of stones (length of the string).
 *                  The algorithm iterates through the string once.
 * Space Complexity: O(N) for storing the input string. Additional variables use O(1) space.
 */
int main(int argc, char const *argv[]) {
  ll number;             // Declares 'number' to store the total count of stones (N).
  ll count = 0;          // Declares 'count' to store the number of stones to be removed, initialized to 0.
  std::string word;      // Declares 'word' to store the sequence of stone colors as a string.

  cin >> number >> word; // Reads the total number of stones and the string representing their colors.

  /**
   * @brief Iterates through the stones to identify and count adjacent identical colors.
   * Block Logic: Compares each stone with its immediate successor. If they are of the same color,
   * it increments the `count` of stones to be removed. This assumes that for every pair of
   * identical adjacent stones, one of them must be removed.
   * Pre-condition: `word` contains the sequence of stone colors, `number` is its length.
   * Invariant: `count` accumulates the minimum removals needed up to the current position.
   */
  for (size_t i = 0; i < number - 1; i++) { // Loop iterates up to `number - 1` to safely access `word[i + 1]`.
    if (word[i] == word[i + 1])             // Checks if the current stone and the next stone have the same color.
      count++;                              // If colors are identical, increments the removal count.
  }

  cout << count << endl; // Prints the total minimum number of stones that need to be removed.

  return 0; // Indicates successful program execution.
}
