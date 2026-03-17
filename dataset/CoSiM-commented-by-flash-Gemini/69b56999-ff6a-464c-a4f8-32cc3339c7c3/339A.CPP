/**
 * @69b56999-ff6a-464c-a4f8-32cc3339c7c3/339A.CPP
 * @brief Problem: Helpful Maths.
 *        Given a string representing a sum of numbers (e.g., "3+1+2"),
 *        reorder the numbers in non-decreasing order and print the resulting sum.
 * Algorithm: Parse, Sort, and Print.
 *            1. Extract all numbers from the input string. Since the format is "num+num+...",
 *               numbers are at even indices.
 *            2. Store these numbers (as characters) in a dynamic array (vector).
 *            3. Sort the extracted numbers.
 *            4. Print the sorted numbers, re-inserting '+' signs between them.
 * Time Complexity: O(L log L) where L is the number of numbers in the sum.
 *                  Extracting numbers is O(N) where N is string length.
 *                  Sorting `L` numbers takes O(L log L). Printing is O(L).
 *                  Given L <= N/2, this simplifies to O(N log N).
 * Space Complexity: O(L) for storing the extracted numbers in a vector.
 */

/* Monil Soni sonil06 */
/* DA-IICT  */

#include <bits/stdc++.h> // Functional Utility: Includes standard C++ libraries, often used in competitive programming.

using namespace std;

typedef long long LL;     // Functional Utility: Defines LL as a shorthand for long long.
typedef vector<int> vi;   // Functional Utility: Defines vi as a shorthand for vector of ints.
typedef vector<long long> vl; // Functional Utility: Defines vl as a shorthand for vector of long longs.

#define MOD 1000000007 // Functional Utility: Macro for a common modulo value, likely for modular arithmetic.
// Functional Utility: Macros for common loop patterns.
#define _for(i, n) for (int i = 0; i < n; i++) 
#define __for(i, n) for (int i = 1; i <= n; i++)
#define REP(i, s, n) for (int i = (s), _n = (n); i <= _n; i++)
#define FOR(i, n) for (int i = 0, _n = (n); i < _n; i++)
#define REPD(i, e, s) for (int i = (e), _s = (s); i >= _s; i--)
// Functional Utility: Macro for iterating through containers.
#define tr(container, it)                                                      \
  for (typeof(container.begin()) it = container.begin();                       \
       it != container.end(); it++)
#define pb push_back // Functional Utility: Macro for vector's push_back method.
#define mp make_pair // Functional Utility: Macro for creating pairs.
#define fi first     // Functional Utility: Macro for accessing the first element of a pair.
#define se second    // Functional Utility: Macro for accessing the second element of a pair.
#define ALL(x) x.begin(), x.end() // Functional Utility: Macro for passing iterator range to algorithms.

int main() {
  string str;
  // Block Logic: Reads the input sum string (e.g., "3+1+2").
  cin >> str;
  int n = str.length(); // Functional Utility: Stores the length of the input string.
  vector<char> v;      // Functional Utility: A dynamic array to store the extracted numbers as characters.

  // Block Logic: Extracts numbers from the input string.
  // Pre-condition: Input string 'str' is in the format "num+num+...".
  // Invariant: Only characters at even indices (which are numbers) are added to 'v'.
  for (int i = 0; i < n; i += 2) {
    v.pb(str.at(i)); // Functional Utility: Adds the character (number) to the vector.
  }
  // Functional Utility: Sorts the extracted numbers in non-decreasing order.
  // Algorithm: Standard library sort, typically IntroSort (hybrid of QuickSort, HeapSort, InsertionSort).
  sort(v.begin(), v.end());
  
  n = v.size(); // Functional Utility: Updates 'n' to be the count of extracted numbers.
  
  // Block Logic: Prints the first number.
  // Pre-condition: 'v' contains at least one number.
  printf("%c", v[0]);
  
  // Block Logic: Prints the remaining numbers, prefixed with a '+' sign.
  // Pre-condition: 'v' contains more than one number.
  // Invariant: Each subsequent number is printed with a leading '+'.
  for (int i = 1; i < n; i++) {
    printf("+%c", v[i]); // Functional Utility: Prints '+' followed by the number character.
  }
  printf("\n"); // Functional Utility: Prints a newline character at the end.
  return 0;      // Functional Utility: Indicates successful program execution.
}
