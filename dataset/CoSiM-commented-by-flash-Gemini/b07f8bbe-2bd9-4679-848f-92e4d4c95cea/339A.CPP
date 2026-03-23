/**
 * @file 339A.CPP
 * @brief This competitive programming solution takes a simple sum expression
 *        (e.g., "3+1+2") consisting only of numbers '1', '2', and '3' separated by
 *        '+' signs. It reorders the numbers in non-decreasing order and then
 *        prints the reordered expression.
 * Algorithm: Parsing, Sorting, and Formatted Output.
 *            1. Reads the input character by character, extracting only the digits.
 *            2. Converts the character digits to integers and stores them in a vector.
 *            3. Sorts the vector of integers in ascending order.
 *            4. Prints the sorted integers, separated by '+' signs.
 * Time Complexity: O(L log N), where L is the length of the input string and N
 *                  is the number of digits in the string. The dominant factor is
 *                  the sorting of numbers.
 * Space Complexity: O(N), for storing the extracted numbers in a vector.
 */
/*
        SubmissionId	:	4353166
        ContestId	:	339
        Index	:	A
        ProblemName	:	Helpful Maths
        ProblemTags	:	['greedy', 'implementation', 'sortings',
   'strings'] ProgrammingLanguage	:	GNU C++ Verdict	:	OK
*/

#include <algorithm> // Required for std::sort.
#include <cmath>     // Included but not used in this specific solution.
#include <cstring>   // Included but not used in this specific solution.
#include <fstream>   // Included but not used in this specific solution.
#include <iomanip>   // Included but not used in this specific solution.
#include <iostream>  // Required for standard input/output (cin, cout).
#include <list>      // Included but not used in this specific solution.
#include <map>       // Included but not used in this specific solution.
#include <queue>     // Included but not used in this specific solution.
#include <set>       // Included but not used in this specific solution.
#include <sstream>   // Included but not used in this specific solution.
#include <vector>    // Required for std::vector.

using namespace std;

// Type definitions for convenience.
typedef long long ll;     ///< Alias for long long integer.
typedef pair<int, int> pii; ///< Alias for pair of integers.
#define X first             ///< Macro for accessing the first element of a pair.
#define Y second            ///< Macro for accessing the second element of a pair.
#define pb push_back        ///< Macro for vector's push_back method.
#define mp make_pair        ///< Macro for std::make_pair.
#define SZ(x) ((int)((x).size())) ///< Macro to get the size of a container.
#define ALL(x) (x).begin(), (x).end() ///< Macro for passing a container's begin/end iterators.
#define debug(x) cerr << #x << " = " << x << endl ///< Debug macro to print variable name and value.
// Loop macros, typical in competitive programming for brevity.
#define FOREACH(i, c)                                                          
  for (__typeof((c).begin()) i = (c).begin(); i != (c).end(); i++) ///< Iterator-based loop.
#define FOR(i, a, n) for (__typeof(n) i = (a); i <= (n); i++) ///< Integer loop from 'a' to 'n' (inclusive).
#define FORD(i, n, a) for (__typeof(n) i = (n); i >= (a); i--) ///< Integer loop from 'n' to 'a' (inclusive, decrementing).
#define MAXN 100010 ///< Constant for maximum array size, often used for global arrays.

vector<int> v; ///< Global vector to store the extracted numbers.
char c;        ///< Global character variable to read input one by one.

/**
 * @brief Main function of the program. Parses the input expression, sorts the numbers, and prints the result.
 * Functional Utility: Serves as the entry point for the "Helpful Maths" problem, performing the core logic.
 */
int main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output.
  // This is a common practice in competitive programming to prevent Time Limit Exceeded errors.
  ios::sync_with_stdio(false); 
  
  // Block Logic: Reads characters from standard input until EOF.
  // Invariant: 'v' accumulates all numeric digits from the input string.
  while (cin >> c) {
    if (c != '+') // Checks if the current character is not a '+' sign.
      v.pb(c - 48); // Converts char to int (e.g., '1' -> 1) and adds to vector.
  }
  sort(ALL(v)); // Sorts the vector 'v' in non-decreasing order.

  // Block Logic: Prints the first element of the sorted vector.
  // Invariant: The first number is printed without a preceding '+'.
  cout << v[0];
  
  // Block Logic: Iterates through the rest of the sorted vector, printing each
  //              number preceded by a '+' sign.
  // Invariant: All numbers from the second onwards are printed with '+' separators.
  FOR(i, 1, SZ(v) - 1)
  cout << "+" << v[i];
  
  return 0; // Indicates successful program termination.
}
