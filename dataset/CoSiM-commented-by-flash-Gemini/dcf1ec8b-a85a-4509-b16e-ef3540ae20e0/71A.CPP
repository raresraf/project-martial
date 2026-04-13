//--------------**************---------------------
/*
 #        "    ""#    ""#                      #
 #   m  mmm      #      #     mmm    mmm    mmm#   mmm
 # m"     #      #      #    #"  "  #" "#  #" "#  #"  #
 #"#      #      #      #    #      #   #  #   #  #""""
 #  "m  mm#mm    "mm    "mm  "#mm"  "#m#"  "#m##  "#mm"  */

/**
 * @file 71A.CPP
 * @brief Solution to a competitive programming problem involving conditional string abbreviation.
 *
 * This program implements a common competitive programming task where strings longer than a
 * specified length (10 characters in this case) are abbreviated to their first character,
 * the count of intermediate characters, and their last character. Shorter strings are
 * printed as is.
 *
 * Algorithm: Conditional string abbreviation.
 * Time Complexity: O(T * N), where T is the number of test cases and N is the maximum string length.
 *                  Each test case involves reading a string, calculating its length, and conditional
 *                  character access/printing, all of which are proportional to the string length.
 * Space Complexity: O(N) to store the input string for each test case.
 */

#include <bits/stdc++.h> // Common header in competitive programming, includes most standard libraries for convenience.

/**
 * @brief Standard namespace usage.
 * In competitive programming, 'using namespace std;' is common practice to avoid
 * repeatedly prefixing standard library elements with 'std::'.
 */
using namespace std;

// Macros for common operations and type definitions, frequently used in competitive programming for brevity.
#define pb push_back   ///< @brief Macro for std::vector::push_back.
#define mp make_pair   ///< @brief Macro for std::make_pair.
typedef pair<int, int> pii; ///< @brief Type definition for a pair of integers.
typedef long long ll;     ///< @brief Type definition for long long integer.
typedef double ld;        ///< @brief Type definition for long double.
typedef vector<int> vi;   ///< @brief Type definition for a vector of integers.
#define fi first       ///< @brief Macro for accessing the first element of a pair.
#define se second      ///< @brief Macro for accessing the second element of a pair.
#define fe first       ///< @brief Duplicate macro for first (likely a typo or alternative from source).
#define SZ 666666      ///< @brief A large constant, possibly for array sizing or other limits.
#define si(n) scanf("%d", &n); ///< @brief Macro for fast input of an integer using scanf.
#define sl(n) scanf("%ld", &n); ///< @brief Macro for fast input of a long integer using scanf.
#define pi(n) printf("%d\n", n); ///< @brief Macro for fast output of an integer using printf.
#define pl(n) printf("%ld\n", n); ///< @brief Macro for fast output of a long integer using printf.
#define pf(n) printf("%f\n", n); ///< @brief Macro for fast output of a float using printf.
#define FILL(a, b) memset(a, 0, sizeof(b)); ///< @brief Macro to fill a memory block with zeros. (Note: sizeof(b) might be unintended if 'b' is a pointer)
#define rep(i, n) for (int i = 0; i < n; i++) ///< @brief Macro for a common for-loop pattern (0 to n-1).
#define reps(i, a, b) for (int i = 1; i <= b; i++) ///< @brief Macro for a common for-loop pattern (a to b inclusive).
const int INF = 1e9 + 5; ///< @brief Constant representing infinity, commonly used in algorithms.
const int MOD = 1000000007; ///< @brief Constant for modulo operation, frequently used in problems involving large numbers.

//--------------**************---------------------

/**
 * @brief Main function: entry point of the program.
 * Handles the overall logic for processing multiple test cases of the string abbreviation problem.
 */
int main() {
  int t; // Declares an integer 't' to store the number of test cases.
  si(t); // Reads the number of test cases using a custom macro for potentially faster input.

  /**
   * @brief Loop through each test case.
   * Block Logic: Iteratively processes 't' distinct problem instances.
   * Invariant: At the start of each iteration, 't' represents the number of remaining test cases to process.
   */
  while (t--) {
    string s; // Declares a string 's' to store the input word for the current test case.
    cin >> s; // Reads the input string.
    int l = s.length(); // Calculates the length of the string.

    /**
     * @brief Conditional block for string abbreviation logic.
     * Block Logic: Determines whether to abbreviate the string based on its length.
     * Pre-condition: 's' contains the current input string, and 'l' holds its length.
     */
    if (l <= 10)
      cout << s << endl; // If string length is 10 or less, print the string as is.
    else {
      // Abbreviate the string: print the first character, then the count of intermediate characters, then the last character.
      // Example: "localization" (length 12) becomes "l10n". The intermediate characters are (l-2).
      cout << s[0] << l - 2 << s[l - 1] << endl;
    }
  }
  return 0; // Indicates successful program execution.
}
