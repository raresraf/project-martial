/**
 * @file 110A.CPP
 * @brief This competitive programming solution determines if a given number
 *        is considered "nearly lucky". A number is defined as "nearly lucky"
 *        if the count of its lucky digits (4s and 7s) is itself a lucky number (4 or 7).
 * Algorithm: Digit Counting and Conditional Check.
 *            1. Read the input number as a string to easily iterate through its digits.
 *            2. Iterate through each digit of the number, counting occurrences of '4' and '7'.
 *            3. After counting, check two conditions:
 *               a. If there are any lucky digits at all (`lcnt != 0`).
 *               b. If the count of lucky digits (`lcnt`) is either 4 or 7.
 *            4. Output "YES" if both conditions are met, otherwise "NO".
 * Time Complexity: O(L), where L is the number of digits in the input number (length of the string).
 *                  This involves a single pass through the string.
 * Space Complexity: O(L) for storing the input number as a string.
 */
/* samikshamodi - Samiksha Modi */
#include <bits/stdc++.h> // Includes common standard library headers, including iostream and string.
using namespace std;

// Type definitions for convenience, common in competitive programming.
typedef long long ll;     ///< Alias for long long integer.
typedef pair<int, int> pii; ///< Alias for pair of integers.

// Commonly used macros for competitive programming for brevity and convenience.
#define endl '
'          ///< Macro for newline character, often used for faster output than std::endl.
#define pb push_back       ///< Macro for vector's push_back method.
#define mp make_pair       ///< Macro for std::make_pair.
#define f first            ///< Macro for pair's first element.
#define s second           ///< Macro for pair's second element.

// Macros for I/O optimization and common operations.
#define sync                                                                   
  ios_base::sync_with_stdio(false);                                            
  cin.tie(NULL); ///< Macro for fast I/O: unties C++ streams from C std streams and unties cin from cout.
#define out1d(arr, size)                                                       
  {                                                                            
    for (int i = 0; i < size; i++)                                             
      cout << arr[i] << " ";                                                   
  } ///< Macro to print 1D array elements separated by space.
#define in2d(arr, row, col)                                                    
  for (int i = 0; i < row; i++)                                                
    for (int j = 0; j < col; j++)                                              
      cin >> arr[i][j]; ///< Macro to read elements into a 2D array.
#define out2d(arr, row, col)                                                   
  {                                                                            
    for (int i = 0; i < row; i++) {                                            
      for (int j = 0; j < col; j++)                                            
        cout << arr[i][j] << " ";                                              
      cout << endl;                                                            
    }                                                                          
  } ///< Macro to print 2D array elements.
#define mem(arr, val) memset(arr, val, sizeof(arr)); ///< Macro to set memory of an array.

// Constants, common in competitive programming.
const ll MOD = 1e9 + 7; ///< Modulo for operations, typically for large number arithmetic.
const ll INF = 1e18;    ///< Large value representing infinity.
const ll N = 1e5 + 7;   ///< Maximum array size or problem constraint.

/**
 * @brief Main function of the program. Determines if an input number is "nearly lucky".
 * Functional Utility: Serves as the entry point for execution, implementing the core logic
 *                     to classify numbers based on the count of their lucky digits.
 */
int main() {
  sync; // Optimizes C++ standard streams for faster input/output.
  string s; // Declares a string to hold the input number.
  cin >> s; // Reads the number as a string from standard input.
  int lcnt = 0; // Initializes a counter for lucky digits (4s and 7s).

  // Block Logic: Iterates through each character (digit) of the input string.
  // Invariant: `lcnt` accurately reflects the count of '4's and '7's in the processed portion of the string.
  for (int i = 0; i < s.length(); i += 1) {
    // Converts the character digit to an integer and checks if it's 4 or 7.
    // Inline: `s[i] - '0'` converts a character digit (e.g., '4') to its integer value (4).
    if (s[i] - '0' == 4 || s[i] - '0' == 7)
      lcnt++; // Increments the lucky digit counter if the digit is 4 or 7.
  }

  // Block Logic: Checks if the number is "nearly lucky" based on the count of lucky digits.
  // Functional Utility: Applies the problem's specific definition of a "nearly lucky" number.
  // The two conditions are:
  // 1. `lcnt != 0`: There must be at least one lucky digit in the original number.
  // 2. `(lcnt == 4 || lcnt == 7)`: The count of lucky digits itself must be a lucky number.
  if (lcnt != 0 && (lcnt == 4 || lcnt == 7))
    cout << "YES"; // Prints "YES" if the number is nearly lucky.
  else
    cout << "NO";  // Prints "NO" otherwise.

  return 0; // Indicates successful program termination.
}
