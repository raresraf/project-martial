/**
 * @667bb084-8b2a-4737-aa01-5b84baa69131/753A.CPP
 * @brief This program solves a problem related to partitioning an integer `N`
 * into a sequence of distinct positive integers. The goal is to make the
 * sequence as long as possible, using a greedy approach: taking `1, 2, 3, ...`
 * until the remaining `N` is smaller than the next integer in the sequence,
 * at which point the remainder is added to the last element of the sequence.
 *
 * Algorithm: Greedy decomposition. Iteratively subtracts increasing positive
 * integers (1, 2, 3, ...) from `N` and adds them to a `taken` sequence.
 * If `N` becomes less than the next integer, the remaining `N` is added to
 * the last element of `taken`.
 * Time Complexity: O(sqrt(N)) because the number of distinct integers needed to sum to N
 * is approximately sqrt(2N).
 * Space Complexity: O(sqrt(N)) for storing the `taken` sequence.
 */
#include <iostream> // For standard input/output operations (cin, cout).
#include <vector>   // For using std::vector.
using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to find the longest sequence of distinct positive integers that sum to N.
 * Reads N, greedily constructs the sequence, and prints its size and elements.
 * @return 0 on successful execution.
 */
int main() {
  ios::sync_with_stdio(false); // Functional Utility: Unties C++ standard streams from C standard streams.
  cin.tie(NULL);               // Functional Utility: Prevents `cin` from flushing `cout` before reading input.
                               // These two lines optimize I/O operations for competitive programming.
  int N, orig; // 'N' is the input sum, `orig` stores the original value of N (unused after initialization).
  cin >> N;    // Reads the integer N from standard input.
  vector<int> taken; // Vector to store the distinct positive integers chosen.
  orig = N;          // Stores the original value of N.
  
  /**
   * Block Logic: Greedily selects distinct positive integers (1, 2, 3, ...)
   * to sum up to N, adding them to the `taken` vector. If the remaining `N`
   * is too small for the next integer, it's added to the last `taken` integer.
   * Invariant: `taken` contains a sequence of distinct positive integers that sum to a value
   *            less than or equal to `orig`, and `N` is the remaining sum.
   */
  for (int i = 1; i <= orig; i++) { // Loops with `i` representing the next integer to consider (1, 2, 3, ...).
    if (N < i) { // Pre-condition: If the remaining `N` is less than the current integer `i`.
      taken[taken.size() - 1] += N; // Adds the remaining `N` to the last element of `taken`.
      break;                        // Exits the loop as no more distinct integers can be taken.
    }
    N -= i;        // Subtracts `i` from `N`.
    taken.push_back(i); // Adds `i` to the `taken` vector.
  }
  
  cout << (int)taken.size() << "\n"; // Prints the count of integers in the `taken` sequence.
  cout << taken[0]; // Prints the first element of the `taken` sequence.
  
  /**
   * Block Logic: Prints the remaining elements of the `taken` sequence, separated by spaces.
   * Invariant: All elements of `taken` (except the first, already printed) are printed.
   */
  for (int i = 1; i < (int)taken.size(); i++) // Loops from the second element of `taken`.
    cout << " " << taken[i]; // Prints a space then the current element.
  cout << "\n"; // Prints a final newline.
  return 0; // Indicates successful program termination.
}
