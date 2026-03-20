/**
 * @file 546A.CPP
 * @brief Calculates the amount of money needed to buy 'w' bananas,
 *        where the cost of the i-th banana is 'i * k'.
 * Algorithm: Iterative summation of banana costs, followed by a direct comparison
 *            with available money.
 * Time Complexity: O(w) for the loop, where 'w' is the number of bananas.
 * Space Complexity: O(1) as only a few variables are used.
 */
#include <iostream>
#include <cmath> // Required for abs()

using namespace std;

// This define is likely a typo and should be 'typedef long long ll;' or 'using ll = long long;'
// However, adhering to "Zero Code Mutation", we only document its presence.
#define long long ll; 

// Global variables:
// k: Cost of the first banana (and multiplier for subsequent bananas).
// n: Initial amount of money the soldier has.
// w: Number of bananas the soldier wants to buy.
int k, n, w;

/**
 * @brief Solves the problem by calculating the total cost and required money.
 * Functional Utility: Computes the total cost of 'w' bananas and determines
 *                     the deficit if the total cost exceeds the available money.
 * Invariant: 'totalprice' accumulates the cost of bananas from 1 to 'w'.
 */
void solve() {
  int totalprice = 0;
  // Block Logic: Calculates the cumulative cost of buying 'w' bananas.
  // Pre-condition: 'k' and 'w' are positive integers.
  // Invariant: 'totalprice' accurately reflects the sum of costs for 'i' bananas * 'k' at each iteration.
  for (int i = 1; i <= w; i++) {
    totalprice += i * k;
  }
  // Block Logic: Determines the money deficit.
  // If 'n' (money possessed) is greater than 'totalprice', no money is needed (solution is 0).
  // Otherwise, 'solution' represents the amount of money to borrow.
  int solution = n - totalprice;
  if (solution > 0)
    solution = 0;
  
  // Functional Utility: Prints the absolute value of the money to be borrowed.
  // Note: The problem implicitly asks for the amount to *borrow*, which is the absolute deficit.
  cout << abs(solution);
}

/**
 * @brief Main function to handle input and call the solver.
 * Functional Utility: Entry point of the program. Reads input and initiates the solution calculation.
 */
int main() {
  // Block Logic: Reads the problem parameters (initial cost, money, number of bananas).
  cin >> k >> n >> w;
  solve(); // Calls the function to solve the banana cost problem.
}