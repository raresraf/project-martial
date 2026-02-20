/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem using a greedy approach.
 *
 * The problem is to find the minimum number of coins to take from a set
 * so that the sum of the taken coins is strictly greater than the sum of the
 * remaining coins.
 *
 * Algorithm:
 * 1. Read all coin values and calculate their total sum.
 * 2. The target is to collect a sum of coins that is strictly greater than
 *    half of the total sum.
 * 3. Sort the coins in descending order.
 * 4. Greedily take the largest coins one by one, keeping track of the running
 *    sum and the number of coins taken.
 * 5. Stop as soon as the running sum exceeds half of the total sum.
 * 6. The number of coins taken is the minimum required.
 *
 * Time Complexity: O(N log N), dominated by the sorting operation, where N is
 * the number of coins.
 * Space Complexity: O(N) to store the coin values.
 */
#include <algorithm>
#include <iostream>
using namespace std;

/**
 * @brief Custom comparison function for sorting in descending order.
 */
bool cmp(int a, int b) { return a > b; }

int main() {
  int n; // Number of coins.
  int c[101]; // Array to store coin values.
  int sum = 0; // Total sum of all coins.
  
  cin >> n;

  // Read all coin values.
  for (int i = 0; i < n; i++)
    cin >> c[i];
  
  // Calculate the total sum of all coins.
  for (int i = 0; i < n; i++)
    sum += c[i];
  ; // Stray semicolon, has no effect.

  // Sort the coins in descending order to enable a greedy approach.
  sort(c, c + n, cmp);

  /**
   * Block Logic: Calculate the target sum to exceed.
   * We need our sum to be strictly greater than the other twin's sum.
   * If our sum is S_our and total is S_total, we need S_our > S_total - S_our,
   * which simplifies to 2 * S_our > S_total, or S_our > S_total / 2.
   * This code calculates the integer value that our sum must exceed.
   */
  if (sum % 2 == 1)
    sum += 1;
  sum /= 2;

  int taken = 0; // Counter for the number of coins taken.
  int inHand = 0; // The running sum of coins taken.
  
  /**
   * Block Logic: Greedily take the largest coins.
   * Iterate through the descending-sorted coins, adding them to `inHand`
   * until the sum `inHand` surpasses the target `sum`.
   */
  for (int i = 0; i < n; i++) {
    inHand += c[i];
    taken++;

    if (inHand > sum)
      break; // Stop as soon as we have taken enough.
  }

  cout << taken << endl;
  return 0;
}
