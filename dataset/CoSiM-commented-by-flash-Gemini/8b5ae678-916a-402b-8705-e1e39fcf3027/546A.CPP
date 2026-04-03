/**
 * @8b5ae678-916a-402b-8705-e1e39fcf3027/546A.CPP
 * @brief This program calculates the additional money required to purchase a specified number of items
 * with a linearly increasing cost. The cost of the k-th item is k times the initial cost.
 * It handles a special case for purchasing only one item.
 * Algorithm: Arithmetic series sum calculation.
 * Time Complexity: O(1) as it involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) for storing input variables and calculations.
 */

#include <cstdio> // Required for scanf and printf for competitive programming style I/O.

using namespace std;

int main() {
  // @var count: Stores the total cost of all items (or difference for row == 1).
  // @var row: The number of items to be purchased.
  // @var money: The amount of money the person currently has.
  // @var cost: The initial cost of the first item. The k-th item costs k * cost.
  int count, row, money, cost;
  
  // Block Logic: Read the initial cost, available money, and number of items to buy.
  scanf("%d %d %d", &cost, &money, &row);
  
  // Block Logic: Handle the special case where only one item is to be purchased.
  if (row == 1) {
    // Block Logic: If available money is sufficient for one item.
    if (money > cost)
      printf("0"); // No additional money needed.
    // Block Logic: If additional money is needed for one item.
    else
      printf("%d", cost - money); // Print the deficit.
  } 
  // Block Logic: Handle the general case for purchasing multiple items.
  else {
    // Inline: Calculate the sum of the first 'row' natural numbers using the formula n*(n+1)/2.
    // This represents the sum of multipliers for the initial cost.
    count = (row * (row + 1)) / 2;
    // Inline: Calculate the total cost for 'row' items.
    count = count * cost;
    
    // Block Logic: If available money is sufficient for all items.
    if (money > count)
      printf("0"); // No additional money needed.
    // Block Logic: If additional money is needed for all items.
    else
      printf("%d", count - money); // Print the deficit.
  }
  return 0; // Explicitly return 0 for successful execution.
}
