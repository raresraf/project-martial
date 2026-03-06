/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem using a greedy approach with a priority queue.
 *
 * The problem is to find the minimum number of coins to take from a collection
 * such that the sum of the taken coins is strictly greater than the sum of the
 * remaining coins.
 *
 * @algorithm Greedy approach with a max-heap (std::priority_queue).
 * @details The optimal strategy is to always pick the largest available coin. A
 * `std::priority_queue` is used as a max-heap to efficiently retrieve the largest
 * coin at each step. Coins are taken until the collector's sum exceeds the sum of
 * the remaining coins.
 *
 * @complexity
 * Time: O(N log N), where N is the number of coins. Populating the priority queue
 *       takes N push operations, each O(log N). The subsequent while loop also
 *       involves pop operations, maintaining the complexity.
 * Space: O(N) to store the coins in the priority queue.
 */
#include <iostream>
#include <queue>

// Explicitly using members of the std namespace instead of a full 'using namespace std;'.
using std::cin;
using std::cout;
using std::endl;

/**
 * @brief Main function to handle input and execute the greedy algorithm.
 */
int main() {
  int n;            // The total number of coins.
  int coinsNumber = 0; // Counter for the minimum number of coins to take.
  int twinMoney = 0;   // Starts as the total sum, becomes the sum of remaining coins.
  int myMoney = 0;     // The sum of coins taken.
  
  // A max-heap to store coin values, allowing for efficient access to the largest coin.
  std::priority_queue<int> q;
  
  cin >> n;
  
  // Pre-condition: twinMoney is 0, priority queue is empty.
  // Block Logic: Read all coin values, calculating the total sum (stored in twinMoney)
  // and populating the max-heap.
  for (int val, i = 0; i != n; ++i) {
    cin >> val;
    twinMoney += val;
    q.push(val);
  }

  // Pre-condition: myMoney is 0. twinMoney holds the total sum of all coins.
  // Block Logic: Greedily take the largest coin from the queue and move its value
  // from twinMoney to myMoney until myMoney is strictly greater than twinMoney.
  while (myMoney <= twinMoney) {
    // Get the largest available coin.
    myMoney += q.top();
    // Simultaneously subtract it from the other pile.
    twinMoney -= q.top();
    // Remove the coin from the heap.
    q.pop();
    // Increment the count of coins taken.
    coinsNumber++;
  }

  // Print the minimum number of coins required.
  cout << coinsNumber;
  
  return 0;
}
