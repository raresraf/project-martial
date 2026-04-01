/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem from Codeforces (160A).
 * @details The program determines the minimum number of coins to take from a set such that
 * the sum of the taken coins is strictly greater than the sum of the remaining coins.
 *
 * Algorithm: Greedy Approach.
 * 1. Read all coin values into a vector.
 * 2. Calculate the total sum of all coins.
 * 3. To minimize the number of coins taken, always pick the highest-value coins first.
 *    This is achieved by sorting the coins in ascending order.
 * 4. Iterate through the coins in descending order, accumulating their sum and counting them.
 * 5. Stop as soon as the accumulated sum is strictly greater than half of the total sum.
 *    At this point, the sum of taken coins is guaranteed to be greater than the sum of the rest.
 * 6. The number of coins taken up to this point is the answer.
 *
 * Time Complexity: O(N log N) dominated by the sorting step, where N is the number of coins.
 * Space Complexity: O(N) to store the coin values.
 */
#include <algorithm> // sort
#include <iostream>
#include <stdlib.h> //exit
#include <vector>

using namespace std;

int main() {
  int a, b;
  int sum = 0, sum2 = 0, cnt = 0;
  vector<int> ele;
  cin >> a;
  // Read all coin values into the vector 'ele'.
  while (a--) {
    cin >> b;
    ele.insert(ele.end(), b);
  }
  // Calculate the total sum of all coins.
  for (vector<int>::iterator it = ele.begin(); it != ele.end(); ++it)
    sum += *it;
  
  // Sort the coins in non-decreasing order to facilitate the greedy approach.
  sort(ele.begin(), ele.end());
  // for(vector<int>::iterator it=ele.begin(); it!=ele.end();++it) {cout<<*it;}

  /**
   * Block Logic: Greedily take the largest coins until their sum is greater than
   * the sum of the remaining coins.
   * Invariant: `sum2` holds the sum of the `cnt` largest coins.
   */
  for (vector<int>::iterator i = (ele.end() - 1); i >= ele.begin(); i--) {
    // cout<<*i;
    sum2 += (*i); // Add the largest available coin to our share.
    cnt++;        // Increment the count of coins taken.
    
    // Check if our share is now strictly greater than half the total sum.
    if (sum2 > (sum / 2)) {
      break; // If so, we have taken the minimum number of coins required.
    }
  }
  // Output the minimum number of coins needed.
  cout << cnt;
}
