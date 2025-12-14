#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

int main() {
  int n;
  cin >> n;

  vector<int> coins(n);

  for (int i = 0; i < n; i++) {
    cin >> coins[i];
  }

  // assuming i have nothing, and my twin have all the money
  int my_sum = 0;
  int twin_sum = accumulate(coins.begin(), coins.end(), 0);
  ;

  sort(coins.rbegin(), coins.rend());

  int num_of_coins{};
  for (int i = 0; i < n; i++) {
    // taking a coin with the highest value, and subtract the value from my_twin
    my_sum += coins[i];
    twin_sum -= coins[i];
    num_of_coins++;

    if (my_sum > twin_sum) {
      break;
    }
  }

  cout << num_of_coins << endl;
}