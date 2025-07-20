// CodeForces
// Problem 160A [Twins]

#include <bits/stdc++.h>
using namespace std;

int main() {
  long long total_coins;
  cin >> total_coins;

  long long sum = 0;
  vector<long long> coin_values;
  long long temp;
  for (int index = 0; index < total_coins; index++) {
    cin >> temp;
    coin_values.push_back(temp);
    sum += temp;
  }
  sort(coin_values.begin(), coin_values.end());

  long long counter = 1, out_sum = 0;
  for (long long out_index = coin_values.size() - 1; out_index >= 0;
       out_index--) {
    out_sum += coin_values[out_index];
    if (out_sum > sum - out_sum) {
      cout << counter;
      break;
    } else {
      counter++;
    }
  }
  return 0;
}