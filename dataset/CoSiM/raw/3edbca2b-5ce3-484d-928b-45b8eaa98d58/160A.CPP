#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

int main() {
  int t, temp, sum1 = 0, sum2 = 0, len, count = 0;

  cin >> t;

  vector<int> coins;

  for (int i = 0; i < t; i++) {
    cin >> temp;
    sum1 += temp;
    coins.push_back(temp);
  }

  sort(coins.begin(), coins.end());

  len = coins.size();
  // cout<<"Len: "<<len<<endl;
  // cout<<"Sum 1: "<<sum1<<endl;

  for (int i = 0; i < len; i++) {
    sum2 += coins[i];
    // cout<<"Sum 2: "<<sum2<<endl;
    sum1 -= coins[i];
    if (sum2 >= sum1)
      count++;
  }
  cout << count;
}
