#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int total_sum = 0;
  vector<int> m;
  for (int i = 0; i < n; i++) {
    int temp;
    cin >> temp;
    m.push_back(temp);
    total_sum += temp;
  }

  sort(m.begin(), m.end());

  int sum = 0;
  int coins = 0;
  for (int i = n - 1; i >= 0 && sum <= total_sum - sum; i--) {
    sum += m[i];
    coins++;
  }

  cout << coins;
}
