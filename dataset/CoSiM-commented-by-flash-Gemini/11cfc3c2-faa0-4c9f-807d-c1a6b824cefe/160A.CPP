#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;
  std::vector<int> v(n);

  for (auto &i : v) {
    cin >> i;
  }

  sort(v.begin(), v.end(), greater<int>());
  size_t sum = accumulate(v.begin(), v.end(), 0);
  sum /= 2;
  int count = 0;
  int coinsum = 0;
  for (int i = 0; i < n; ++i) {
    if (coinsum > sum) {
      break;
    } else {
      coinsum += v[i];
      count++;
    }
  }
  cout << count << '\n';
  return 0;
}
