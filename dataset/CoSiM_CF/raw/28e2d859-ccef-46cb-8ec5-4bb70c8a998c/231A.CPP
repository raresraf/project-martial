#include <iostream>

using namespace std;

int main() {

  int n;
  int x;
  int ans = 0;

  cin >> n;

  int sum[n] = {0};

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < 3; ++j) {
      cin >> x;
      sum[i] += x;
    }
  }
  cout << endl;
  for (int i = 0; i < n; ++i) {
    if (sum[i] >= 2) {
      ans += 1;
    }
  }

  cout << ans;

  return 0;
}
