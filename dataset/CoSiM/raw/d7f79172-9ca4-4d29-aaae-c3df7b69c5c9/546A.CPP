#include <iostream>

using namespace std;

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);

  int k, n, w;
  cin >> k >> n >> w;

  int cost = 0;
  for (int i = 1; i <= w; i++) {
    cost += i * k;
  }

  if (cost < n) {
    cout << 0 << "\n";
  } else {
    cout << cost - n << "\n";
  }

  return 0;
}
