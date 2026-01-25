#include <iostream>

using namespace std;

void solve() {
  int n;
  cin >> n;
  int q = 0;
  for (int i = 0; i < n; i++) {
    int a, b, c;
    cin >> a >> b >> c;
    int s = a + b + c;
    if (s > 1) {
      ++q;
    }
  }
  cout << q;
}

int main() {
  solve();
  return 0;
}
