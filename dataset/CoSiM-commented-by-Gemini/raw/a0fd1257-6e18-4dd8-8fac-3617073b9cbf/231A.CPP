#include <iostream>

using namespace std;

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);

  int n;
  cin >> n;

  int probs = 0;

  for (int i = 0; i < n; i++) {
    int p, v, t;
    cin >> p >> v >> t;

    if (p + v + t >= 2) {
      probs++;
    }
  }

  cout << probs << "\n";

  return 0;
}
