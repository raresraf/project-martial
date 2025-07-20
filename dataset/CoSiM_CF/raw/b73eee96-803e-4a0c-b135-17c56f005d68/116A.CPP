#include <iostream>

using namespace std;

int main() {
  int n, a, b, max = 0, inside = 0;

  cin >> n;

  for (int i = 0; i < n; ++i) {
    cin >> a >> b;

    inside -= a;

    inside += b;

    if (inside > max) {
      max = inside;
    }
  }

  cout << max << endl;

  return 0;
}