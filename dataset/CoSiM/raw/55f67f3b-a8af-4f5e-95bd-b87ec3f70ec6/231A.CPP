#include <iostream>

using namespace std;

int main() {
  int n;
  int p, v, t;
  int solve;
  int problems;

  cin >> n;

  problems = 0;
  for (int i = 0; i < n; i++) {
    solve = 0;
    cin >> p >> v >> t;
    if (p) {
      solve++;
    }
    if (v) {
      solve++;
    }
    if (t) {
      solve++;
    }
    if (solve >= 2) {
      problems++;
    }
  }
  cout << problems;
}
