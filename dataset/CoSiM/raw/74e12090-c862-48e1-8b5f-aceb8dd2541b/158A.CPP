#include <iostream>
using namespace std;

int main() {
  int n, k, a, total = 0, score;
  cin >> n;
  cin >> k;

  for (int i = 1; i <= n; i++) {
    cin >> a;
    if (i < k) {
      if (a > 0) {
        total++;
      }
    } else if (i == k) {
      if (a > 0) {
        score = a;
        total++;
      }
    } else {
      if (a >= score) {
        total++;
      }
    }
  }
  cout << total;
  return 0;
}
