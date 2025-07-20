#include <iostream>
using namespace std;

int main() {
  int m, k = 0, sum = 0;

  cin >> m >> k;

  int p[m];

  for (int i = 0; i < m; i++) {
    cin >> p[i];
  }
  for (int i = 0; i < m; i++) {
    if (p[i] >= p[k - 1] && p[i] > 0) {
      sum++;
    }
  }

  cout << sum;
}
