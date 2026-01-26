#include <iostream>

using namespace std;

int main() {
  int n, k;
  int count = 0;
  cin >> n >> k;
  int c[n];
  for (int i = 0; i < n; i++) {
    cin >> c[i];
  }
  for (int i = 0; i < n; i++) {
    while (c[i] > 0) {
      if (c[i] >= c[k - 1]) {
        count++;
        break;
      }
      break;
    }
  }
  cout << count;
  return 0;
}