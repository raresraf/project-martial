#include <iostream>

using namespace std;

int main() {
  int n = 0;
  int k = 0;
  int sum = 0;

  cin >> n >> k;

  int ar[n] = {0};

  for (int i = 1; i <= n; ++i) {
    cin >> ar[i];
  }

  for (int i = 1; i <= n; ++i) {
    if (ar[i] <= 0 || ar[i] < ar[k]) {
      break;
    } else {
      sum += 1;
    }
  }

  cout << endl << sum;

  return 0;
}
