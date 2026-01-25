#include <iostream>
using namespace std;

int main() {

  int n, k, a[100], t = 0;

  cin >> n >> k;

  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  for (int i = 0; i < n; i++) {
    if (a[i] >= a[k - 1] && a[i] > 0) {

      t++;
    }
  }

  cout << t << endl;

  return 0;
}
