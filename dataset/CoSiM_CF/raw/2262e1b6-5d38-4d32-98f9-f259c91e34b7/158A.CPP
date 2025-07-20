#include <iostream>
using namespace std;
int main() {
  int n, k, c = 0;
  int a[51];

  cin >> n >> k;

  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  for (int i = 0; i < n; i++) {
    if (a[i] >= a[k - 1] && a[i] > 0) {
      c++;
    }
  }
  cout << c << endl;
  /// cout << a[k];

  return 0;
}