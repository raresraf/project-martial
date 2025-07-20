#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  int a[n];
  for (int i = 0; i < n; i++) {
    int x;
    cin >> x >> a[i];
  }
  int x;
  cin >> x;
  int i;
  for (i = 0; i < n; i++) {
    if (x <= a[i])
      break;
  }

  cout << (n - i) << endl;
  return 0;
}