#include <iostream>
using namespace std;
int main() {
  int n, k, a[50], c = 0;
  cin >> n >> k;
  for (int i = 0; i < n; i++)
    cin >> a[i];

  for (int i = 0; i < n; i++) {
    if (a[i] >= a[k - 1] && a[i] > 0)
      c++;
  }
  cout << c;
  return 0;
}
