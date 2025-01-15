// 158A - Next Round

#include <iostream>

using namespace std;

int main() {
  int n, k, i, a[100], c = 0;

  cin >> n >> k;

  for (i = 0; i < n; i++)
    cin >> a[i];

  for (i = 0; i < n; i++)
    if (a[i] >= a[k - 1] && a[i] != 0)
      c++;

  cout << c;

  return 0;
}
