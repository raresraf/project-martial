#include <iostream>
using namespace std;
int main() {
  int a[50];
  int coun = 0, n, i, k;
  cin >> n >> k;
  for (i = 1; i <= n; i++) {
    cin >> a[i - 1];
  }
  int val = a[k - 1];
  if (a[0] == 0)
    cout << "0";
  else {
    for (int j = 0; j < n; j++) {
      if (a[j] >= val)
        coun++;
    }
    cout << coun;
  }

  return 0;
}
