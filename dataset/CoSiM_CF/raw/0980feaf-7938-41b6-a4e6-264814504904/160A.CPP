#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;
  int a[n];
  for (int i = 0; i < n; i++)
    cin >> a[i];
  sort(a, a + n);
  int min, max;
  min = a[0];
  max = a[n - 1];
  int cmax;
  cmax = 1;
  int i, j;
  i = 0;
  j = n - 1;
  for (; (i + 1) < j;) {
    if (min < max) {
      i++;
      min += a[i];
    } else {
      j--;
      max += a[j];
      cmax++;
    }
  }

  if (n != 1) {
    if (n == 2) {
      if (max == min)
        cout << n;
      else
        cout << cmax;
    } else {
      // cout << cmax << i;
      if (max == min)
        cout << cmax + 1;
      else {
        // if((min+a[i]) > max) cout << cmax+1;
        if (min > max)
          cout << cmax + 1;
        else
          cout << cmax;
      }
    }
  } else
    cout << n;
  return 0;
}