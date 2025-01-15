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
  int j = n - 1, i = 0;
  int suml = 0, sumr = 0;
  while (j > i && j >= 0) {
    if (suml >= sumr) {
      sumr += a[j];
      j--;
    } else if (suml < sumr) {
      suml += a[i];
      i++;
    }
  }
  // cout<<j<<i<<endl;
  if (j == i && sumr <= suml + a[i]) {
    sumr += a[j];
    j--;
  }
  // cout<<i<<j;
  cout << n - j - 1 << endl;
}