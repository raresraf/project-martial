#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;

  int a[n];
  int total = 0;
  int sum = 0;

  for (int i = 0; i < n; i++) {
    cin >> a[i];
    total += a[i];
  }
  sort(a, a + n);

  int count = 0;

  for (int i = n - 1; i >= 0; i--) {
    sum += a[i];
    total -= a[i];
    count++;
    // cout<<"sum = "<<sum<<", total = "<<total<<endl;
    if (sum > total)
      break;
  }

  cout << count << endl;

  return 0;
}
