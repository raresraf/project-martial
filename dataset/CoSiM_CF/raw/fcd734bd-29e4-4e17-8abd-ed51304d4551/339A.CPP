// 09.02.19 @odnaks
#include <iostream>

using namespace std;

int main() {
  long long int n, m, k;
  cin >> n >> m;
  long long int a[100005];
  k = 0;
  long long int j = 1;
  for (long long int i = 0; i < m; i++) {
    cin >> a[i];
    if (a[i] > j) {
      k += (a[i] - j);
      j = a[i];
    } else if (a[i] < j) {
      k += (n - j) + a[i];
      j = a[i];
    }
  }
  cout << k;
  return (0);
}