#include <bits/stdc++.h>
#include <math.h>
#define pi 2.0 * acos(0.0)
#define MAX 100005
typedef long long ll;
using namespace std;

int main() {
  int a[MAX], kth, i, n, k, count = 0;
  cin >> n >> k;
  for (i = 0; i < n; i++)
    cin >> a[i];
  kth = a[k - 1];
  for (i = 0; i < n; i++) {
    if (a[i] >= kth && a[i] != 0)
      count++;
  }
  cout << count;
  return 0;
}
