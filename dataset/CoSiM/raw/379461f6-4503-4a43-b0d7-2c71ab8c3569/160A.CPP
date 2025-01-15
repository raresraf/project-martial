#include <bits/stdc++.h>
using namespace std;
typedef long long int ll;
#define MOD 1000000007
#define MAX 1000003

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  int n, m, i, j, k, z = 0, sum = 0;
  cin >> n;
  int A[n];
  for (i = 0; i < n; i++) {
    cin >> A[i];
    sum += A[i];
  }
  sort(A, A + n);
  i = n - 1;
  while (i >= 0) {
    z += A[i];
    if (z > (sum / 2)) {
      break;
    }
    i--;
  }
  cout << n - i << endl;
  return 0;
}