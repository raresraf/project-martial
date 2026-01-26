#include <bits/stdc++.h>
#define pi 2.0 * acos(0.0)
#define MAX 100005
typedef long long ll;
using namespace std;

int main() {
  int i, j, k, n, count = 0;
  char x[MAX];
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> x;
    if (x[1] == '+')
      count = count + 1;
    else if (x[1] == '-')
      count = count - 1;
  }
  cout << count;
  return 0;
}
