#include <bits/stdc++.h>
#include <math.h>
#define pi 2.0 * acos(0.0)
#define MAX 100005
typedef long long ll;
using namespace std;

int main() {
  char a[MAX];
  int l, i, count;
  count = 0;
  cin >> a;
  l = strlen(a);
  for (i = 0; i < l; i++) {
    if (a[i] == a[i + 1])
      count++;
    if (a[i] != a[i + 1])
      count = 0;
    if (count == 6) {
      cout << "YES";
      return 0;
    }
  }
  cout << "NO";
  return 0;
}
