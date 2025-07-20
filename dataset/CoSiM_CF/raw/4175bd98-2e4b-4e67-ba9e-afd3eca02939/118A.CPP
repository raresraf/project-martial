#include <bits/stdc++.h>
#include <math.h>
#define pi 2.0 * acos(0.0)
#define MAX 100005
typedef long long ll;
using namespace std;

int main() {
  ll i, j, l;
  char a[MAX], b[MAX];
  j = 0;
  cin >> a;
  l = strlen(a);
  for (i = 0; i < l; i++) {
    if (tolower(a[i]) != 'a' && tolower(a[i]) != 'e' && tolower(a[i]) != 'i' &&
        tolower(a[i]) != 'o' && tolower(a[i]) != 'u' && tolower(a[i]) != 'y') {
      b[j] = '.';
      j++;
      b[j] = tolower(a[i]);
      j++;
    }
  }
  cout << b;
  return 0;
}
