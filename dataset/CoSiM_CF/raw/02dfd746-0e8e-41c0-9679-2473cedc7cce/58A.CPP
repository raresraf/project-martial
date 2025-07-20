#include <bits/stdc++.h>
using namespace std;
#define MAX 100005
#define pi acos(-1.0)
typedef long long ll;

int main() {
  ll i, j, k, l, m, f1 = 0, f2 = 0, f3 = 0, f4 = 0, f5 = 0;
  char a[MAX];
  cin >> a;
  for (i = 0; i < strlen(a); i++) {
    if (a[i] == 'h') {
      f1 = 1;
      break;
    }
  }
  for (j = i + 1; j < strlen(a); j++) {
    if (a[j] == 'e') {
      f2 = 1;
      break;
    }
  }
  for (k = j + 1; k < strlen(a); k++) {
    if (a[k] == 'l') {
      f3 = 1;
      break;
    }
  }
  for (l = k + 1; l < strlen(a); l++) {
    if (a[l] == 'l') {
      f4 = 1;
      break;
    }
  }
  for (m = l + 1; m < strlen(a); m++) {
    if (a[m] == 'o') {
      f5 = 1;
      break;
    }
  }
  if (f1 == 1 && f2 == 1 && f3 == 1 && f4 == 1 && f5 == 1)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
