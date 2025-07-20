#include <bits/stdc++.h>
using namespace std;

int main() {
  char a[200];
  cin >> a;
  int i, j, n;
  n = 0;
  int l = strlen(a);
  sort(a, a + l);
  for (i = 0; i < l; i++) {
    if (a[i] != a[i + 1])
      n++;
  }
  if (n % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";

  return 0;
}
