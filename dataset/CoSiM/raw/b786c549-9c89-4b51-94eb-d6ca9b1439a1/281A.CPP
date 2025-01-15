#include <bits/stdc++.h>
using namespace std;
int main() {
  char a[10000];
  cin >> a;
  for (int i = 0; a[i] != '\0'; i++) {
    if (a[0] >= 97 && a[0] <= 122)
      a[0] = a[0] - 32;
  }

  cout << a << endl;
}