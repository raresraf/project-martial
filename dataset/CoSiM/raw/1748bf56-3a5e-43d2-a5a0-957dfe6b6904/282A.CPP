#include <bits/stdc++.h>
using namespace std;
int main() {
  int init = 0, n;
  char str[4];
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> str;
    if (!strcmp(str, "++X"))
      ++init;
    if (!strcmp(str, "X++"))
      init++;

    if (!strcmp(str, "--X"))
      --init;
    if (!strcmp(str, "X--"))
      init--;
  }
  cout << init << endl;
  return 0;
}
