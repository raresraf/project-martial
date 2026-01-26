#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  string inp;
  scanf("%d", &n);
  getline(cin, inp);
  for (int i = 0; i < n; i++) {
    cin >> inp;
    if (inp.length() > 10)
      cout << inp[0] << inp.length() - 2 << inp[inp.length() - 1] << endl;
    else
      cout << inp << endl;
  }
  return 0;
}