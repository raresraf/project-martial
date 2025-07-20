#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, x = 0;
  cin >> n;

  for (int i = 0; i < n; i++) {
    string inp;
    cin >> inp;
    if (inp.find_first_of('+') != -1)
      x++;
    if (inp.find_first_of('-') != -1)
      x--;
  }

  cout << x;
}
