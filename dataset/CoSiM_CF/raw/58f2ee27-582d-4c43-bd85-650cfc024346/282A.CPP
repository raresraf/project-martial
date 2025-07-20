#include <bits/stdc++.h>

using namespace std;

int main() {

  int n;
  int x = 0;
  string s;

  scanf("%d", &n);

  while (n--) {

    cin >> s;

    if (s[0] == '+' || s[2] == '+')
      x++;

    if (s[0] == '-' || s[2] == '-')
      x--;
  }

  cout << x << endl;
}