#include <bits/stdc++.h>

using namespace std;

int main() {

  int n, cont = 0;
  char s, aux = ' ';

  cin >> n;

  for (int i = 0; i < n; i++) {
    cin >> s;
    if (s == aux)
      cont++;

    aux = s;
  }

  cout << cont << endl;

  return 0;
}
