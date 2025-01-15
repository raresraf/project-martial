#include <bits/stdc++.h>

using namespace std;

int count(string &c, int n);
void to_lowercase(string &c, int n);
void to_uppercase(string &c, int n);

int main() {
  int n = 0, i = 0;
  string cadena = {};

  cin >> n;
  i = n;

  while (i--) {
    cin >> cadena;
    if (cadena.size() <= 10) {
      cout << cadena << endl;
    } else {
      int red = cadena.size() - 2;
      cout << cadena[0] << red << cadena[cadena.size() - 1] << endl;
    }
  }

  return 0;
}