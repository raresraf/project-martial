#include <bits/stdc++.h>

using namespace std;

void convert(string &c, int s);

int main() {
  string cadena = {};

  while (cin >> cadena) {
    if (cadena.size() > 1) {
      convert(cadena, cadena.size());
    } else
      cout << cadena << endl;
  }

  return 0;
}

void convert(string &c, int s) {
  vector<int> sumas = {};
  int num_sig = 0, num_ele = 0, j = 0;

  if ((s % 2) == 0) { // si es par, el n mero de "+" es impar. tam/2 +1
    // num_sig = (s/2)+1;
    // num_ele = s - num_sig;
    for (int i = 0; i < s; i + 2) {
      sumas.push_back(c[i]);
    }

    sort(sumas.begin(), sumas.end());

    for (int i = 0; i < s; i + 2) {
      cout << sumas[++j];
      if (i != s - 1)
        cout << "+";
    }
    j = 0;
  } else { // si es impar, el n mero de "+" es par. tam/2 +1
    for (int i = 0; i < s; i + 2) {
      sumas.push_back(c[i]);
    }

    sort(sumas.begin(), sumas.end());

    for (int i = 0; i < s; i + 2) {
      cout << sumas[++j];
      if (i != s - 1)
        cout << "+";
    }
    j = 0;
  }
  cout << endl;
}