#include <bits/stdc++.h>

using namespace std;

int bucket(string &c);

int main() {
  string cadena = {};

  while (cin >> cadena) {
    if (bucket(cadena) % 2 == 0)
      cout << "CHAT WITH HER!" << endl;
    else
      cout << "IGNORE HIM!" << endl;
  }
  return 0;
}

int bucket(string &c) {
  int counter = 0;
  vector<int> bucky(26, 0);

  for (int i = 0; i < c.size(); ++i) {
    ++bucky[c[i] - 97];
  }

  for (int i = 0; i < bucky.size(); ++i) {
    if (bucky[i] > 0)
      ++counter;
  }

  return counter;
}