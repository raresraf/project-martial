#include <iostream>

using namespace std;

int main() {
  int n, t;
  string fila;
  cin >> n >> t >> fila;

  while (t--) {
    for (int i = 0; i < n; i++) {
      if (fila[i] == 'B' && fila[i + 1] == 'G') {
        fila[i] = 'G';
        fila[i + 1] = 'B';
        i++;
      }
    }
  }
  cout << fila << endl;

  return 0;
}

// http://codeforces.com/problemset/problem/266/B
