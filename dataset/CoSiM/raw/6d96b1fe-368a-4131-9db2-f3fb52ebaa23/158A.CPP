#include <bits/stdc++.h>

using namespace std;

int main() {

  int n;
  cin >> n;
  int k;
  cin >> k;

  int vector[n];
  int contador = 0;

  for (int i = 0; i < n; i++) {
    cin >> vector[i];
  }

  for (int i = 0; i < n; i++) {
    if (vector[i] != 0) {
      if (vector[i] >= vector[k - 1]) {
        contador++;
      }
    }
  }

  cout << contador;

  return 0;
}
