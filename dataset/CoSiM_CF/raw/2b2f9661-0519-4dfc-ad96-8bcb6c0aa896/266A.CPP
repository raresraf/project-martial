#include <bits/stdc++.h>

using namespace std;

int main() {
  string str;
  int tamanho = 0, i = 0;
  int contador = 0;
  cin >> tamanho;
  cin >> str;
  tamanho = str.length();
  for (i = 0; i < tamanho - 1; i++) {
    if (str[i] == str[i + 1]) {
      contador++;
    }
  }
  cout << contador << endl;

  return 0;
}
