#include <algorithm>
#include <iostream>
#include <string.h>

using namespace std;

int main() {
  string conta{};
  cin >> conta;
  sort(conta.begin(),
       conta.end()); // all '+' will be in first place and the numbers sorted.
  for (int i = 0; i < conta.length() - 1; i++) {
    if (conta[i] != '+') {
      cout << conta[i] << '+';
    }
  }
  cout << conta[conta.length() - 1];

  return 0;
}