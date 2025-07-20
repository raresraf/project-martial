#include <iostream>
using namespace std;

int main() {
  int n, P, V, T, problemas = 0;
  cin >> n;

  for (int i = 1; i <= n; ++i) {
    cin >> P;
    cin >> V;
    cin >> T;
    if (P + V + T >= 2) {
      problemas++;
    }
  }

  cout << problemas;

  return 0;
}
