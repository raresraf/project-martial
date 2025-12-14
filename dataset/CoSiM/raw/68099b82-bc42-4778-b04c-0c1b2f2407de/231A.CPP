#include <iostream>
using namespace std;

int main() {
  int n, a, b, c, cont = 0;
  cin >> n;
  while (n > 0) {
    cin >> a >> b >> c;
    if (a + b + c > 1)
      cont++;
    n--;
  }
  cout << cont;
}
